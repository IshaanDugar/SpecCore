"""SpecCore task runner. Reads speccore.toml (the single source of every decision).

  python tools/sc.py show   decisions + derived numbers
  python tools/sc.py gen    write firmware/include/sc_config.h and ext/rtl/sc_config.vh
  python tools/sc.py fw NAME SRC...      build C/asm for the core -> build/NAME/AA_IROM.mem, AA_DMEM.mem, .dis
  python tools/sc.py sim NAME [--gate NETLIST.v] [--bp] [--max N] [--vcd] [--ddr]
                                         run build/NAME in xsim (UART output, SVA count in RTL;
                                         --vcd writes build/NAME/sim.vcd for WaveQ; --ddr loads the model image
                                         checkpoints/export/model.img at 0x2000_0000)
  fw / native / run also take -DNAME=VAL (C defines); run takes --ddr, --max N, --bp and --gate NETLIST.v
  python tools/sc.py native NAME SRC...  build + run the same C on the PC (WSL gcc) -> build/NAME/native.txt
  python tools/sc.py run NAME SRC...     fw + sim + native, then diff core output vs PC output
  python tools/sc.py isa    generate + run the RV32IM self-checking test (tools/gen_isa_test.py, BP off and on)

Other scripts: `from sc import load` after adding tools/ to sys.path.
New commands get added here as each part of the project is built.
"""
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = ("fixed", "bimodal", "gshare", "perceptron", "tage")
OPC_CUSTOM0 = 0x0B


def params(m, vocab):
    """Parameter count: tied embedding + per layer (attention 4*d^2, MLP 2*d*h, 2 norms) + final norm."""
    d, h = m["dim"], m["mlp_mult"] * m["dim"]
    return vocab * d + m["layers"] * (4 * d * d + 2 * d * h + 2 * d) + d


def streamed_bytes(m, vocab, bits):
    """Weight bytes read per pass: every matrix once, output head = embedding table. Norms ignored."""
    d, h = m["dim"], m["mlp_mult"] * m["dim"]
    return (vocab * d + m["layers"] * (4 * d * d + 2 * d * h)) * bits // 8


def load(path=ROOT / "speccore.toml"):
    with open(path, "rb") as f:
        c = tomllib.load(f)
    errs = []
    v = c["tokenizer"]["vocab"]
    for name in ("draft", "target"):
        m = c[name]
        if m["dim"] % m["heads"]:
            errs.append(f"{name}: dim {m['dim']} not divisible by heads {m['heads']}")
        m["head_dim"] = m["dim"] // m["heads"]
        m["hidden"] = m["mlp_mult"] * m["dim"]
        m["params"] = params(m, v)
        m["bytes"] = streamed_bytes(m, v, c["quant"]["weight_bits"])
        m["cycles"] = m["bytes"] // (c["hw"]["weight_port_bits"] // 8)
    s, hw, isa = c["spec"], c["hw"], c["isa"]
    s["K"] = s["gamma"] + 1
    if s["K"] > hw["lanes"]:
        errs.append(f"K = gamma+1 = {s['K']} exceeds hw.lanes = {hw['lanes']}")
    if any(k > hw["lanes"] for k in s["k_sweep"]):
        errs.append(f"k_sweep {s['k_sweep']} exceeds hw.lanes = {hw['lanes']}")
    if s["predictor"] not in PREDICTORS:
        errs.append(f"spec.predictor must be one of {PREDICTORS}")
    if hw["attention"] == "kv_stream" and not isa["setsrc"]:
        errs.append("hw.attention = kv_stream needs isa.setsrc = true (O1)")
    if isa["vdotk4"] and c["quant"]["weight_bits"] != 4:
        errs.append("isa.vdotk4 needs quant.weight_bits = 4")
    vj = ROOT / c["tokenizer"]["dir"] / "vocab.json"
    if vj.exists():
        n = len(json.loads(vj.read_text(encoding="utf-8")))
        if n != v:
            errs.append(f"tokenizer {vj} has {n} entries, config says {v}")
    else:
        errs.append(f"missing tokenizer file {vj}")
    if errs:
        sys.exit("speccore.toml problems:\n  " + "\n  ".join(errs))
    return c


def model_cfg(c, name):
    """Model hyperparameters as plain kwargs for model.py."""
    m = c[name]
    return dict(vocab=c["tokenizer"]["vocab"], dim=m["dim"], n_layers=m["layers"],
                n_heads=m["heads"], ctx=m["ctx"], mlp_mult=m["mlp_mult"])


def show(c):
    d, t, s, hw = c["draft"], c["target"], c["spec"], c["hw"]
    mhz = hw["clock_mhz"]
    print("Decisions (speccore.toml)")
    for sec in ("tokenizer", "quant", "spec", "hw", "isa"):
        items = {k: v for k, v in c[sec].items() if not isinstance(v, dict) and k != "K"}
        print(f"  [{sec}] " + ", ".join(f"{k}={v}" for k, v in items.items()))
    print("\nDerived (estimates from the formulas in tools/sc.py, not measurements)")
    print(f"  {'':8}{'params':>10}{'MB/pass':>9}{'cycles/pass':>13}{'ms/pass':>9}")
    for name, m in (("draft", d), ("target", t)):
        print(f"  {name:8}{m['params']/1e6:>9.3f}M{m['bytes']/1e6:>9.2f}{m['cycles']:>13,}{m['cycles']/mhz/1e3:>9.2f}")
    cost = d["bytes"] / t["bytes"]
    a, g = 0.7, s["gamma"]
    tok = (1 - a ** (g + 1)) / (1 - a)
    print(f"  draft cost per guess = {cost:.3f} target passes; K = {s['K']}")
    print(f"  speedup at alpha {a} (assumed), gamma {g}: {tok / (1 + g * cost):.2f}x (weight stream only, ignores attention)")


def gen(c):
    t, d, s, hw, isa = c["target"], c["draft"], c["spec"], c["hw"], c["isa"]
    head = "GENERATED from speccore.toml by `python tools/sc.py gen`. Edit the toml, not this file."
    C = [f"/* {head} */", "#ifndef SC_CONFIG_H", "#define SC_CONFIG_H", ""]
    V = [f"// {head}", "`ifndef SC_CONFIG_VH", "`define SC_CONFIG_VH", ""]

    def both(name, val, vfmt=None):
        C.append(f"#define SC_{name} {val}")
        V.append(f"`define SC_{name} {vfmt if vfmt is not None else val}")

    C.append(f"#define SC_VOCAB {c['tokenizer']['vocab']}")
    C.append(f"#define SC_EOT_ID {c['tokenizer']['eot_id']}")
    for name, m in (("DRAFT", d), ("TARGET", t)):
        for k in ("dim", "layers", "heads", "head_dim", "hidden", "ctx"):
            C.append(f"#define SC_{name}_{k.upper()} {m[k]}")
    C.append(f"#define SC_GAMMA {s['gamma']}")
    C.append(f"#define SC_K {s['K']}")
    for i, p in enumerate(PREDICTORS):
        C.append(f"#define SC_PRED_{p.upper()} {i}")
    C.append(f"#define SC_PREDICTOR SC_PRED_{s['predictor'].upper()}")
    C.append(f"#define SC_MMIO_CYCLES 0x{c['mmio']['cycle_counter']:08X}u")
    C.append(f"#define SC_CLOCK_MHZ {hw['clock_mhz']}")
    C.append(f"#define SC_KV_V_TRANSPOSED {int(hw['kv_v_transposed'])}")
    C.append("")
    both("LANES", hw["lanes"])
    both("WEIGHT_PORT_BITS", hw["weight_port_bits"])
    both("MUL_DSP", int(hw["mul"] == "dsp"))
    both("OPC_CUSTOM0", f"0x{OPC_CUSTOM0:02X}", f"7'b{OPC_CUSTOM0:07b}")
    for k, v in c["isa"]["funct3"].items():
        both(f"F3_{k.upper()}", v, f"3'd{v}")
    for k in ("vdotk", "setsrc", "accq", "vresolve", "vdotk4"):
        both(f"ISA_{k.upper()}", int(isa[k]))
    C += ["", "#endif", ""]
    V += ["", "`endif", ""]
    outs = {ROOT / "firmware/include/sc_config.h": C, ROOT / "ext/rtl/sc_config.vh": V}
    for p, lines in outs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")


# ---------------- firmware build + simulation ----------------
import os
import shutil
import subprocess

WSL = ["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-c"]
VIVADO_BIN = Path(os.environ.get("SC_VIVADO_BIN", r"C:\AMDDesignTools\2025.2\Vivado\bin"))
RV_FLAGS = ("-march=rv32im -mabi=ilp32 -O2 -ffreestanding -nostdlib -nostartfiles "
            "-fno-builtin -Wall -T firmware/link.ld -Ifirmware/include -DSC_ON_CORE")
IROM_BYTES, DMEM_BYTES = 65536, 1 << 20
# Verilog: the CG3207 core (kept as Verilog). SystemVerilog: everything new (decision 12 Nov), incl. testbenches.
V_SRC = sorted((ROOT / "core/rtl").glob("*.v")) + [ROOT / "core/rtl/nexys/Wrapper.v"]
SV_SRC = sorted((ROOT / "ext/rtl").glob("*.sv")) + [ROOT / "core/tb/tb_sc.sv"]
RTL = V_SRC + SV_SRC


def wsl_path(p):
    p = Path(p).resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def wsl(cmd):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    r = subprocess.run(WSL + [f"cd '{wsl_path(ROOT)}' && {cmd}"], capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit(f"WSL command failed:\n{cmd}\n{r.stdout}{r.stderr}")
    return r.stdout


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def words_hex(data):
    data += b"\0" * (-len(data) % 4)
    return "\n".join(f"{int.from_bytes(data[i:i + 4], 'little'):08x}" for i in range(0, len(data), 4)) + "\n"


def fw(name, srcs, defs=""):
    out = ROOT / "build" / name
    out.mkdir(parents=True, exist_ok=True)
    o = rel(out)
    files = " ".join(["firmware/crt0.S", "firmware/lib.c"] + [rel(s) for s in srcs])
    pre = "riscv64-unknown-elf-"
    wsl(f"{pre}gcc {RV_FLAGS} {defs} -o {o}/{name}.elf {files} -lgcc && "
        f"{pre}objcopy -O binary -j .text {o}/{name}.elf {o}/text.bin && "
        f"{pre}objcopy -O binary -j .data {o}/{name}.elf {o}/data.bin && "
        f"{pre}objdump -d -M no-aliases,numeric {o}/{name}.elf > {o}/{name}.dis")
    sizes = wsl(f"{pre}size -A {o}/{name}.elf")
    sec = {l.split()[0]: int(l.split()[1]) for l in sizes.splitlines() if l.startswith(".")}
    text, data, bss = (out / "text.bin").read_bytes(), (out / "data.bin").read_bytes(), sec.get(".bss", 0)
    (out / "AA_IROM.mem").write_text(words_hex(text))
    (out / "AA_DMEM.mem").write_text(words_hex(data))
    print(f"fw {name}: text {len(text)}/{IROM_BYTES} B, data {len(data)} + bss {bss} B, "
          f"stack room {DMEM_BYTES - len(data) - bss} B")
    if len(text) > IROM_BYTES or len(data) + bss > DMEM_BYTES:
        sys.exit("does not fit in IROM/DMEM")
    return out


def _vivado(tool, *args, cwd, secs=None):
    """Run a Vivado tool inside a Windows Job Object with a HARD commit cap (tools/memguard.py).
    On 14/15 Nov an xsimk.exe committed ~103 GB (RAM + pagefile) and froze the laptop.
    SC_MEM_MB (default 2048) = cap on the whole process tree; Windows refuses allocations past it.
    SC_TIME_S / secs = wall-clock limit. Refuses to start with < SC_HEADROOM_MB (default 6000) of
    system commit headroom. Returns (rc, log); rc 99 = stopped by the guard."""
    import tempfile
    from memguard import preflight, run_limited
    mem_mb = int(os.environ.get("SC_MEM_MB", "2048"))
    secs = secs or (float(os.environ["SC_TIME_S"]) if "SC_TIME_S" in os.environ else None)
    preflight(int(os.environ.get("SC_HEADROOM_MB", "6000")))
    if tool == "xelab":
        args = ("--mt", "4", *args)                   # default is 18 threads: keep the laptop usable
    with tempfile.TemporaryFile("w+b") as out:
        rc, why, peak = run_limited([str(VIVADO_BIN / f"{tool}.bat"), *args], cwd=cwd, stdout=out,
                                    mem_mb=mem_mb, secs=secs)
        out.seek(0)
        log = out.read().decode("utf-8", errors="replace")
    msg = f"MEMGUARD: {tool} peak commit {peak} MB (cap {mem_mb} MB)" + (f", STOPPED: {why}" if why else "")
    if why or tool == "xsim":
        print(msg, file=sys.stderr)
    return rc, log + "\n" + msg


def ddr_mem(simdir):
    """AA_DDR.mem (hex words) from checkpoints/export/model.img, rebuilt only when the image is newer."""
    img, mem = ROOT / "checkpoints/export/model.img", simdir / "AA_DDR.mem"
    if not img.exists():
        sys.exit("no model image: run python model/export.py")
    if not mem.exists() or img.stat().st_mtime > mem.stat().st_mtime:
        mem.write_text(words_hex(img.read_bytes()))


def _module_names(path, skip_glbl=False):
    """Read declarations without loading a netlist; optionally skip its guarded GLBL block."""
    declaration = re.compile(r"^\s*module\s+(?:(?:automatic|static)\s+)?(\\\S+|[A-Za-z_$][\w$]*)")
    names, depth = set(), 0
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if skip_glbl and re.match(r"\s*`ifndef\s+GLBL\b", line):
                depth += 1
                continue
            if depth:
                if re.match(r"\s*`if(?:n?def)?\b", line):
                    depth += 1
                elif re.match(r"\s*`endif\b", line):
                    depth -= 1
                continue
            if m := declaration.match(line):
                names.add(m.group(1).lstrip("\\"))
    return names


def _gate_sources(netlist, glbl, simdir):
    """Compile the synthesized RV with only its behavioral simulation shell."""
    wrapper, tb = ROOT / "core/rtl/nexys/Wrapper.v", ROOT / "core/tb/tb_sc.sv"
    supplied = _module_names(netlist, skip_glbl=True)
    if "RV" not in supplied:
        sys.exit(f"gate netlist must define the synthesized top module RV: {netlist}")
    seen = set(supplied)
    for source in (wrapper, tb, glbl):
        modules = _module_names(source)
        if overlap := seen & modules:
            sys.exit(f"duplicate gate simulation modules in {source}: {', '.join(sorted(overlap))}")
        seen.update(modules)
    # All modules in core/rtl/*.v and ext/rtl/*.sv belong to the synthesized core.
    # Compare declarations for diagnostics, but exclude them even if synthesis flattened/renamed them.
    rtl_modules = set()
    for source in RTL:
        if source not in (wrapper, tb):
            rtl_modules.update(_module_names(source))
    print("gate: netlist/RTL module overlap: " + ", ".join(sorted(supplied & rtl_modules))
          + "; all core/extension RTL excluded")
    # Wrapper's memories, DDR model and MMIO peripherals are inline; it instantiates only RV.
    # write_verilog fixes PC_INIT to RV's default (0x00400000, also Wrapper's value) and removes parameters.
    # Named ports also tolerate any port reordering in the generated netlist.
    ports = ("CLK", "RESET", "BP_Enable", "Instr", "ReadData_in", "MemRead", "MemWrite_out",
             "PC", "ALUResult", "WriteData_out", "WS_Addr", "WS_Data")
    pattern = r"\bRV\s*#\s*\(\s*\.PC_INIT\s*\(\s*PC_INIT\s*\)\s*\)\s*RV1\s*\(([^;]+)\);"
    text = wrapper.read_text(encoding="utf-8")
    instance = re.search(pattern, text)
    if instance is None or tuple(p.strip() for p in instance.group(1).split(",")) != ports:
        sys.exit("gate: expected Wrapper's RV1 parameter override and positional port map")
    replacement = "RV RV1(\n" + ",\n".join(f"\t.{p}({p})" for p in ports) + "\n);"
    text, count = re.subn(pattern, replacement, text)
    if count != 1:
        sys.exit("gate: expected one RV #(.PC_INIT(PC_INIT)) RV1 instance in Wrapper.v")
    gate_wrapper = simdir / "Wrapper_gate.v"
    gate_wrapper.write_text(text, encoding="utf-8")
    return [gate_wrapper, glbl], [tb]


def sim(name, bp=False, max_cycles=5_000_000, vcd=False, ddr=False, gate=None):
    """Cache RTL and gate snapshots separately; retain all run plusargs and output paths."""
    sim_name = os.environ.get("SC_SIMDIR", "xsim")   # gate always gets its own work library
    simdir = ROOT / "build" / (sim_name + "_gate" if gate is not None else sim_name)
    snapshot = "tb_sc_gate_sim" if gate is not None else "tb_sc_sim"
    signature = None
    if gate is not None:
        netlist = Path(gate).resolve()
        glbl = Path(os.environ.get("XILINX_VIVADO") or VIVADO_BIN.parent) / "data/verilog/src/glbl.v"
        inputs = [netlist, glbl, ROOT / "core/rtl/nexys/Wrapper.v", ROOT / "core/tb/tb_sc.sv", Path(__file__)]
        for source in inputs:
            if not source.is_file():
                sys.exit(f"missing gate simulation source: {source} (glbl.v uses XILINX_VIVADO or SC_VIVADO_BIN)")
        # Metadata only: do not hash the potentially large netlist on every run.
        signature = json.dumps({"vivado_bin": str(VIVADO_BIN), "inputs": [
            [str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns] for f in inputs]}, sort_keys=True)
    simdir.mkdir(parents=True, exist_ok=True)
    stamp = simdir / "snapshot.stamp"
    rebuild = (not stamp.exists() or stamp.read_text() != signature) if gate is not None else (
        not stamp.exists() or max(f.stat().st_mtime for f in RTL) > stamp.stat().st_mtime)
    if rebuild:
        v_src, sv_src = _gate_sources(netlist, glbl, simdir) if gate is not None else (V_SRC, SV_SRC)
        defines = ["-d", "GATE"] if gate is not None else []
        libraries = ["glbl", "-L", "unisims_ver", "-L", "secureip"] if gate is not None else []
        print("xsim: compiling " + ("gate netlist ..." if gate is not None else "RTL ..."))
        if gate is not None:
            # An interrupted/failed rebuild must not leave a reusable signature.
            stamp.write_text("")
            # Some Vivado exports embed a guarded glbl. Suppress that copy, then compile the
            # installation's glbl.v separately without GLBL defined, so exactly one is linked.
            rc, log = _vivado("xvlog", *defines, "-d", "GLBL", str(netlist), cwd=simdir)
            if rc: sys.exit(log)
        rc, log = _vivado("xvlog", *defines, *[str(f) for f in v_src], cwd=simdir)
        if rc: sys.exit(log)
        rc, log = _vivado("xvlog", "-sv", *defines, *[str(f) for f in sv_src], cwd=simdir)
        if rc: sys.exit(log)
        rc, log = _vivado("xelab", "tb_sc", *libraries, "-s", snapshot, "-timescale", "1ns/1ps", cwd=simdir)
        if rc: sys.exit(log)
        if signature is not None:
            stamp.write_text(signature)
        else:
            stamp.touch()
    for m in ("AA_IROM.mem", "AA_DMEM.mem"):
        shutil.copy(ROOT / "build" / name / m, simdir / m)
    args = [snapshot, "-R", "-testplusarg", f"maxcyc_{max_cycles}"] + (["-testplusarg", "bp"] if bp else [])
    args += ["-testplusarg", "vcd"] if vcd else []
    if ddr:
        ddr_mem(simdir); args += ["-testplusarg", "ddr"]
    rc, log = _vivado("xsim", *args, cwd=simdir)
    if vcd:
        shutil.move(simdir / "sim.vcd", ROOT / "build" / name / "sim.vcd")
    # keep only what the program printed plus the tb lines
    body = log[log.find("[tb] start"):] if "[tb] start" in log else log
    body = body[:body.find("$finish called")] if "$finish called" in body else body
    out = "\n".join(l for l in body.splitlines() if not l.startswith(("run -all", "exit", "INFO:")))
    (ROOT / "build" / name / ("sim_bp.txt" if bp else "sim.txt")).write_text(out + "\n")
    print(out)
    return out


def native(name, srcs, defs=""):
    out = ROOT / "build" / name
    out.mkdir(parents=True, exist_ok=True)
    o = rel(out)
    files = " ".join(rel(s) for s in srcs)
    txt = wsl(f"gcc -O2 -Wall -Ifirmware/include {defs} -o {o}/{name}_native {files} && ./{o}/{name}_native")
    (out / "native.txt").write_text(txt)
    return txt


def _program_lines(sim_out):
    return [l for l in sim_out.splitlines() if not l.startswith(("[tb]", "[cov]"))]


def run(name, srcs, defs="", ddr=False, max_cycles=5_000_000, bp=False, gate=None):
    fw(name, srcs, defs)
    core = _program_lines(sim(name, bp=bp, ddr=ddr, max_cycles=max_cycles, gate=gate))
    pc = native(name, srcs, defs).splitlines()
    keep = lambda ls: [l for l in ls if not l.startswith("cycles")]   # timing lines differ by design
    same = keep(core) == keep(pc)
    print(f"core vs PC output: {'IDENTICAL' if same else 'DIFFERENT'} ({len(keep(pc))} lines compared)")
    if not same:
        import difflib
        print("\n".join(difflib.unified_diff(keep(pc), keep(core), "pc", "core", lineterm="")))
        sys.exit(1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    a = sys.argv[2:]
    defs = " ".join(x for x in a if x.startswith("-D"))           # e.g. -DRR_TARGET=0
    mx = int(a[a.index("--max") + 1]) if "--max" in a else 5_000_000
    if "--gate" in a and (a.index("--gate") + 1 == len(a) or a[a.index("--gate") + 1].startswith("--")):
        sys.exit("--gate requires a functional RV netlist path")
    gate = a[a.index("--gate") + 1] if "--gate" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("-") and (i == 0 or a[i - 1] not in ("--max", "--gate"))]
    if cmd in ("show", "gen"):
        {"show": show, "gen": gen}[cmd](load())
    elif cmd == "fw":
        fw(pos[0], pos[1:], defs)
    elif cmd == "sim":
        sim(pos[0], bp="--bp" in a, max_cycles=mx, vcd="--vcd" in a, ddr="--ddr" in a, gate=gate)
    elif cmd == "native":
        print(native(pos[0], pos[1:], defs), end="")
    elif cmd == "run":
        run(pos[0], pos[1:], defs, ddr="--ddr" in a, max_cycles=mx, bp="--bp" in a, gate=gate)
    elif cmd == "uvm":
        # standalone UVM testbench of the LLM unit: ext/tb/tb_vdotk_uvm.sv
        d = ROOT / "build" / "uvm"
        d.mkdir(parents=True, exist_ok=True)
        seed = a[a.index("--seed") + 1] if "--seed" in a else "1"
        n = a[a.index("--n") + 1] if "--n" in a else "2000"
        srcs = [str(ROOT / "ext/rtl/sc_vdotk.sv"), str(ROOT / "ext/tb/tb_vdotk_uvm.sv")]
        for tool, args in (("xvlog", ["-sv", "-L", "uvm", *srcs]),
                           ("xelab", ["tb_vdotk_uvm", "-L", "uvm", "-s", "uvm_sim", "-timescale", "1ns/1ps"]),
                           ("xsim", ["uvm_sim", "-R", "--sv_seed", seed, "-testplusarg", f"n_{n}"]
                                    + (["-testplusarg", f"vcd_{a[a.index('--vcd') + 1]}"] if "--vcd" in a else []))):
            rc, log = _vivado(tool, *args, cwd=d, secs=120 if tool == "xsim" else 300)
            if rc or tool == "xsim":
                print("\n".join(l for l in log.splitlines()
                                if any(k in l for k in ("UVM_ERROR", "UVM_FATAL", "SB", "COV", "PROG", "UVM PASS", "UVM FAIL", "ERROR", "Error", "GUARD"))))
            if rc == 99:
                print("--- last log lines ---\n" + "\n".join(log.splitlines()[-15:]))
            if rc:
                sys.exit(1)
        sys.exit(0 if "UVM PASS" in log else 1)
    elif cmd == "isa":
        subprocess.run([sys.executable, str(ROOT / "tools/gen_isa_test.py")], check=True)
        fw("isa", [ROOT / "build/gen/isa_test.S"])
        res = [sim("isa"), sim("isa", bp=True)]
        ok = all("EXIT code 0 " in r and "SVA failures: 0" in r for r in res)
        print("ISA TEST", "PASS (BP off and on)" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    else:
        sys.exit(f"unknown command {cmd}; see the docstring at the top of tools/sc.py")
