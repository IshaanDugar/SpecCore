"""Build and deploy a SpecCore board app (bitstream + device-tree overlay) for the KV260.

  python tools/board_build.py build hello     Vivado batch build under the memory guard, then bootgen
  python tools/board_build.py deploy hello    copy to the board, compile the overlay there, install, load
  python tools/board_build.py test hello      run the app's board test (board/<app>/<app>_test.py)

Memory: Vivado runs inside the same Windows Job Object as every other Vivado call (tools/memguard.py).
Cap SC_MEM_MB (default here 4096: a small KV260 build, estimate 2.5-3.5 GB), SC_TIME_S default 3600.
Board access: board/kv.sh (access guide settings). Root steps use only the narrow sudo rule
(speccore-install-app, xmutil, speccore-devperm).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("SC_MEM_MB", "4096")
os.environ.setdefault("SC_TIME_S", "3600")
import sc  # noqa: E402

# Git Bash by full path: a bare "bash" from Python resolves to System32/bash.exe (WSL) first.
GIT_BASH = "C:/Program Files/Git/bin/bash.exe"


def kv(*args, check=True):
    r = subprocess.run([GIT_BASH, (ROOT / "board/kv.sh").as_posix(), *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(out)
    if check and r.returncode:
        sys.exit(f"board command failed (rc {r.returncode})")
    return out


def dts_regs(dts):
    """[(name, addr, size)] of the generic-uio nodes in an overlay source."""
    txt = dts.read_text(encoding="utf-8")
    return [(m.group(1), int(m.group(2), 16), int(m.group(4), 16))
            for m in re.finditer(r"(\w+)@([0-9a-f]+)\s*\{[^}]*?reg\s*=\s*<0x0\s+0x([0-9a-f]+)\s+0x0\s+0x([0-9a-f]+)>", txt)]


def build(app):
    src, out = ROOT / "board" / app, ROOT / "build" / "board" / app
    out.mkdir(parents=True, exist_ok=True)
    tcl = src / f"build_{app}.tcl"
    rc, log = sc._vivado("vivado", "-mode", "batch", "-nojournal", "-log", str(out / "vivado.log"),
                         "-source", str(tcl), "-tclargs", str(out), cwd=out)
    tail = "\n".join(log.splitlines()[-25:])
    if rc or "HELLO BUILD DONE" not in log and "BUILD DONE" not in log:
        sys.exit(f"vivado failed (rc {rc}):\n{tail}")
    print("\n".join(l for l in log.splitlines() if "MEMGUARD" in l or "BUILD DONE" in l))
    # address map vs overlay
    amap = (out / "address_map.txt").read_text().split("\n")
    vivado = {int(a, 16): int(r, 16) for _, a, r in (l.split() for l in amap if l.strip())}
    for name, addr, size in dts_regs(src / f"speccore_{app}.dts"):
        if vivado.get(addr) != size:
            sys.exit(f"overlay node {name}@{addr:#x} size {size:#x} does not match Vivado map {vivado}")
    print(f"address map matches the overlay: {', '.join(f'{a:#x}+{r:#x}' for a, r in vivado.items())}")
    # bitstream -> .bit.bin for the FPGA manager
    (out / "bit.bif").write_text("all:\n{\n  [destination_device = pl] hello.bit\n}\n".replace("hello", app))
    rc, log = sc._vivado("bootgen", "-image", "bit.bif", "-arch", "zynqmp", "-process_bitstream", "bin", "-w", cwd=out)
    binf = out / f"{app}.bit.bin"
    if rc or not binf.exists():
        sys.exit(f"bootgen failed:\n{log[-2000:]}")
    timing = (out / "timing.txt").read_text(errors="replace")
    wns = re.search(r"WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)", timing)
    print(f"bitstream {binf.stat().st_size:,} B; setup WNS {wns.group(1) if wns else '?'} ns")


def deploy(app):
    name, out, src = f"speccore_{app}", ROOT / "build" / "board" / app, ROOT / "board" / app
    d = f"speccore/apps/{name}"
    kv("put", (out / f"{app}.bit.bin").as_posix(), f"{d}/{name}.bit.bin")
    kv("put", (src / f"{name}.dts").as_posix(), f"{d}/{name}.dts")
    kv("put", (ROOT / "board/shell.json").as_posix(), f"{d}/shell.json")
    kv(f"cd ~/{d} && dtc -@ -q -I dts -O dtb -o {name}.dtbo {name}.dts && ls -l")
    kv(f"sudo -n /usr/local/sbin/speccore-install-app {name}")
    kv("sudo -n /usr/bin/xmutil unloadapp", check=False)
    kv(f"sudo -n /usr/bin/xmutil loadapp {name}")
    kv("sudo -n /usr/local/sbin/speccore-devperm")
    kv("for u in /sys/class/uio/uio*; do echo \"$u $(cat $u/name) $(cat $u/maps/map0/addr) $(cat $u/maps/map0/size)\"; done")


def test(app):
    t = ROOT / "board" / app / f"{app}_test.py"
    kv("put", (t).as_posix(), f"speccore/apps/speccore_{app}/{t.name}")
    kv(f"python3 ~/speccore/apps/speccore_{app}/{t.name}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "deploy", "test"):
        sys.exit(__doc__)
    {"build": build, "deploy": deploy, "test": test}[sys.argv[1]](sys.argv[2])
