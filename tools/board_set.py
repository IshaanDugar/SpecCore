"""The 50-prompt test set on the KV260 (measured, replaces the integer-model estimate).

  python tools/board_set.py build     golden header + firmware per prompt -> build/bset/pNN/ (CPU, ~10 min)
  python tools/board_set.py run       deploy the core app, run all prompts on the board (needs env KV_PW =
                                      the board sudo password, sent on stdin only), restore k26-starter-kits
  python tools/board_set.py report    -> logs/board/set50_summary.txt

Test set = qmodel --spec seed 0: prompts(50, 16, seed 0), 32 new tokens, gamma 2, BP on.
Firmware: -DQR_UNIT -DQR_ACCQ -DRR_BSET -DRR_DRAFT=0 (plain target + spec; each checked against its golden).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "build" / "bset"
LOG = ROOT / "logs" / "board"
N, PLEN, NEW = 50, 16, 32
GB = "C:/Program Files/Git/bin/bash.exe"
KV = (ROOT / "board/kv.sh").as_posix()
RD = "speccore/apps/speccore_core"


def sh(*a, **k):
    return subprocess.run(list(a), cwd=ROOT, capture_output=True, text=True, **k)


# program variants per prompt (same golden): "" both phases, p plain only, s spec only, z neither (setup only).
# Wall time per phase = variant - z; core cycles of the full program stay comparable with every earlier run.
VAR = {"": [], "p": ["-DRR_SPEC=0"], "s": ["-DRR_TARGET=0"], "z": ["-DRR_TARGET=0", "-DRR_SPEC=0"]}


def build():
    for i in range(N):
        d = B / f"p{i:02d}"
        if not (d / "golden.h").exists():
            r = sh(sys.executable, "model/qmodel.py", "--golden-extra", "0", str(PLEN), str(NEW), "golden_bset.h",
                   "--set", str(N), str(i))
            if r.returncode:
                sys.exit(r.stdout + r.stderr)
            d.mkdir(parents=True, exist_ok=True)
            (d / "golden.h").write_text((ROOT / "ref/golden_bset.h").read_text())
        for v, extra in VAR.items():
            dv = B / f"p{i:02d}{v}"
            if (dv / "AA_IROM.mem").exists():
                continue
            (ROOT / "ref/golden_bset.h").write_text((d / "golden.h").read_text())
            r = sh(sys.executable, "tools/sc.py", "fw", "bset_fw", "ref/qref.c", "ref/run_ref.c",
                   "-DQR_UNIT", "-DQR_ACCQ", "-DRR_BSET", "-DRR_DRAFT=0", *extra)
            if r.returncode:
                sys.exit(r.stdout + r.stderr)
            dv.mkdir(parents=True, exist_ok=True)
            for m in ("AA_IROM.mem", "AA_DMEM.mem"):
                (dv / m).write_text((ROOT / "build/bset_fw" / m).read_text())
        print(f"p{i:02d} built", flush=True)


def run():
    pw = os.environ.get("KV_PW")
    if not pw:
        sys.exit("set KV_PW (board sudo password) for this command only")
    r = sh(sys.executable, "tools/board_build.py", "deploy", "core")
    print(r.stdout[-400:])
    sh("tar", "-cf", "build/bset.tar", "-C", "build", "bset")
    print(sh(GB, KV, "put", (ROOT / "build/bset.tar").as_posix(), f"{RD}/bset.tar").stdout[-200:])
    loop = (f"cd /home/ubuntu/{RD} && tar -xf bset.tar && for d in bset/p*; do ln -sf ../../model.img $d/model.img; "
            f"echo \"=== $d\"; python3 core_run.py $d --bp --timeout 300; done")
    try:
        r = subprocess.run([GB, KV, f"sudo -S -p '' sh -c '{loop}'"], input=pw + "\n", cwd=ROOT,
                           capture_output=True, text=True, timeout=7200)
        LOG.mkdir(parents=True, exist_ok=True)
        (LOG / "set50_raw.txt").write_text(r.stdout + r.stderr)
    finally:
        print(sh(GB, KV, "sudo -n /usr/bin/xmutil unloadapp; sudo -n /usr/bin/xmutil loadapp k26-starter-kits").stdout)
    report()


def report():
    txt = (LOG / "set50_raw.txt").read_text()
    rows, fails = [], []
    W_ = {}
    for blk in txt.split("=== ")[1:]:
        name = blk.split()[0]
        w = re.search(r'"core_cycles": (\d+), "wall_cycles": (\d+)', blk)
        if w:
            W_[name.split("/")[-1]] = (int(w.group(1)), int(w.group(2)))
        if not re.fullmatch(r"p\d\d", name.split("/")[-1]):
            continue
        pl = re.search(r"cycles plain target total (\d+)", blk)
        sp = re.search(r"cycles spec total (\d+)", blk)
        st = re.search(r"spec gamma 2: rounds (\d+) accepted (\d+) catchup (\d+)", blk)
        w = re.search(r'"core_cycles": (\d+), "wall_cycles": (\d+)', blk)
        ok = "REF PASS" in blk
        if not (pl and sp and ok):
            fails.append(name)
            continue
        rows.append((name, int(pl.group(1)), int(sp.group(1)), st.groups() if st else ("?",) * 3,
                     int(w.group(1)), int(w.group(2))))
    P, S = sum(r[1] for r in rows), sum(r[2] for r in rows)
    C, W = sum(r[4] for r in rows), sum(r[5] for r in rows)
    sp = sorted(r[1] / r[2] for r in rows)
    out = ["# 50-prompt test set on the KV260 (measured; core cycles; gamma 2, BP on, 16 + 32 tokens)", "",
           f"prompts REF PASS: {len(rows)}/{N}" + (f"  FAILED: {', '.join(fails)}" if fails else ""),
           f"sum plain target {P:,} / sum spec {S:,} = {P / S:.3f}x",
           f"per prompt: min {sp[0]:.3f}x, median {sp[len(sp) // 2]:.3f}x, max {sp[-1]:.3f}x, "
           f"slower than plain: {sum(x < 1 for x in sp)}" if sp else "",
           f"real DDR: wall cycles / core cycles = {W / C:.3f} (whole program)", "",
           "| prompt | plain | spec | speedup | rounds | accepted | catch-up |", "| --- | --- | --- | --- | --- | --- | --- |"]
    ids = [n.split("/")[-1] for n, *_ in rows]
    if all(f"{k}{v}" in W_ for k in ids for v in "psz"):
        wp = sum(W_[k + "p"][1] - W_[k + "z"][1] for k in ids)
        ws = sum(W_[k + "s"][1] - W_[k + "z"][1] for k in ids)
        cp = sum(W_[k + "p"][0] - W_[k + "z"][0] for k in ids)
        cs = sum(W_[k + "s"][0] - W_[k + "z"][0] for k in ids)
        out[6:6] = [f"WALL TIME (real DDR, plain-only / spec-only programs minus setup-only): "
                    f"plain {wp:,} / spec {ws:,} = {wp / ws:.3f}x (same programs in core cycles: {cp / cs:.3f}x; "
                    f"DDR overhead plain {wp / cp - 1:+.1%}, spec {ws / cs - 1:+.1%})", ""]
    out += [f"| {n} | {p:,} | {s:,} | {p / s:.3f}x | {a} | {b} | {c} |" for n, p, s, (a, b, c), _, _ in rows]
    (LOG / "set50_summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out[:7]))


if __name__ == "__main__":
    {"build": build, "run": run, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
