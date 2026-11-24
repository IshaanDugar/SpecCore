"""xsim run of the board top (board/core/sc_board.v + tb_board.sv) on a firmware build.

  python tools/board_sim.py NAME [--bp] [--max N]    uses build/NAME/AA_IROM.mem, AA_DMEM.mem + the model image

Same memory guard as every Vivado call (tools/sc.py _vivado). Output: build/board_sim/NAME.txt
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sc  # noqa: E402

SIM = ROOT / "build" / "board_sim"
V = sorted((ROOT / "core/rtl").glob("*.v")) + [ROOT / "board/core/sc_board.v"]
SV = sorted((ROOT / "ext/rtl").glob("*.sv")) + [ROOT / "board/core/tb_board.sv"]


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    name, bp = a[0], "--bp" in a
    mx = int(a[a.index("--max") + 1]) if "--max" in a else 3_000_000
    SIM.mkdir(parents=True, exist_ok=True)
    stamp = SIM / "snapshot.stamp"
    if not stamp.exists() or max(f.stat().st_mtime for f in V + SV) > stamp.stat().st_mtime:
        glbl = sc.VIVADO_BIN.parent / "../data/verilog/src/glbl.v"
        for tool, args in (("xvlog", [str(f) for f in V] + [str(glbl.resolve())]),
                           ("xvlog", ["-sv"] + [str(f) for f in SV]),
                           ("xelab", ["tb_board", "glbl", "-L", "xpm", "-L", "unisims_ver", "-s", "tbb",
                                      "-timescale", "1ns/1ps"])):
            rc, log = sc._vivado(tool, *args, cwd=SIM)
            if rc:
                sys.exit(log[-3000:])
        stamp.touch()
    for m in ("AA_IROM.mem", "AA_DMEM.mem"):
        shutil.copy(ROOT / "build" / name / m, SIM / m)
    sc.ddr_mem(SIM)
    args = ["tbb", "-R", "-testplusarg", f"maxcyc_{mx}"] + (["-testplusarg", "bp"] if bp else [])
    rc, log = sc._vivado("xsim", *args, cwd=SIM)
    out = SIM / f"{name}{'_bp' if bp else ''}.txt"
    out.write_text(log)
    print("\n".join(l for l in log.splitlines() if l.startswith("[tbb]") or "PASS" in l or "FAIL" in l
                    or "ERROR" in l or "Error" in l)[-4000:])


if __name__ == "__main__":
    main()
