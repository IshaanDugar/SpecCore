"""Measure the SpecCore CPU (RV + LLM unit) on the KV260's chip: out-of-context synth + place + route.

  python tools/synth_core.py [period_ns, default 10]     -> build/synth_core/<period>/ reports
  python tools/synth_core.py 8 --netlist-only           -> build/synth_core/8/RV_funcsim.v (no place/route)

Prints achieved clock (1000 / (period - WNS)), LUT / FF / DSP / BRAM use, and the LLM unit's share.
Runs under the Vivado memory guard (tools/memguard.py): SC_MEM_MB default 4096, SC_TIME_S 3600.
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("SC_MEM_MB", "4096")
os.environ.setdefault("SC_TIME_S", "3600")
import sc  # noqa: E402


def cell(txt, name):
    m = re.search(r"\|\s*" + re.escape(name) + r"\s*\|\s*([\d.]+)\s*\|", txt)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("period", nargs="?", default="10", help="clock period in ns (default: 10)")
    parser.add_argument("--netlist-only", action="store_true", help="write a functional netlist after synthesis and exit")
    args = parser.parse_args()
    period = args.period
    out = ROOT / "build" / "synth_core" / period
    out.mkdir(parents=True, exist_ok=True)
    logfile = "vivado_netlist.log" if args.netlist_only else "vivado.log"
    rc, log = sc._vivado("vivado", "-mode", "batch", "-nojournal", "-log", str(out / logfile),
                         "-source", str(ROOT / "board/synth/synth_core.tcl"),
                         "-tclargs", str(out), period, str(ROOT),
                         *(["netlist-only"] if args.netlist_only else []), cwd=out)
    marker = "CORE NETLIST DONE" if args.netlist_only else "CORE IMPL DONE"
    done = [l for l in log.splitlines() if marker in l or "MEMGUARD" in l]
    print("\n".join(done))
    if rc or not any(marker in l for l in done):
        stage = "synthesis" if args.netlist_only else "implementation"
        sys.exit(stage + " failed:\n" + "\n".join(log.splitlines()[-30:]))
    if args.netlist_only:
        netlist = out / "RV_funcsim.v"
        if not netlist.is_file():
            sys.exit(f"missing functional netlist: {netlist}")
        print(f"functional netlist: {netlist}")
        return
    wns = float(re.search(r"WNS (-?[\d.]+)", " ".join(done)).group(1))
    p = float(period)
    util = (out / "util_routed.txt").read_text(errors="replace")
    print(f"achieved clock: {1000 / (p - wns):.1f} MHz (period {p} ns, WNS {wns:+.3f} ns)  [measured, post-route OOC]")
    for k in ("CLB LUTs", "CLB Registers", "DSPs", "Block RAM Tile", "URAM"):
        v = cell(util, k)
        if v is not None:
            print(f"  {k:15s} {v:,.0f}")
    hier = (out / "util_hier.txt").read_text(errors="replace")
    print("hierarchy (first lines of util_hier.txt):")
    print("\n".join(l for l in hier.splitlines() if l.startswith("|") and ("Instance" in l or "RV" in l or "vdotk" in l.lower() or "MCycle" in l))[:3000])


if __name__ == "__main__":
    main()
