"""Results page: headline numbers + three charts, every number read from a file (MEASURED or labelled ESTIMATE).

  1. Roofline of the LLM unit (target model): roof = what the unit can do per cycle at K tokens,
     points = what a pass achieves (matvec only, whole pass). MEASURED K-sweep.
  2. Speedup vs plain decoding: measured end-to-end run, calibrated model, what-if hardware cuts
     (ESTIMATES from tools/cyclemodel.py on the measured costs).
  3. Chip resources by module (Vivado post-route utilisation, MEASURED).

  python tools/chart_results.py  ->  artifacts/charts/results.html
Inputs: build/ksweep_accq_bp_core.txt, build/ref_accq_kd_bp_core.txt + build/ref_seed{11,7}_core.txt, build/synth_core/8/*.txt, speccore.toml.
"""
import html
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import cyclemodel as cm  # noqa: E402

KSWEEP = ROOT / "build/ksweep_accq_bp_core.txt"            # 22 Nov: ACCQ + branch predictor
E2E = ROOT / "build/ref_accq_kd_bp_core.txt"               # golden prompt, distilled draft
E2E_MORE = [ROOT / "build/ref_seed11_core.txt", ROOT / "build/ref_seed7_core.txt"]
# integer-model acceptance, 50 prompts x 32 tokens (logs/qspec_int_50.txt)
INT_TOK = {1: 1.778, 2: 2.381, 3: 2.888, 4: 3.200}
INT_CATCHUP = {1: 0.75, 2: 0.59, 3: 0.47, 4: 0.38}
SYN = ROOT / "build/synth_core/8"
OUT = ROOT / "artifacts/charts/results.html"
WS_BYTES_PER_CYCLE = 4          # weight port: one 32-bit word per cycle (sc_vdotk.sv, WS_GAP = 0)
MACS_PER_LANE = 4               # 4 int8 multipliers per lane
LANES = 8


def roofline():
    c = tomllib.loads((ROOT / "speccore.toml").read_text(encoding="utf-8"))
    macs = sum(r * k for r, k in cm.matrices(cm.model_from(c, "target")))   # MACs per token in matvecs
    sweep, ctx = cm.read_ksweep(KSWEEP)
    pts = []
    for K in sorted(sweep["target"]):
        b = sweep["target"][K]
        pts.append(dict(K=K, roof=min(WS_BYTES_PER_CYCLE * K, MACS_PER_LANE * LANES),
                        matvec=K * macs / b["matvec"], whole=K * macs / b["total"],
                        mv_cyc=b["matvec"], tot_cyc=b["total"]))
    return pts, macs, ctx


def speedups():
    sweep, _ = cm.read_ksweep(KSWEEP)
    cal = cm.calibrate(sweep)
    tok = INT_TOK
    cm.MEASURED_CATCHUP.clear(); cm.MEASURED_CATCHUP.update(INT_CATCHUP)
    gs = [g for g in sorted(tok) if g in cm.MEASURED_CATCHUP]
    txt = E2E.read_text(errors="replace")
    plain = int(re.search(r"cycles plain target total (\d+)", txt).group(1))
    spec = int(re.search(r"cycles spec total (\d+)", txt).group(1))
    global PER_STEP
    PER_STEP = int(re.search(r"cycles target per step: .*? total (\d+)", txt).group(1))
    rows = [dict(label="Core run, golden prompt, γ2", sub="16-token prompt, 8 new tokens",
                 v=plain / spec, g=2, kind="measured")]
    tp, ts = plain, spec
    for f, nm in zip(E2E_MORE, ("seed 11", "seed 7")):
        t2 = f.read_text(errors="replace")
        p2 = int(re.search(r"cycles plain target total (\d+)", t2).group(1))
        s2 = int(re.search(r"cycles spec total (\d+)", t2).group(1))
        rows.append(dict(label=f"Core run, prompt {nm}, γ2", sub="8-token prompt, 16 new tokens", v=p2 / s2, g=2, kind="measured"))
        tp += p2; ts += s2
    global SPD_SUM
    SPD_SUM = tp / ts
    names = {"measured (no change)": ("Today, 50 prompts, best γ", "integer acceptance × measured pass costs"),
             "matvec per-lane (readout+fill) -80%": ("Readout + bank fill −80%", "per-token matvec work in hardware"),
             "readout+fill + norm/quant -80%": ("Readout + fill and norm/quant −80%", ""),
             "all per-lane -80%": ("All per-token work −80%", ""),
             "draft pass -50% (smaller draft)": ("Draft model 2× cheaper", "no hardware change"),
             "all per-lane -80% + draft -50%": ("All per-token −80% + draft 2× cheaper", "")}
    for label, cuts in cm.WHATIF:
        if label not in names:
            continue
        cs = cm.cal_scaled(cal, cuts)
        best = max(gs, key=lambda g: cm.cal_speedup(cs, g, tok))
        t, s = names[label]
        rows.append(dict(label=t, sub=s, v=cm.cal_speedup(cs, best, tok), g=best, kind="estimate"))
    return rows, plain, spec


def resources():
    util = (SYN / "util_routed.txt").read_text(errors="replace")
    lut_total = int(re.search(r"\|\s*CLB LUTs\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)", util).group(1))
    lut_avail = int(re.search(r"\|\s*CLB LUTs\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)", util).group(1))
    hier = (SYN / "util_hier.txt").read_text(errors="replace")
    mods = {}
    for line in hier.splitlines():
        m = re.match(r"\|\s{3}(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|", line)
        if m:
            mods[m.group(1)] = int(m.group(3))
    nice = {"LLM1": "LLM unit (VDOTK)", "MCycle1": "Divider (MCycle)", "BranchPredictor1": "Branch predictor",
            "RegFile1": "Register file", "ProgramCounter1": "Program counter"}
    rows = [dict(label=nice.get(k, k), v=v) for k, v in mods.items() if k in nice]
    rows.append(dict(label="Rest of pipeline (see note)", v=lut_total - sum(r["v"] for r in rows)))
    rows.sort(key=lambda r: -r["v"])
    wns = float(re.search(r"WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)", (SYN / "timing_routed.txt").read_text(errors="replace"), re.S).group(1))
    dsp = int(re.search(r"\|\s*DSPs\s*\|\s*(\d+)", util).group(1))
    return rows, lut_total, lut_avail, 1000 / (8.0 - wns), dsp


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SpecCore: results</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ee; --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e69;
  --grid: #e6e5e0; --axis: #b9b8b2; --ink: #0b0b0b;
  --series-1: #2a78d6; --series-2: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #242423; --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --grid: #2e2e2c; --axis: #55544f; --ink: #f0efec;
    --series-1: #3987e5; --series-2: #d95926;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #242423; --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --grid: #2e2e2c; --axis: #55544f; --ink: #f0efec;
  --series-1: #3987e5; --series-2: #d95926;
}
body { margin: 0; }
.viz-root { background: var(--surface-1); color: var(--text-primary); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 28px 32px 40px; max-width: 920px; }
h1 { font-size: 22px; font-weight: 650; margin: 0 0 4px; }
h2 { font-size: 17px; font-weight: 600; margin: 34px 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 14px; max-width: 780px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0 6px; }
.tile { background: var(--surface-2); border-radius: 8px; padding: 12px 14px; }
.tile .n { font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .l { color: var(--text-secondary); font-size: 12.5px; }
.badge { display: inline-block; font-size: 11px; color: var(--text-muted); border: 1px solid var(--axis); border-radius: 4px; padding: 0 5px; margin-left: 4px; vertical-align: 1px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; color: var(--text-secondary); font-size: 13px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 12px; border-radius: 3px; display: inline-block; }
.legend i.line { height: 2px; border-radius: 1px; }
svg text { fill: var(--text-secondary); font-size: 12px; }
svg .val { fill: var(--text-primary); font-weight: 600; paint-order: stroke; stroke: var(--surface-1); stroke-width: 4px; stroke-linejoin: round; }
.tip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--axis); border-radius: 6px; padding: 8px 10px; font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,.15);
  display: none; min-width: 180px; max-width: 300px; }
.tip b { display: block; margin-bottom: 2px; }
.tip .m { color: var(--text-secondary); }
.note { color: var(--text-muted); font-size: 12.5px; margin-top: 8px; max-width: 800px; }
button { font: inherit; font-size: 13px; color: var(--text-primary); background: transparent; border: 1px solid var(--axis);
  border-radius: 6px; padding: 4px 10px; cursor: pointer; margin-top: 10px; }
table { border-collapse: collapse; margin-top: 8px; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
</style></head>
<body><div class="viz-root">
<h1>SpecCore: what was measured</h1>
<p class="sub">Speculative decoding on a custom RISC-V core with an LLM unit, for the AMD Kria KV260. Cycle counts from cycle-accurate
simulation (Vivado xsim) of the real RTL; clock and area from Vivado place-and-route on the KV260's chip.</p>
<div class="tiles">
  <div class="tile"><div class="n">__SPD__×</div><div class="l">speedup vs plain decoding, γ2, Σ over 3 prompts <span class="badge">measured</span><br>per prompt 1.11-1.37×, none slower</div></div>
  <div class="tile"><div class="n">__CPT__M</div><div class="l">cycles per target token, plain (ctx 23) <span class="badge">measured</span></div></div>
  <div class="tile"><div class="n">__MHZ__ MHz</div><div class="l">core + LLM unit clock on xck26 <span class="badge">measured</span></div></div>
  <div class="tile"><div class="n">__LUTP__%</div><div class="l">of the chip's LUTs (__LUT__ of __LUTA__) <span class="badge">measured</span></div></div>
</div>

<h2>1. The LLM unit does far less work per cycle than it could</h2>
<p class="sub">Roofline for one target-model pass that checks K tokens. Each streamed weight byte feeds K multiply-adds, so the
roof (one 32-bit weight word per cycle into K lanes of 4 multipliers) rises with K, up to 32 MACs/cycle at 8 lanes.
The achieved points barely rise: the per-token work around the multiplies (readout, bank fill, attention, norms) grows with K too.</p>
<div class="legend"><span><i class="line" style="background:var(--ink)"></i>Roof (unit peak)</span>
<span><i style="background:var(--series-1)"></i>Achieved, matrix-vector part</span>
<span><i style="background:var(--series-2)"></i>Achieved, whole pass</span></div>
<svg id="roof" role="img" aria-label="Roofline: achieved multiply-adds per cycle vs tokens per pass" width="860" height="330"></svg>
<div class="note">Measured (xsim, branch predictor on, context __CTX__): __MACS__ multiply-adds per token in the matrix-vector products.
Log-log axes. Source: build/ksweep_accq_bp_core.txt.</div>
<button data-t="t1" aria-expanded="false">Show table</button><div id="t1" hidden></div>

<h2>2. ACCQ + a better draft get above 1× on the core; the rest is what-if</h2>
<p class="sub">Speedup of speculative decoding over plain decoding on the same core. The first three bars are full measured runs on the core (three prompts); the rest are
estimates from the cycle model: measured pass costs × guess acceptance from the bit-exact integer model on 50 prompts (the same model matched two earlier end-to-end runs within 1.2%).
Each estimate shows its best γ (guesses per round).</p>
<div class="legend"><span><i style="background:var(--series-1)"></i>Measured</span>
<span><i style="background:repeating-linear-gradient(135deg,var(--series-1) 0 3px,transparent 3px 6px);border:1px solid var(--series-1)"></i>Estimate</span>
<span><i class="line" style="background:var(--ink)"></i>1× = plain decoding</span></div>
<svg id="spd" role="img" aria-label="Speedup: measured and estimated" width="860" height="340"></svg>
<div class="note">The what-if rows apply the same saving to plain decoding, so they are not "verify pass only" gains.
Which of these to build is the student's decision (D13 to D15 in docs/review/decisions.md). Sources: build/ref_accq_kd_bp_core.txt, build/ref_seed{11,7}_core.txt,
tools/cyclemodel.py --calib build/ksweep_accq_bp_core.txt with integer-model acceptance (logs/qspec_int_50.txt).</div>
<button data-t="t2" aria-expanded="false">Show table</button><div id="t2" hidden></div>

<h2>3. The whole CPU fits in about 6% of the chip</h2>
<p class="sub">LUTs by module after place-and-route on the KV260's xck26 (8 ns target met: __MHZ__ MHz), core + LLM unit only
(no memories, no board shell). The LLM unit's 32 int8 multipliers are built from LUTs; the chip's __DSP__ DSP blocks used are the core's mul.</p>
<svg id="res" role="img" aria-label="LUTs by module" width="860" height="250"></svg>
<div class="note">Per-module split is approximate: Vivado flattens and rebuilds the hierarchy, so some pipeline logic is credited
to other modules; the total is exact. Source: build/synth_core/8/util_routed.txt, util_hier.txt.</div>
<button data-t="t3" aria-expanded="false">Show table</button><div id="t3" hidden></div>
<div class="tip" id="tip" role="tooltip"></div>
</div>
<script>
const ROOF = __ROOF__, SPD = __SPDROWS__, RES = __RES__;
const NS = 'http://www.w3.org/2000/svg';
const el = (p, t, a = {}, st = {}) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]);
  Object.assign(e.style, st); p.appendChild(e); return e; };
const tip = document.getElementById('tip');
const show = (ev, h) => { tip.innerHTML = h; tip.style.display = 'block';
  const r = tip.getBoundingClientRect(); let px = ev.clientX + 14, py = ev.clientY + 14;
  if (px + r.width > innerWidth - 8) px = ev.clientX - r.width - 14; if (py + r.height > innerHeight - 8) py = ev.clientY - r.height - 14;
  tip.style.left = px + 'px'; tip.style.top = py + 'px'; };
const hide = () => tip.style.display = 'none';
const hov = (e, h) => { e.addEventListener('mousemove', ev => show(ev, h)); e.addEventListener('mouseleave', hide); };
const M = v => (v / 1e6).toFixed(2) + 'M';

// ---- 1. roofline (log-log) ----
(() => {
  const svg = document.getElementById('roof'), W = 860, H = 330, L = 60, R = 200, T = 12, B = 42;
  const lx = v => L + (W - L - R) * Math.log2(v / 0.8) / Math.log2(16 / 0.8);
  const ly = v => T + (H - T - B) * (1 - Math.log2(v / 0.75) / Math.log2(48 / 0.75));
  [1, 2, 4, 8, 16].forEach(k => { el(svg, 'line', {x1: lx(k), x2: lx(k), y1: T, y2: H - B}, {stroke: 'var(--grid)'});
    el(svg, 'text', {x: lx(k), y: H - B + 18, 'text-anchor': 'middle'}).textContent = k; });
  [1, 2, 4, 8, 16, 32].forEach(v => { el(svg, 'line', {x1: L, x2: W - R, y1: ly(v), y2: ly(v)}, {stroke: 'var(--grid)'});
    el(svg, 'text', {x: L - 8, y: ly(v) + 4, 'text-anchor': 'end'}).textContent = v; });
  el(svg, 'text', {x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle'}).textContent = 'K = tokens per pass = multiply-adds per weight byte';
  el(svg, 'text', {x: 14, y: (T + H - B) / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${(T + H - B) / 2})`}).textContent = 'multiply-adds per cycle';
  // roof: 4K up to 8 lanes, then flat at 32
  el(svg, 'path', {d: `M${lx(0.8)},${ly(3.2)} L${lx(8)},${ly(32)} L${lx(16)},${ly(32)}`, fill: 'none', 'stroke-width': 2}, {stroke: 'var(--ink)'});
  el(svg, 'text', {x: lx(16) + 6, y: ly(32) + 4}).textContent = '32: all 8 lanes';
  el(svg, 'text', {x: lx(2.6), y: ly(10.4) - 8, 'text-anchor': 'end'}).textContent = 'roof = 4 × K';
  const series = [['matvec', 'var(--series-1)', 'matrix-vector'], ['whole', 'var(--series-2)', 'whole pass']];
  series.forEach(([k, col, name]) => {
    el(svg, 'path', {d: ROOF.map((p, i) => (i ? 'L' : 'M') + lx(p.K) + ',' + ly(p[k])).join(' '), fill: 'none', 'stroke-width': 2}, {stroke: col});
    ROOF.forEach(p => {
      el(svg, 'circle', {cx: lx(p.K), cy: ly(p[k]), r: 5, 'stroke-width': 2}, {fill: col, stroke: 'var(--surface-1)'});
      const hit = el(svg, 'circle', {cx: lx(p.K), cy: ly(p[k]), r: 14}, {fill: 'transparent', cursor: 'default'});
      hov(hit, `<b>K = ${p.K}, ${name}</b><span class="m">achieved</span> ${p[k].toFixed(2)} MAC/cycle<br><span class="m">roof</span> ${p.roof} MAC/cycle<br><span class="m">share of roof</span> ${(100 * p[k] / p.roof).toFixed(0)}%<br><span class="m">cycles</span> ${M(k === 'matvec' ? p.mv_cyc : p.tot_cyc)}`);
    });
    const last = ROOF[ROOF.length - 1];
    const first = ROOF[0];
    el(svg, 'text', {x: lx(last.K) + 10, y: ly(last[k]) + 4, class: 'val'}).textContent =
      `${name}: ${(100 * first[k] / first.roof).toFixed(0)}% → ${(100 * last[k] / last.roof).toFixed(0)}% of roof`;
  });
})();

// ---- 2. speedup bars ----
(() => {
  const svg = document.getElementById('spd'), W = 860, L = 290, R = 60, T = 6, BH = 26, G = 14;
  const H = T + SPD.length * (BH + G) + 26; svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const max = 2.0, x = v => L + (W - L - R) * v / max;
  const defs = el(svg, 'defs'), pat = el(defs, 'pattern', {id: 'hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)'});
  el(pat, 'rect', {width: 3, height: 6}, {fill: 'var(--series-1)'});
  [0, 0.5, 1.0, 1.5, 2.0].forEach(g => { el(svg, 'line', {x1: x(g), x2: x(g), y1: T, y2: H - 22}, {stroke: 'var(--grid)'});
    el(svg, 'text', {x: x(g), y: H - 6, 'text-anchor': 'middle'}).textContent = g.toFixed(1) + '×'; });
  el(svg, 'line', {x1: x(1), x2: x(1), y1: T - 4, y2: H - 22, 'stroke-width': 2}, {stroke: 'var(--ink)'});
  SPD.forEach((r, i) => {
    const y = T + i * (BH + G), w = x(r.v) - L;
    el(svg, 'text', {x: L - 12, y: y + (r.sub ? 11 : 17), 'text-anchor': 'end', class: 'val'}).textContent = r.label;
    if (r.sub) el(svg, 'text', {x: L - 12, y: y + 25, 'text-anchor': 'end'}).textContent = r.sub;
    const d = `M${L},${y} H${L + w - 4} a4,4 0 0 1 4,4 V${y + BH - 4} a4,4 0 0 1 -4,4 H${L} Z`;
    if (r.kind === 'measured') el(svg, 'path', {d}, {fill: 'var(--series-1)'});
    else { el(svg, 'path', {d}, {fill: 'url(#hatch)'}); el(svg, 'path', {d, 'stroke-width': 1.5}, {fill: 'none', stroke: 'var(--series-1)'}); }
    let tx = L + w + 8; if (tx < x(1) + 6 && tx + 40 > x(1)) tx = x(1) + 6;   // keep the label off the 1x line
    el(svg, 'text', {x: tx, y: y + 18, class: 'val'}).textContent = r.v.toFixed(2) + '×';
    const hit = el(svg, 'rect', {x: L, y: y - 4, width: Math.max(w, 40) + 50, height: BH + 8}, {fill: 'transparent'});
    hov(hit, `<b>${r.label}</b>${r.v.toFixed(3)}× at γ${r.g}<br><span class="m">${r.kind === 'measured' ? 'measured, full simulation' : 'estimate: cycle model on measured costs'}</span>`);
  });
})();

// ---- 3. resources ----
(() => {
  const svg = document.getElementById('res'), W = 860, L = 170, R = 90, T = 6, BH = 22, G = 10;
  const H = T + RES.rows.length * (BH + G) + 24; svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const max = 6000, x = v => L + (W - L - R) * v / max;
  [0, 1000, 2000, 3000, 4000, 5000, 6000].forEach(g => { el(svg, 'line', {x1: x(g), x2: x(g), y1: T, y2: H - 20}, {stroke: 'var(--grid)'});
    el(svg, 'text', {x: x(g), y: H - 4, 'text-anchor': 'middle'}).textContent = g ? (g / 1000) + 'K' : '0'; });
  RES.rows.forEach((r, i) => {
    const y = T + i * (BH + G), w = Math.max(x(r.v) - L, 2);
    el(svg, 'text', {x: L - 12, y: y + 15, 'text-anchor': 'end'}).textContent = r.label;
    el(svg, 'path', {d: `M${L},${y} H${L + w - 4} a4,4 0 0 1 4,4 V${y + BH - 4} a4,4 0 0 1 -4,4 H${L} Z`}, {fill: 'var(--series-1)'});
    el(svg, 'text', {x: L + w + 8, y: y + 15, class: 'val'}).textContent = r.v.toLocaleString('en-US');
    const hit = el(svg, 'rect', {x: L, y: y - 3, width: w + 60, height: BH + 6}, {fill: 'transparent'});
    hov(hit, `<b>${r.label}</b>${r.v.toLocaleString('en-US')} LUTs<br><span class="m">${(100 * r.v / RES.total).toFixed(1)}% of the design, ${(100 * r.v / RES.avail).toFixed(2)}% of the chip</span>`);
  });
})();

// ---- table views ----
const tables = {
  t1: ['<tr><th>K</th><th>roof MAC/cycle</th><th>matvec MAC/cycle</th><th>whole pass MAC/cycle</th><th>matvec cycles</th><th>pass cycles</th></tr>',
       ROOF.map(p => `<tr><td>${p.K}</td><td>${p.roof}</td><td>${p.matvec.toFixed(2)}</td><td>${p.whole.toFixed(2)}</td><td>${p.mv_cyc.toLocaleString('en-US')}</td><td>${p.tot_cyc.toLocaleString('en-US')}</td></tr>`).join('')],
  t2: ['<tr><th>case</th><th>speedup</th><th>γ</th><th>kind</th></tr>',
       SPD.map(r => `<tr><td>${r.label}</td><td>${r.v.toFixed(3)}×</td><td>${r.g}</td><td>${r.kind}</td></tr>`).join('')],
  t3: ['<tr><th>module</th><th>LUTs</th><th>% of design</th></tr>',
       RES.rows.map(r => `<tr><td>${r.label}</td><td>${r.v.toLocaleString('en-US')}</td><td>${(100 * r.v / RES.total).toFixed(1)}%</td></tr>`).join('')],
};
document.querySelectorAll('button[data-t]').forEach(b => b.addEventListener('click', () => {
  const d = document.getElementById(b.dataset.t), open = d.hidden;
  if (open && !d.innerHTML) d.innerHTML = '<table>' + tables[b.dataset.t].join('') + '</table>';
  d.hidden = !open; b.textContent = open ? 'Hide table' : 'Show table'; b.setAttribute('aria-expanded', open);
}));
</script></body></html>
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pts, macs, ctx = roofline()
    spd, plain, spec = speedups()
    res, lut, lut_avail, mhz, dsp = resources()
    rep = {"__ROOF__": json.dumps(pts), "__SPDROWS__": json.dumps(spd),
           "__RES__": json.dumps(dict(rows=res, total=lut, avail=lut_avail)),
           "__SPD__": f"{SPD_SUM:.2f}", "__CPT__": f"{PER_STEP / 1e6:.2f}", "__MHZ__": f"{mhz:.0f}",
           "__LUTP__": f"{100 * lut / lut_avail:.1f}", "__LUT__": f"{lut:,}", "__LUTA__": f"{lut_avail:,}",
           "__CTX__": str(ctx), "__MACS__": f"{macs:,}", "__DSP__": str(dsp)}
    page = TEMPLATE
    for k, v in rep.items():
        page = page.replace(k, html.escape(v) if not k in ("__ROOF__", "__SPDROWS__", "__RES__") else v)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}")
    for p in pts:
        print(f"  K={p['K']}: roof {p['roof']}, matvec {p['matvec']:.2f}, whole {p['whole']:.2f} MAC/cycle")
    for r in spd:
        print(f"  {r['label']:<42} {r['v']:.3f}x  g{r['g']}  {r['kind']}")
    print(f"  LUTs {lut:,} / {lut_avail:,} ({100 * lut / lut_avail:.2f}%), {mhz:.1f} MHz, DSP {dsp}: " +
          ", ".join(f"{r['label']} {r['v']}" for r in res))


if __name__ == "__main__":
    main()
