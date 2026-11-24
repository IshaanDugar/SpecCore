"""Chart: where the cycles of one target pass go as K (tokens checked) grows. MEASURED data.

Reads a K-sweep core run (ref/run_ref.c -DRR_KSWEEP) and writes a self-contained HTML page
(SVG stacked bars, hover tooltips, legend, table view, light/dark).
The matvec bucket is split into the weight stream (shared: the linear fit's constant part) and
readout + bank fill (the rest, grows with K); the other buckets are per-token already.

  python tools/chart_verify_pass.py [build/ksweep_fast2_bp_core.txt] [artifacts/charts/verify_pass_cost.html]
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyclemodel import calibrate, read_ksweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEGS = [  # key, label, css var
    ("weights", "Weight stream (shared)", "--series-1"),
    ("readout", "Readout + bank fill", "--series-2"),
    ("attn", "Attention", "--series-3"),
    ("normq", "Norm / quant", "--series-4"),
    ("other", "Other", "--series-5"),
]
ROLE = {1: "plain decoding", 2: "verify, γ1", 3: "verify, γ2", 5: "verify, γ4"}


def rows_from(path):
    sweep, ctx = read_ksweep(path)
    cal = calibrate(sweep)
    S_mv = cal["target"]["matvec"][0]
    rows = []
    for K in sorted(sweep["target"]):
        b = sweep["target"][K]
        rows.append(dict(K=K, role=ROLE.get(K, ""), weights=S_mv, readout=b["matvec"] - S_mv,
                         attn=b["attn"], normq=b["normq"], other=b["other"], total=b["total"]))
    S, P, _ = cal["target"]["total"]
    return rows, ctx, S, P


def page(rows, ctx, S, P, src):
    data = json.dumps(rows)
    segs = json.dumps([[k, l, v] for k, l, v in SEGS])
    k1, k3 = rows[0]["total"], next(r["total"] for r in rows if r["K"] == 3)
    title = "Only the weight stream is shared: every extra token checked adds ~1.7M cycles"
    sub = (f"One target-model pass on the RISC-V core with the LLM unit, by number of tokens checked (K). "
           f"Measured in cycle-accurate simulation (xsim), branch predictor on, context {ctx}. "
           f"Fit: {S / 1e6:.2f}M shared + {P / 1e6:.2f}M per token ({P / (S + P):.2f} of a plain pass).")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>SpecCore: cost of a verify pass</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
.viz-root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e69;
  --grid: #e6e5e0; --axis: #b9b8b2;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100; --series-5: #e87ba4;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --grid: #2e2e2c; --axis: #55544f;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark;
  --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --grid: #2e2e2c; --axis: #55544f;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
}}
body {{ margin: 0; }}
.viz-root {{ background: var(--surface-1); color: var(--text-primary); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 28px 32px; max-width: 920px; }}
h1 {{ font-size: 19px; font-weight: 600; margin: 0 0 6px; }}
.sub {{ color: var(--text-secondary); margin: 0 0 18px; max-width: 760px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 10px; color: var(--text-secondary); font-size: 13px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
svg text {{ fill: var(--text-secondary); font-size: 12px; }}
svg .val {{ fill: var(--text-primary); font-weight: 600; }}
svg .seglab {{ fill: #fff; font-size: 11px; font-weight: 600; pointer-events: none; }}
.tip {{ position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--axis); border-radius: 6px; padding: 8px 10px; font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,.15);
  display: none; min-width: 190px; }}
.tip b {{ display: block; margin-bottom: 2px; }}
.tip .m {{ color: var(--text-secondary); }}
.note {{ color: var(--text-muted); font-size: 12.5px; margin-top: 10px; }}
button {{ font: inherit; font-size: 13px; color: var(--text-primary); background: transparent; border: 1px solid var(--axis);
  border-radius: 6px; padding: 4px 10px; cursor: pointer; margin-top: 12px; }}
table {{ border-collapse: collapse; margin-top: 10px; font-size: 13px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--grid); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--text-secondary); font-weight: 600; }}
</style></head>
<body><div class="viz-root">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(sub)}</p>
<div class="legend" id="legend"></div>
<svg id="chart" role="img" aria-label="{html.escape(title)}" width="860" height="300"></svg>
<div class="note">K=1 to K=3 (a γ2 verify pass): {k1 / 1e6:.2f}M → {k3 / 1e6:.2f}M cycles, +{(k3 - k1) / k1:.0%}.
Readout + bank fill = matvec minus its shared part (linear fit). Source: {html.escape(src)}.</div>
<button id="tbtn" aria-expanded="false">Show table</button>
<div id="tbl" hidden></div>
<div class="tip" id="tip" role="tooltip"></div>
</div>
<script>
const ROWS = {data};
const SEGS = {segs};
const css = n => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim();
const M = v => (v / 1e6).toFixed(2) + 'M';
const svg = document.getElementById('chart'), NS = 'http://www.w3.org/2000/svg';
const W = 860, L = 150, R = 70, T = 8, BH = 36, GAP = 22, H = T + ROWS.length * (BH + GAP) + 26;
svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
const maxV = Math.ceil(Math.max(...ROWS.map(r => r.total)) / 2e6) * 2e6;
const x = v => L + (W - L - R) * v / maxV;
const el = (t, a, p = svg) => {{ const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; }};
const lg = document.getElementById('legend');
SEGS.forEach(([k, lab, v]) => {{ lg.insertAdjacentHTML('beforeend', `<span><i style="background:var(${{v}})"></i>${{lab}}</span>`); }});
for (let g = 0; g <= maxV; g += 2e6) {{
  el('line', {{x1: x(g), x2: x(g), y1: T - 4, y2: H - 22, stroke: css('--grid'), 'stroke-width': 1}});
  el('text', {{x: x(g), y: H - 6, 'text-anchor': 'middle'}}).textContent = g === 0 ? '0' : (g / 1e6) + 'M';
}}
const tip = document.getElementById('tip');
const show = (ev, h) => {{ tip.innerHTML = h; tip.style.display = 'block';
  const r = tip.getBoundingClientRect(); let px = ev.clientX + 14, py = ev.clientY + 14;
  if (px + r.width > innerWidth - 8) px = ev.clientX - r.width - 14; if (py + r.height > innerHeight - 8) py = ev.clientY - r.height - 14;
  tip.style.left = px + 'px'; tip.style.top = py + 'px'; }};
const hide = () => tip.style.display = 'none';
ROWS.forEach((r, i) => {{
  const y = T + i * (BH + GAP);
  const lab = el('text', {{x: L - 12, y: y + BH / 2 - 3, 'text-anchor': 'end'}});
  lab.textContent = 'K = ' + r.K; lab.setAttribute('class', 'val');
  el('text', {{x: L - 12, y: y + BH / 2 + 12, 'text-anchor': 'end'}}).textContent = r.role;
  const cid = 'c' + i, end = x(r.total);
  const cp = el('clipPath', {{id: cid}});
  el('path', {{d: `M${{L}},${{y}} H${{end - 4}} a4,4 0 0 1 4,4 V${{y + BH - 4}} a4,4 0 0 1 -4,4 H${{L}} Z`}}, cp);
  const grp = el('g', {{'clip-path': `url(#${{cid}})`}});
  let acc = 0;
  SEGS.forEach(([k, lab, v], j) => {{
    const v0 = acc, v1 = acc + r[k]; acc = v1;
    const x0 = x(v0) + (j ? 1 : 0), x1 = x(v1) - (j < SEGS.length - 1 ? 1 : 0);   // 2px surface gap between fills
    if (x1 <= x0) return;
    const hit = el('rect', {{x: x0, y, width: x1 - x0, height: BH, fill: css(v)}}, grp);
    if (x1 - x0 > 46) {{ const t = el('text', {{x: (x0 + x1) / 2, y: y + BH / 2 + 4, 'text-anchor': 'middle', class: 'seglab'}}, svg);
      t.textContent = M(r[k]); }}
    const share = (100 * r[k] / r.total).toFixed(0);
    const kind = k === 'weights' ? 'shared: paid once per pass' : 'grows with each token checked';
    const h = `<b>K = ${{r.K}} · ${{lab}}</b>${{M(r[k])}} cycles (${{share}}% of this pass)<div class="m">${{kind}}</div>`;
    hit.addEventListener('mousemove', e => show(e, h)); hit.addEventListener('mouseleave', hide);
  }});
  el('text', {{x: end + 8, y: y + BH / 2 + 4, class: 'val'}}).textContent = M(r.total);
}});
const tb = document.getElementById('tbl'), btn = document.getElementById('tbtn');
tb.innerHTML = '<table><tr><th>K</th>' + SEGS.map(s => `<th>${{s[1]}}</th>`).join('') + '<th>Total</th></tr>' +
  ROWS.map(r => `<tr><td>K = ${{r.K}} (${{r.role}})</td>` + SEGS.map(s => `<td>${{r[s[0]].toLocaleString('en', {{maximumFractionDigits: 0}})}}</td>`).join('') +
  `<td>${{r.total.toLocaleString('en')}}</td></tr>`).join('') + '</table>';
btn.onclick = () => {{ const o = tb.hidden; tb.hidden = !o; btn.textContent = o ? 'Hide table' : 'Show table'; btn.setAttribute('aria-expanded', o); }};
</script>
</body></html>
"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "build/ksweep_fast2_bp_core.txt"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "artifacts/charts/verify_pass_cost.html")
    rows, ctx, S, P = rows_from(ROOT / src if not Path(src).is_absolute() else src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(rows, ctx, S, P, src), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
