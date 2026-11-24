"""SpecCore demo data: replays speculative decoding round by round for the demo page.

    python tools/demo.py                          default prompt set -> artifacts/demo/SpecCore_demo.html
    python tools/demo.py --text "def add(a, b):"  your own prompt (added to the set)

Where every number comes from:
  tokens, guesses, accept/reject  integer model (model/qmodel.py), the same arithmetic as the core, bit-exact
                                  (checked on the golden prompts: the core's output equals the C reference)
  cycles per pass                 MEASURED pass costs on the core in xsim (build/ksweep_accq_bp_core.txt,
                                  fitted as shared + per-lane, ctx 15), so the replay clock is an ESTIMATE
                                  built from measured parts; context growth inside a run is not modelled
  core totals                     MEASURED on the KV260 board (logs/board/core_bq*.txt; = xsim cycle for cycle)
  50-prompt panel                 MEASURED on the board (logs/board/set50_summary.txt)
The page itself (artifacts/demo/demo_template.html) only animates this data."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "tools")]
import qmodel  # noqa: E402
import cyclemodel as cm  # noqa: E402

# (label, seed, plen, n_new, measured-core log or None)
# readable starts first (picked from clean code openings, stopped before the model repeats itself),
# then the three test prompts that were also run on the board
TEXTS = [("argparse setup", "def main():\n    parser = argparse.ArgumentParser()\n", 22),
         ("imports", "try:\n    import numpy as np\nexcept ImportError:\n", 20),
         ("file reader", "import os\n\n\ndef read_file(path):\n", 22)]
DEFAULT = [("board test 1", 1, 16, 8, "core_bqg.txt"),
           ("board test 2", 7, 8, 16, "core_bqs7.txt"),
           ("board test 3", 11, 8, 16, "core_bqs11.txt")]


def tokenizer():
    from tokenizers import ByteLevelBPETokenizer
    d = ROOT / "model/tokenizer/bpe2048"
    return ByteLevelBPETokenizer(str(d / "vocab.json"), str(d / "merges.txt"))


def core_totals(name):
    p = ROOT / "logs/board" / name if name else None      # measured on the KV260 (24 Nov)
    if not p or not p.exists():
        return None
    t = p.read_text(errors="replace")
    pl = re.search(r"cycles plain target total (\d+)", t)
    sp = re.search(r"cycles spec total (\d+)", t)
    return dict(plain=int(pl.group(1)), spec=int(sp.group(1))) if pl and sp and "REF PASS" in t else None


def costs():
    sweep, ctx = cm.read_ksweep(ROOT / "build/ksweep_accq_bp_core.txt")
    cal = cm.calibrate(sweep)
    return dict(ctx=ctx, **{m: {b: [round(cal[m][b][0]), round(cal[m][b][1])] for b in cm.CAL_BUCKETS + ("total",)}
                            for m in ("draft", "target")})


def run_prompt(tok, label, prompt, n_new, core=None):
    piece = lambda t: tok.decode([int(t)])
    plain = qmodel.qgreedy(qmodel.QModel("target"), prompt, n_new)
    runs = {}
    for key, gamma, gate in (("g2", 2, None), ("gate", 4, 4096)):
        tr = []
        out, st = qmodel.qspec(qmodel.QModel("draft"), qmodel.QModel("target"), prompt, n_new, gamma, gate, trace=tr)
        assert out == plain, f"{label} {key}: speculative output differs from plain greedy"
        runs[key] = dict(rounds=tr, stats=st)
    vocab = sorted({int(t) for t in prompt + plain} | {int(g) for r in runs.values() for x in r["rounds"] for g in x["guesses"]})
    return dict(label=label, prompt=[int(t) for t in prompt], plain=[int(t) for t in plain],
                pieces={t: piece(t) for t in vocab}, runs=runs, core=core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", default=[], help="extra prompt text (tokenised with the project BPE)")
    ap.add_argument("--new", type=int, default=32, help="tokens to generate for --text prompts")
    ap.add_argument("--out", default=str(ROOT / "artifacts/demo/SpecCore_demo.html"))
    ap.add_argument("--render-only", action="store_true", help="reuse artifacts/demo/demo_data.json (page edits only)")
    a = ap.parse_args()
    cache = ROOT / "artifacts/demo/demo_data.json"
    if a.render_only:
        return render(cache.read_text(encoding="utf-8"), a.out)
    spec, _, _ = qmodel.float_models()
    tok = tokenizer()
    data = []
    for label, txt, n_new in TEXTS:
        data.append(run_prompt(tok, label, tok.encode(txt).ids, n_new))
        print("done", label, flush=True)
    for label, seed, plen, n_new, log in DEFAULT:
        p = spec.prompts(1, plen, seed=seed)[0]
        data.append(run_prompt(tok, label, p, n_new, core_totals(log)))
        print("done", label, flush=True)
    for i, txt in enumerate(a.text):
        p = tok.encode(txt).ids[-64:]
        data.append(run_prompt(tok, txt[:24], p, a.new))
        print("done", txt[:30], flush=True)
    payload = json.dumps(dict(costs=costs(), prompts=data), separators=(",", ":"))
    cache.write_text(payload, encoding="utf-8")
    render(payload, a.out)


def board50():
    """Per-prompt (plain, spec) core cycles of the 50-prompt test set on the KV260 + summary numbers."""
    t = (ROOT / "logs/board/set50_summary.txt").read_text(encoding="utf-8")
    rows = [[int(a.replace(",", "")), int(b.replace(",", ""))]
            for a, b in re.findall(r"\| (?:bset/)?p\d\d \| ([\d,]+) \| ([\d,]+) \|", t)]
    w = re.search(r"= ([\d.]+)x \(same programs in core cycles: ([\d.]+)x; DDR overhead plain \+([\d.]+)%, spec \+([\d.]+)%", t)
    return dict(rows=rows, wall=float(w.group(1)), wall_core=float(w.group(2)),
                ddr_plain=float(w.group(3)), ddr_spec=float(w.group(4)))


def render(payload, out):
    d = json.loads(payload)
    logs = {lab: log for lab, _, _, _, log in DEFAULT}
    for pr in d["prompts"]:
        pr["core"] = core_totals(logs.get(pr["label"]))
    d["board50"] = board50()
    payload = json.dumps(d, separators=(",", ":"))
    tpl = (ROOT / "artifacts/demo/demo_template.html").read_text(encoding="utf-8")
    out = Path(out)
    out.write_text(tpl.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print("wrote", out, f"({len(payload) // 1024} KB data)")


if __name__ == "__main__":
    main()
