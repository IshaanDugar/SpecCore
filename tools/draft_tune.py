"""Model-pair decision (23 Nov): can a fixed-shape 128x2 draft make the long-trained target pay off?

    python tools/draft_tune.py mix      CPU: old draft (export_v2) x new target (export_v3), 50 test prompts
    python tools/draft_tune.py train    GPU: fine-tune candidates from d128x2_kd2, pick on dev prompts, test the best
    python tools/draft_tune.py report   table + verdict -> logs/draft_tune/summary.txt

Test set = qmodel --spec seed 0, 50 prompts x plen 16 x 32 new (same as logs/qspec_int_50*.txt).
Dev set = seed 1 (candidate selection only, so the test number is not picked on itself).
Estimated speedup: measured ACCQ+BP pass costs (build/ksweep_accq_bp_core.txt, ctx 15) fitted as
shared + per-lane x K; spec = rounds x T_shared + (rounds + guesses) x T_lane + draft passes x D(K=1)
+ catch-up tokens x D_lane; plain = tokens x T(K=1). Same formula for every row, so rows compare fairly;
absolute values differ slightly from earlier hand accounting (old pair is shown as the reference row).
Adopt rule (decided 23 Nov, threshold from the astra review): new pair only if est >= old x 1.16/1.175.
Every job runs under tools/capped.py (hard commit cap)."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints"
OUT = ROOT / "logs" / "draft_tune"
PY = sys.executable
N_TOK = 50 * 32
RULE = 1.16 / 1.175
TEACHER = "checkpoints/target_long.pt"
# name: (train.py extra args, tokens). All warm-start from d128x2_kd2, teacher = target_long.
CANDS = {
    "kd2_hard100": (["--distil", "1.0", "--hard"], 100e6),
    "kd2_soft100": (["--distil", "1.0"], 100e6),
    "kd2_mix300": (["--distil", "0.7", "--hard"], 300e6),
}


def run(cmd, log, mb, secs, env=None):
    OUT.mkdir(parents=True, exist_ok=True)
    full = [PY, str(ROOT / "tools/capped.py"), str(mb), str(secs), str(OUT / log), "--"] + cmd
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run(full, cwd=ROOT, env={**os.environ, **(env or {})})
    if r.returncode:
        sys.exit(f"FAILED ({r.returncode}): see {OUT / log}")


def pair_dir(name, draft_npz):
    d = CK / f"export_{name}"
    d.mkdir(exist_ok=True)
    if draft_npz != d / "draft.npz":
        shutil.copy(draft_npz, d / "draft.npz")
    shutil.copy(CK / "export_v3" / "target.npz", d / "target.npz")
    return d


def spec(d, seed, log):
    run([PY, "model/qmodel.py", "--spec", "--prompts", "50", "--plen", "16", "--new", "32",
         "--gammas", "2", "--seed", str(seed)], log, 3000, 4 * 3600, {"SC_EXPORT": str(d)})


def parse(path):
    """gamma-2 line of a qmodel --spec log -> stats dict (None if missing)."""
    if not Path(path).exists():
        return None
    for ln in Path(path).read_text(errors="ignore").splitlines():
        m = re.search(r"gamma 2: identical (\d+)/(\d+) \| tokens/target pass ([\d.]+) .*accepted (\d+)/(\d+) "
                      r"\| draft passes (\d+) \(peek (\d+)\) \| catch-up (\d+)", ln)
        if m:
            same, n, tpp, acc, prop, dp, pk, cu = m.groups()
            return dict(same=int(same), n=int(n), tpp=float(tpp), acc=int(acc), prop=int(prop),
                        dpass=int(dp), catchup=int(cu), rounds=round(N_TOK / float(tpp)))
    return None


def costs():
    fit = {}
    for mdl in ("draft", "target"):
        pts = [(int(k), int(t)) for k, t in re.findall(
            rf"cycles ksweep {mdl} K (\d+) .* total (\d+)", (ROOT / "build/ksweep_accq_bp_core.txt").read_text())]
        n = len(pts); sk = sum(k for k, _ in pts); st = sum(t for _, t in pts)
        skk = sum(k * k for k, _ in pts); skt = sum(k * t for k, t in pts)
        lane = (n * skt - sk * st) / (n * skk - sk * sk)
        fit[mdl] = ((st - lane * sk) / n, lane)
    return fit


def est(s, fit):
    (ds, dl), (ts, tl) = fit["draft"], fit["target"]
    spec_c = s["rounds"] * ts + (s["rounds"] + s["prop"]) * tl + s["dpass"] * (ds + dl) + s["catchup"] * dl
    return N_TOK * (ts + tl) / spec_c


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    fit = costs()
    rows = [("old pair (active, all core results)", ROOT / "logs/qspec_int_50.txt"),
            ("new target + its draft d128x2_kd2", ROOT / "logs/qspec_int_50_v3.log"),
            ("new target + OLD draft d128x2_kd", OUT / "mix_test.log")]
    best = json.loads((OUT / "best.json").read_text()) if (OUT / "best.json").exists() else None
    if best:
        rows.append((f"new target + {best['name']} (picked on dev)", OUT / f"{best['name']}_test.log"))
    lines = ["# Model-pair decision, test set (50 prompts x 32 tokens, integer, gamma 2)", "",
             f"pass-cost fit (measured, ctx 15): draft {fit['draft'][0]/1e6:.3f}M + {fit['draft'][1]/1e6:.3f}M/lane, "
             f"target {fit['target'][0]/1e6:.3f}M + {fit['target'][1]/1e6:.3f}M/lane", "",
             "| pair | identical | tok/target pass | accepted | catch-up/round | est. speedup | vs old |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    old = None
    verdict = []
    for name, p in rows:
        s = parse(p)
        if not s:
            lines.append(f"| {name} | (not run) | | | | | |")
            continue
        e = est(s, fit)
        old = old or e
        lines.append(f"| {name} | {s['same']}/{s['n']} | {s['tpp']:.3f} | {s['acc']}/{s['prop']} | "
                     f"{s['catchup']/s['rounds']:.2f} | {e:.3f}x | {e/old:.3f} |")
        if "picked" in name:
            ok = e >= old * RULE and s["same"] == s["n"]
            verdict.append(f"VERDICT: {'ADOPT new pair (then golden regen + 3 core runs)' if ok else 'KEEP OLD PAIR'}"
                           f" ({e/old:.3f} of old, rule >= {RULE:.3f})")
    if best:
        lines += ["", "dev set (seed 1) tok/target pass: " + ", ".join(f"{k} {v:.3f}" for k, v in best["dev"].items())]
    lines += [""] + (verdict or ["VERDICT: pending (run train)"])
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    if what == "mix":
        spec(pair_dir("mix", CK / "export_v2" / "draft.npz"), 0, "mix_test.log")
    elif what == "train":
        import torch
        if not torch.cuda.is_available():
            sys.exit("GPU is off (torch.cuda.is_available() False): turn it on first")
        dev = {"d128x2_kd2": None}
        for name, (extra, tok) in CANDS.items():
            if not (CK / f"{name}.pt").exists():
                run([PY, "model/train.py", "--model", "draft", "--name", name, "--init", "checkpoints/d128x2_kd2.pt",
                     "--teacher", TEACHER, "--lr", "5e-4", "--tokens", str(tok), "--hours", "0.5"] + extra,
                    f"{name}_train.log", 6000, 3600)
            dev[name] = None
        for name in dev:
            d = CK / f"export_{name}"
            if not (d / "draft.npz").exists():
                run([PY, "model/export.py", "--models", "draft", "--ckpt", name, "--no-img"],
                    f"{name}_export.log", 3000, 1800, {"SC_EXPORT": str(d)})
            pair_dir(name, d / "draft.npz")
            spec(d, 1, f"{name}_dev.log")
            dev[name] = parse(OUT / f"{name}_dev.log")["tpp"]
        name = max(dev, key=dev.get)
        spec(CK / f"export_{name}", 0, f"{name}_test.log")
        (OUT / "best.json").write_text(json.dumps({"name": name, "dev": dev}))
    report()


if __name__ == "__main__":
    main()
