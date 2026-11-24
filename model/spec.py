"""Speculative decoding on the PC (greedy). Explained in primer 04.

One round:
  1. draft guesses gamma tokens, one at a time (cheap model, gamma passes)
  2. target checks all of them in ONE pass over the K = gamma + 1 new positions
  3. keep guesses up to the first mismatch, then append the target's own token there
     (if all gamma match, the target's extra token is a free bonus)
So every round emits 1..K tokens, and the output equals plain greedy decoding of the target.

Usage:
  python model/spec.py --selftest          random weights; checks identity + draft==target gives alpha 1.0
  python model/spec.py                     trained checkpoints; 20 prompts, alpha vs gamma, guess log
  python model/spec.py --draft d64x2_kd --prompts 200 --logdir logs/d15/d64x2_kd   a D15 draft variant
Guess log (for the D12 predictor bake-off): logs/guesses_g<gamma>.csv
  columns: prompt, round, slot, pos, last_tok, guess, accepted, conf (draft top-1 minus top-2 logit)

No KV cache here: every pass recomputes the whole sequence. Slow but simple, and the
math is the same. Total length is capped at ctx (128) so no window shifting happens.
Float64 on CPU so tiny rounding differences can't flip an argmax between the two paths.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))
from model import DRAFT, TARGET, Model  # noqa: E402
import sc  # noqa: E402


@torch.no_grad()
def logits_at(m, seq):
    """All-position logits (T, vocab) for one sequence (list of ids)."""
    lg, _ = m(torch.tensor([seq]))
    return lg[0]


@torch.no_grad()
def greedy(target, prompt, n_new):
    seq = list(prompt)
    for _ in range(n_new):
        seq.append(int(logits_at(target, seq)[-1].argmax()))
    return seq[len(prompt):]


@torch.no_grad()
def speculative(draft, target, prompt, n_new, gamma, log=None, pid=0):
    """Returns (new tokens, stats). stats: rounds, accepted guesses, proposed guesses, target passes."""
    seq = list(prompt)
    end = len(prompt) + n_new
    st = dict(rounds=0, acc=0, prop=0, tpass=0, dpass=0)
    while len(seq) < end:
        g = min(gamma, end - len(seq) - 1)          # leave room for the target's own token
        # 1. draft guesses g tokens
        guesses, confs = [], []
        d = list(seq)
        for _ in range(g):
            lg = logits_at(draft, d)[-1]
            top2 = lg.topk(2).values
            t = int(lg.argmax())
            guesses.append(t); confs.append(float(top2[0] - top2[1]))
            d.append(t)
            st["dpass"] += 1
        # 2. target checks seq + guesses in one pass; position i predicts token i+1
        lg = logits_at(target, seq + guesses)
        want = lg[len(seq) - 1:].argmax(-1).tolist()  # g+1 target picks
        st["tpass"] += 1
        # 3. accept up to the first mismatch, then take the target's token
        n_ok = 0
        while n_ok < g and guesses[n_ok] == want[n_ok]:
            n_ok += 1
        if log is not None:
            for i in range(g):
                last = seq[-1] if i == 0 else guesses[i - 1]
                log.writerow([pid, st["rounds"], i, len(seq) + i, last, guesses[i],
                              int(i < n_ok) if i <= n_ok else "", f"{confs[i]:.4f}"])
        seq += guesses[:n_ok] + [want[n_ok]]
        st["rounds"] += 1; st["acc"] += n_ok; st["prop"] += g
    return seq[len(prompt):end], st


def load(cfg, name):
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location="cpu")
    if "cfg" in ck:                       # draft variants (D15) carry their own shape
        cfg = type(cfg)(**ck["cfg"])
    m = Model(cfg)
    m.load_state_dict(ck["model"])
    return m.double().eval()


def prompts(n, plen, seed=0):
    val = np.memmap(ROOT / "data/tok2048/val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    return [val[s:s + plen].astype(int).tolist() for s in rng.integers(0, len(val) - plen, n)]


def run(draft, target, gammas, n_prompts, plen, n_new, logdir=None):
    ps = prompts(n_prompts, plen)
    ref = [greedy(target, p, n_new) for p in ps]
    rows = []
    for g in gammas:
        f = w = None
        if logdir:
            logdir.mkdir(exist_ok=True)
            f = open(logdir / f"guesses_g{g}.csv", "w", newline="")
            w = csv.writer(f)
            w.writerow(["prompt", "round", "slot", "pos", "last_tok", "guess", "accepted", "conf"])
        same = 0; tot = dict(rounds=0, acc=0, prop=0, tpass=0, dpass=0)
        for i, p in enumerate(ps):
            out, st = speculative(draft, target, p, n_new, g, w, i)
            same += out == ref[i]
            for k in tot: tot[k] += st[k]
        if f: f.close()
        alpha = tot["acc"] / max(tot["prop"], 1)
        tpr = n_prompts * n_new / tot["rounds"]
        rows.append((g, same, alpha, tpr))
        print(f"gamma {g}: identical {same}/{n_prompts} | alpha {alpha:.3f} | tokens/target pass {tpr:.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--plen", type=int, default=32)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--draft", default="draft", help="draft checkpoint name (D15 variants)")
    ap.add_argument("--logdir", default="logs", help="guess-log directory (relative to the repo)")
    a = ap.parse_args()
    assert a.plen + a.new <= TARGET.ctx, "keep prompt + new tokens within ctx"
    torch.manual_seed(0)
    if a.selftest:
        t = Model(TARGET).double().eval()
        d = Model(DRAFT).double().eval()
        print("A) random draft vs random target (alpha can be anything; output must still be identical)")
        ra = run(d, t, [4], 5, a.plen, 32)
        print("B) draft = target (alpha must be 1.000, tokens/pass must be gamma+1)")
        rb = run(t, t, [4], 5, a.plen, 30)
        ok = ra[0][1] == 5 and rb[0][1] == 5 and rb[0][2] == 1.0
        print("SELFTEST", "PASS" if ok else "FAIL")
        return
    print(f"default gamma in speccore.toml: {sc.load()['spec']['gamma']}")
    print(f"draft: checkpoints/{a.draft}.pt")
    run(load(DRAFT, a.draft), load(TARGET, "target"), a.gammas, a.prompts, a.plen, a.new, ROOT / a.logdir)


if __name__ == "__main__":
    main()
