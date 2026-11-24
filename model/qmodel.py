"""Integer-only reference model: the exact arithmetic the core and the C reference must reproduce.

    python model/qmodel.py --selftest            scale helpers + batched == one-by-one + spec identity
    python model/qmodel.py --agree [--prompts N]  float vs integer agreement table (both models)
    python model/qmodel.py --spec  [--prompts N]  integer speculative decoding: identity + tokens/pass + draft catch-up
        [--gate T.. --gmax 4 --gate-mode peek next]  + D14 confidence gate (T in Q12 logit units)

Needs checkpoints/export/<name>.npz (python model/export.py).

Every number here is an integer (numpy int64 or Python int). No floats after export.
Formats:
  residual x            int32, Q12 (real = x / 4096)
  matvec input          int8 vector + one scale for the whole vector ("per-token scale", D4)
  scale                 (m, e): real = m * 2^-e, m in [2^29, 2^30) or m = 0  (D5)
  K/V cache             int8 per position, one scale per (position, head)
  softmax               base 2, Q12 exponent, 256-entry Q15 table, then int8 weights

The four helpers below are the whole number system. The C version must do the same
operations in the same order (64-bit products, floor division, arithmetic right shift,
round-half-up via + 2^(sh-1)).
  norm_scale(num, den)   exact floor(num / den * 2^e) normalised into [2^29, 2^30)
  smul(a, b)             product of two scales, floor, renormalised
  apply(v, s, frac)      v * s * 2^frac, rounded: (v*m + 2^(sh-1)) >> sh, sh = e - frac
  quant(v, frac)         int vector (real = v * 2^-frac) -> int8 (+-127) + scale

Decode, one token at position pos (the verify pass of K tokens = K of these in order;
lane j only reads cache rows <= its own position, so batched == one-by-one exactly):
  x = emb row (Q12)
  per layer:  a = rmsquant(x); q,k,v = qkv @ a; RoPE(q, k); per head: quant q, k, v, store k, v;
              scores = q.k_t (int8 dots) * scale -> Q12 log2 units; p_t = 2^-(max - s_t) (Q15);
              w_t = p_t * sv_t -> int8; y = sum_t w_t v_t; x += proj @ quant(y)
              a = rmsquant(x); h = relu(fc1 @ a); x += fc2 @ quant(h)
  logits = out @ rmsquant(x); token = argmax (first index on ties)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORT = Path(os.environ["SC_EXPORT"]) if os.environ.get("SC_EXPORT") else ROOT / "checkpoints" / "export"
M_LO, M_HI = 1 << 29, 1 << 30
EPS_Q24 = 168                      # 1e-5 * 2^24, the RMSNorm eps in Q24 (x^2 of Q12)
I64 = np.int64


# ---------------- the number system ----------------
def norm_scale(num, den):
    """num / den as a scale. num >= 0, den > 0 (Python ints). m = floor(num * 2^e / den)."""
    num, den = int(num), int(den)
    assert num >= 0 and den > 0
    if num == 0:
        return 0, 0
    e = 29 - (num.bit_length() - den.bit_length())
    while True:
        m = (num << e) // den if e >= 0 else num // (den << -e)
        if m < M_LO:
            e += 1
        elif m >= M_HI:
            e -= 1
        else:
            return m, e


def smul(am, ae, bm, be):
    """Scale product. Works on numpy arrays (elementwise) or ints. m = 0 stays 0."""
    p = np.asarray(am, I64) * np.asarray(bm, I64)          # < 2^60
    hi = p >= (I64(1) << 59)
    m = np.where(hi, p >> 30, p >> 29)
    e = np.asarray(ae, I64) + np.asarray(be, I64) - np.where(hi, 30, 29)
    zero = p == 0
    return np.where(zero, 0, m), np.where(zero, 0, e)


def apply(v, m, e, frac):
    """round(v * m * 2^-sh), sh = e - frac > 0. v int array, |v| < 2^33; m, e scalar or array.
    Only right shifts happen (asserted), so the C version needs no left-shift case."""
    v = np.asarray(v, I64)
    assert np.all(np.abs(v) < (I64(1) << 33)), "apply: |v| >= 2^33"
    m = np.asarray(m, I64); sh = np.asarray(e, I64) - frac
    assert np.all((sh > 0) | (m == 0)), "apply: scale too large for a right shift"
    shc = np.clip(sh, 1, 62)
    out = (v * m + (I64(1) << (shc - 1))) >> shc
    return np.where((m == 0) | (sh > 62), 0, out)


def quant(v, frac):
    """Q-frac int vector -> (int8 values, m, e) with v * 2^-frac ~= q * m * 2^-e."""
    v = np.asarray(v, I64)
    amax = int(np.abs(v).max()) if v.size else 0
    if amax == 0:
        return np.zeros(v.shape, I64), 0, 0
    rm, re_ = norm_scale(127, amax)
    q = np.clip(apply(v, rm, re_, 0), -127, 127)
    m, e = norm_scale(amax, 127)
    return q, m, e + frac


def rmsquant(x):
    """RMSNorm (gain already folded into the next matrix) + per-token int8 quantisation.
    real(a) = x / rms(x) ~= q * scale. The 1/rms only changes the scale, not q."""
    x = np.asarray(x, I64)
    ss = int((x * x).sum())
    assert ss < (1 << 62)
    C = len(x)
    rms = isqrt_exact(ss // C + EPS_Q24)
    amax = int(np.abs(x).max())
    if amax == 0:
        return np.zeros(C, I64), 0, 0
    rm, re_ = norm_scale(127, amax)
    q = np.clip(apply(x, rm, re_, 0), -127, 127)
    m, e = norm_scale(amax, 127 * rms)
    return q, m, e


def isqrt_exact(n):
    """floor(sqrt(n)) for n >= 0, exact (C does the same with a bit-by-bit loop)."""
    import math
    return math.isqrt(int(n))


def matvec(W, a, am, ae, frac=12):
    """int8 W (R, C) @ int8 a, then each row to Q-frac with scale = a's scale * row scale."""
    w, wm, we = W
    acc = w.astype(I64) @ a
    m, e = smul(am, ae, wm, we)
    return apply(acc, m, e, frac)


# ---------------- the model ----------------
class QModel:
    def __init__(self, name):
        z = np.load(EXPORT / f"{name}.npz")
        self.name = name
        self.vocab, self.C, self.L, self.H, self.ctx, _ = (int(t) for t in z["cfg"])
        self.D = self.C // self.H
        g = lambda k: (z[k + ".w"], z[k + ".m"].astype(I64), z[k + ".e"].astype(I64))
        self.emb, self.out = g("emb"), g("out")
        self.qkv = [g(f"L{i}.qkv") for i in range(self.L)]
        self.proj = [g(f"L{i}.proj") for i in range(self.L)]
        self.fc1 = [g(f"L{i}.fc1") for i in range(self.L)]
        self.fc2 = [g(f"L{i}.fc2") for i in range(self.L)]
        self.cos, self.sin = z["rope_cos"].astype(I64), z["rope_sin"].astype(I64)
        self.lut = z["exp2_lut"].astype(I64)
        self.am, self.ae = int(z["attn_m"]), int(z["attn_e"])
        self.reset()

    def reset(self):
        L, H, T, D = self.L, self.H, self.ctx, self.D
        self.k8 = np.zeros((L, H, T, D), I64); self.v8 = np.zeros((L, H, T, D), I64)
        self.km = np.zeros((L, H, T), I64); self.ke = np.zeros((L, H, T), I64)
        self.vm = np.zeros((L, H, T), I64); self.ve = np.zeros((L, H, T), I64)
        self.n = 0                                           # tokens in the cache
        self.stats = dict(max_resid=0)

    def truncate(self, n):
        """Roll the cache back to n tokens (rejected guesses). Rows past n are simply ignored."""
        assert n <= self.n
        self.n = n

    def rope(self, x, pos):
        """x (H, D) Q12, pairs (0,1),(2,3)...; cos/sin Q14."""
        c, s = self.cos[pos], self.sin[pos]
        x1, x2 = x[:, 0::2], x[:, 1::2]
        y = np.empty_like(x)
        y[:, 0::2] = (x1 * c - x2 * s + (1 << 13)) >> 14
        y[:, 1::2] = (x1 * s + x2 * c + (1 << 13)) >> 14
        return y

    def step(self, tok):
        """Feed one token at position self.n; returns the logits (vocab,) int Q12."""
        pos = self.n
        assert pos < self.ctx
        w, wm, we = self.emb
        x = apply(w[tok].astype(I64), wm[tok], we[tok], 12)
        C, H, D = self.C, self.H, self.D
        for i in range(self.L):
            a, m, e = rmsquant(x)
            qkv = matvec(self.qkv[i], a, m, e)
            q = self.rope(qkv[:C].reshape(H, D), pos)
            k = self.rope(qkv[C:2 * C].reshape(H, D), pos)
            v = qkv[2 * C:].reshape(H, D)
            y = np.empty((H, D), I64)
            for h in range(H):
                self.k8[i, h, pos], self.km[i, h, pos], self.ke[i, h, pos] = quant(k[h], 12)
                self.v8[i, h, pos], self.vm[i, h, pos], self.ve[i, h, pos] = quant(v[h], 12)
                q8, qm, qe = quant(q[h], 12)
                y[h] = self.attend(i, h, pos, q8, qm, qe)
            a, m, e = quant(y.reshape(C), 12)
            x = x + matvec(self.proj[i], a, m, e)
            a, m, e = rmsquant(x)
            hdn = np.maximum(matvec(self.fc1[i], a, m, e), 0)
            a, m, e = quant(hdn, 12)
            x = x + matvec(self.fc2[i], a, m, e)
            self.stats["max_resid"] = max(self.stats["max_resid"], int(np.abs(x).max()))
        assert np.abs(x).max() < (1 << 31)
        a, m, e = rmsquant(x)
        self.n = pos + 1
        return matvec(self.out, a, m, e)

    def attend(self, i, h, pos, q8, qm, qe):
        """One head, one query at pos over cache rows 0..pos. Returns y (D,) Q12."""
        T = pos + 1
        dots = self.k8[i, h, :T] @ q8                        # int8 . int8, |dot| < 2^20
        c0m, c0e = smul(qm, qe, self.am, self.ae)            # q scale * log2(e)/sqrt(D)
        cm, ce = smul(c0m, c0e, self.km[i, h, :T], self.ke[i, h, :T])
        s = apply(dots, cm, ce, 12)                          # score in log2 units, Q12
        d = s.max() - s                                      # >= 0
        p = np.where(d >> 12 >= 31, 0, self.lut[(d & 4095) >> 4] >> np.minimum(d >> 12, 31))
        tot = int(p.sum())                                   # >= 32768 (the max position gives 2^15)
        wv = apply(p, self.vm[i, h, :T], self.ve[i, h, :T], 16)
        assert np.all(wv < (1 << 31))
        w8, wsm, wse = quant(wv, 16)                         # attention weights as int8 (>= 0)
        acc = w8 @ self.v8[i, h, :T]                         # (D,) this is the V-transposed dot over t
        rm, re_ = norm_scale(1, tot)
        ym, ye = smul(wsm, wse, rm, re_)
        return apply(acc, ym, ye, 12)

    def feed(self, toks):
        """Feed several tokens; returns the argmax after each one (list)."""
        return [int(np.argmax(self.step(t))) for t in toks]

    def feed_m(self, toks):
        """Like feed, also returns the margin after the last token: top-1 minus top-2 logit
        (Q12, duplicates count, so a tie gives 0). Same as qref.c with QR_MARGIN."""
        out = [self.step(t) for t in toks]
        lg = out[-1]
        top2 = np.partition(lg, -2)[-2:]
        return [int(np.argmax(o)) for o in out], int(top2[1] - top2[0])


# ---------------- decoding ----------------
def qgreedy(m, prompt, n_new):
    m.reset()
    nxt = m.feed(prompt)[-1]
    out = []
    for _ in range(n_new):
        out.append(nxt)
        if len(out) == n_new:
            break
        nxt = m.feed([nxt])[0]
    return out


def qspec(dm, tm, prompt, n_new, gamma, gate=None, gate_mode="peek", trace=None):
    """Integer speculative decoding with KV caches and rollback.
    Counts target passes, draft passes and draft catch-up lanes (audit e3).
    gate (D14 confidence gate, Q12 logit units): gamma is then the cap (gamma max). Guess 0 is
    always submitted; guess i >= 1 is drafted and submitted only while the draft's margin
    (top-1 minus top-2 logit) is > gate:
      "peek" (default, = tools/bakeoff.py conf>t): guess i is submitted iff ITS OWN margin > gate.
             The draft pass that produced a refused guess is spent (its input, guess i-1, stays
             in the draft cache, so it is not lost when guess i-1 is accepted). C: run_ref.c RR_GATE.
      "next": guess i+1 is drafted iff the margin of guess i > gate (no spent pass; Python only)."""
    dm.reset(); tm.reset()
    seq = list(prompt)
    end = len(prompt) + n_new
    st = dict(rounds=0, acc=0, prop=0, tpass=0, dpass=0, catchup=0, peek=0)
    tm.feed(seq[:-1])                                        # both caches hold all but the last token
    dm.feed(seq[:-1])
    while len(seq) < end:
        g = min(gamma, end - len(seq) - 1)
        d0 = st["dpass"]
        # draft: first catch up on every token it has not seen except the last, then guess
        behind = len(seq) - 1 - dm.n
        if behind > 0:
            dm.feed(seq[dm.n:len(seq) - 1]); st["catchup"] += behind
        guesses = []
        d_last = seq[-1]
        mg = None                                            # margin of the last drafted guess
        while len(guesses) < g:
            if gate is not None and gate_mode == "next" and guesses and mg <= gate:
                break
            [t], mg = dm.feed_m([d_last]); st["dpass"] += 1
            if gate is not None and gate_mode == "peek" and guesses and mg <= gate:
                st["peek"] += 1                              # drafted, not submitted
                break
            guesses.append(t); d_last = t
        g = len(guesses)
        # target: one verify pass over K = g + 1 tokens (last real token + g guesses)
        base = tm.n
        want = tm.feed([seq[-1]] + guesses)
        st["tpass"] += 1
        n_ok = 0
        while n_ok < g and guesses[n_ok] == want[n_ok]:
            n_ok += 1
        seq += guesses[:n_ok] + [want[n_ok]]
        tm.truncate(base + 1 + n_ok)                        # keep the last real token + accepted guesses
        dm.truncate(min(dm.n, len(seq) - 1))
        st["rounds"] += 1; st["acc"] += n_ok; st["prop"] += g
        if trace is not None:                                # demo replay (tools/demo.py); no effect on results
            trace.append(dict(guesses=guesses, n_ok=n_ok, fix=int(want[n_ok]), behind=max(behind, 0), dpass=st["dpass"] - d0))
    return seq[len(prompt):end], st


# ---------------- checks ----------------
def selftest():
    ok = True
    # helpers
    for num, den in [(1, 3), (127, 5000), (10**9, 7), (3, 2**40)]:
        m, e = norm_scale(num, den)
        ok &= M_LO <= m < M_HI and abs(m * 2.0 ** -e - num / den) <= 2.0 ** -e
    m, e = smul(M_LO, 30, M_LO, 30)                         # 0.5 * 0.5
    ok &= abs(int(m) * 2.0 ** -int(e) - 0.25) < 1e-9
    ok &= int(apply(np.array([3]), M_LO, 30, 0)[0]) == 2   # 3 * 0.5 = 1.5 -> 2 (half up)
    q, m, e = quant(np.array([4096, -2048, 0]), 12)
    ok &= q.tolist() == [127, -63, 0]  # -63.5 rounds half up to -63 and abs(m * 2.0 ** -e * 127 - 1.0) < 1e-6
    print("helpers:", "PASS" if ok else "FAIL")
    if not (EXPORT / "draft.npz").exists():
        print("no export found; run python model/export.py"); return ok
    # one-by-one vs a fresh feed of the same tokens (determinism + rollback)
    d = QModel("draft")
    toks = list(range(5, 25))
    a = d.feed(toks)
    d.truncate(10); b = d.feed(toks[10:])
    det = a[10:] == b
    print("rollback then refeed gives the same outputs:", "PASS" if det else "FAIL")
    ok &= det
    # spec identity: draft = target model must accept everything; real pair must be identical
    t = QModel("target"); d2 = QModel("draft")
    p = toks[:16]
    ref = qgreedy(t, p, 12)
    out, st = qspec(d2, t, p, 12, 3)
    same = out == ref
    print(f"integer spec == integer greedy (draft/target, gamma 3): {'PASS' if same else 'FAIL'} "
          f"(rounds {st['rounds']}, accepted {st['acc']}/{st['prop']})")
    tt = QModel("target")
    out2, st2 = qspec(QModel("target"), tt, p, 12, 3)
    full = out2 == ref and st2["acc"] == st2["prop"]
    print(f"draft = target accepts every guess: {'PASS' if full else 'FAIL'}")
    ok &= same and full
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def float_models():
    sys.path.insert(0, str(ROOT / "model"))
    import spec  # noqa: E402
    from model import DRAFT, TARGET  # noqa: E402
    return spec, spec.load(DRAFT, "draft"), spec.load(TARGET, "target")


def agree(n_prompts, plen, n_new):
    """Float64 vs integer. Teacher-forced top-1 agreement on the float greedy text, and the
    first position where free-running integer greedy leaves float greedy."""
    import torch
    spec, fd, ft = float_models()
    ps = spec.prompts(n_prompts, plen)
    print(f"{n_prompts} prompts, {plen} prompt tokens, {n_new} new; float64 vs integer")
    for name, fm in (("draft", fd), ("target", ft)):
        qm = QModel(name)
        top1 = tot = same_runs = 0
        first_div = []
        for p in ps:
            ref = spec.greedy(fm, p, n_new)
            with torch.no_grad():
                fl = spec.logits_at(fm, p + ref[:-1])[len(p) - 1:].argmax(-1).tolist()
            qm.reset()
            qa = qm.feed(p + ref[:-1])[len(p) - 1:]
            top1 += sum(a == b for a, b in zip(qa, fl)); tot += len(fl)
            qg = qgreedy(qm, p, n_new)
            same_runs += qg == ref
            first_div.append(next((i for i, (a, b) in enumerate(zip(qg, ref)) if a != b), n_new))
        print(f"  {name:6s}: teacher-forced top-1 agreement {top1 / tot:.3f} ({top1}/{tot}) | "
              f"greedy identical {same_runs}/{n_prompts} | mean first divergence at token "
              f"{np.mean(first_div):.1f} of {n_new} | max |residual| {qm.stats['max_resid']} (Q12)")


def spec_run(n_prompts, plen, n_new, gammas, gates=(), gmax=4, modes=("peek",), cd=0.084, lane=0.132, seed=0):
    """Fixed gammas, then the confidence gate (Q12 thresholds, cap gmax) in each mode.
    tok/cost: cost of a round = 1 + n * (cd + lane) (n = guesses submitted) + cd per draft pass
    that was not submitted (peek), same model as tools/bakeoff.py."""
    spec, _, _ = float_models()
    ps = spec.prompts(n_prompts, plen, seed=seed)
    dm, tm = QModel("draft"), QModel("target")
    refs = [qgreedy(tm, p, n_new) for p in ps]
    print(f"integer speculative decoding, {n_prompts} prompts x {n_new} tokens "
          f"(tok/cost: c_draft {cd:g}, lane {lane:g}, peek charged)")
    runs = [(f"gamma {g}", g, None, "peek") for g in gammas]
    runs += [(f"gate {mode} >{t} gmax {gmax}", gmax, t, mode) for mode in modes for t in gates]
    for name, g, t, mode in runs:
        same = 0; tot = dict(rounds=0, acc=0, prop=0, tpass=0, dpass=0, catchup=0, peek=0)
        for p, r in zip(ps, refs):
            out, st = qspec(dm, tm, p, n_new, g, t, mode)
            same += out == r
            for k in tot: tot[k] += st[k]
        cost = tot["rounds"] + tot["prop"] * (cd + lane) + tot["peek"] * cd
        print(f"  {name}: identical {same}/{n_prompts} | tokens/target pass "
              f"{n_prompts * n_new / tot['rounds']:.3f} | tok/cost {n_prompts * n_new / cost:.3f} | "
              f"accepted {tot['acc']}/{tot['prop']} | draft passes {tot['dpass']} (peek {tot['peek']}) | "
              f"catch-up {tot['catchup']} ({tot['catchup'] / tot['rounds']:.2f} per round)")


def golden(plen=16, n_new=8, gamma=2, fname="golden.h", gate=None, seed=1, set_n=1, set_i=0):
    """ref/golden.h: one val prompt, per-step argmax + logit sums for both models, spec stats.
    ref/run_ref.c must reproduce every number (bit-exact check of the C reference).
    gate (Q12): spec rounds with the D14 confidence gate ("peek"), gamma = the cap; adds
    G_GATE, G_GMAX, G_SPEC_PEEK (ref/golden_gate.h, used by run_ref.c -DRR_GATE)."""
    spec, _, _ = float_models()
    p = spec.prompts(set_n, plen, seed=seed)[set_i]     # set_n/set_i: prompt i of a --spec test set
    lines = ["/* GENERATED by python model/qmodel.py --golden. Do not edit. */",
             "#ifndef GOLDEN_H", "#define GOLDEN_H", "#include <stdint.h>",
             f"#define G_PLEN {plen}", f"#define G_NEW {n_new}", f"#define G_GAMMA {gamma}",
             "static const int16_t g_prompt[G_PLEN] = {" + ", ".join(map(str, p)) + "};"]
    for name in ("draft", "target"):
        m = QModel(name)
        toks, sums = [], []
        seq = list(p)
        for i in range(plen + n_new - 1):                  # teacher-free: feed prompt then own picks
            lg = m.step(seq[i])
            t = int(np.argmax(lg)); s_ = int(lg.sum())
            if i >= plen - 1:
                toks.append(t); sums.append(s_)
                if len(seq) == i + 1:
                    seq.append(t)
        lines.append(f"static const int16_t g_{name}_tok[G_NEW] = {{" + ", ".join(map(str, toks)) + "};")
        lines.append(f"static const int64_t g_{name}_lsum[G_NEW] = {{" + ", ".join(f"{v}LL" for v in sums) + "};")
    out, st = qspec(QModel("draft"), QModel("target"), p, n_new, gamma, gate)
    assert out == toks, "spec output != target greedy"
    lines += [f"static const int16_t g_spec_tok[G_NEW] = {{" + ", ".join(map(str, out)) + "};",
              f"#define G_SPEC_ROUNDS {st['rounds']}", f"#define G_SPEC_ACC {st['acc']}",
              f"#define G_SPEC_CATCHUP {st['catchup']}"]
    if gate is not None:
        lines += [f"#define G_GATE {gate}", f"#define G_GMAX {gamma}", f"#define G_SPEC_PEEK {st['peek']}"]
    lines += ["#endif", ""]
    (ROOT / "ref" / fname).write_text("\n".join(lines))
    print(f"wrote ref/{fname}:", out, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--golden-extra", nargs=4, metavar=("SEED", "PLEN", "NEW", "FILE"),
                    help="only write one extra golden header (multi-prompt core runs): ref/FILE from val prompt SEED")
    ap.add_argument("--set", nargs=2, type=int, default=[1, 0], metavar=("N", "I"),
                    help="--golden-extra: use prompt I of prompts(N, PLEN, SEED) (= the --spec test set order)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--agree", action="store_true")
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="--spec: prompt sample seed (0 = the reported test set)")
    ap.add_argument("--plen", type=int, default=32)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--gate", type=int, nargs="*", default=[],
                    help="--spec: confidence-gate thresholds (Q12; 4096 = 1.0 logit); --golden: the first one "
                         "goes into ref/golden_gate.h (default 4096)")
    ap.add_argument("--gmax", type=int, default=4, help="--spec / --golden: gamma cap for the gate")
    ap.add_argument("--gate-mode", nargs="+", default=["peek"], choices=["peek", "next"])
    ap.add_argument("--cdraft", type=float, default=0.084, help="--spec tok/cost: one draft pass in target passes")
    ap.add_argument("--lane", type=float, default=0.132, help="--spec tok/cost: extra verify cost per guess")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.golden_extra:
        sd, pl, nw, fn = a.golden_extra
        return golden(int(pl), int(nw), 2, fn, seed=int(sd), set_n=a.set[0], set_i=a.set[1])
    if a.golden:
        golden()
        golden(3, 2, 1, "golden_small.h")              # short run for the core in xsim (-DRR_SMALL)
        golden(16, 8, a.gmax, "golden_gate.h", a.gate[0] if a.gate else 4096)   # -DRR_GATE
    if a.agree:
        agree(a.prompts, a.plen, a.new)
    if a.spec:
        spec_run(a.prompts, a.plen, a.new, a.gammas, a.gate, a.gmax, a.gate_mode, a.cdraft, a.lane, a.seed)


if __name__ == "__main__":
    main()
