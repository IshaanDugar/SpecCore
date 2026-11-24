"""SpecCore cycle model. Every number this file prints is an ESTIMATE, not a measurement.

What it models (one forward pass of a model on the core, K tokens side by side):
  weight_stream       every weight matrix streamed once through the weight port (VDOTK);
                      costs bytes / (port bytes per cycle), the same for any K <= hw.lanes.
                      K > lanes: the weights stream ceil(K / lanes) times.
  readout             per output row: ACCRD + fixed-point scale + store for each of the K lanes, plus the loop
  attention           software: scalar int8 MACs on the RV32IM core (scores q.k and the value mix), K times
                      kv_stream (O1): the KV cache streamed through the same port, shared by the K lanes
  norms_rope_softmax  integer RMSNorm, RoPE and softmax, in software, once per token
  other               residual adds, embedding / KV-cache copies, argmax, per-matrix and per-pass setup

A speculative round = gamma draft passes (K = 1) + one target pass (K = gamma + 1).
Speedup = plain cycles per token (one target pass at K = 1) / (round cycles / tokens per round).

What it ignores: DDR latency and refresh, weight-FIFO empty stalls, cache effects, the growth of the
context inside a round (every lane is charged the same ctx), and the core's branch-mispredict and
load-use stalls (unless you raise SW_STALL_FACTOR). The draft-length predictor (D12) is not modelled.

All numbers are estimates until the same passes are measured in xsim (cycle counter MMIO).

  python tools/cyclemodel.py                       tables
  python tools/cyclemodel.py --json out.json       same content as JSON
  python tools/cyclemodel.py --alpha 0.65 --gamma 2 4 --ctx 64
  python tools/cyclemodel.py --measured 2:1.93 4:2.21
  python tools/cyclemodel.py --set target.dim=192 hw.lanes=4 cost.SW_MAC_CYCLES=6
  python tools/cyclemodel.py --attention kv_stream  (only that mode)
  python tools/cyclemodel.py --selftest
  python tools/cyclemodel.py --calib build/ksweep_core.txt   (MEASURED pass costs, see cal_report)
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sc  # noqa: E402

# ---------------- cost constants (cycles). Source of each number is in its comment. ----------------
# From the docs:
WEIGHT_WORD_CYCLES = 1.0    # cycles per weight-port word. decisions.md D3 option B ("1 cycle per word") and
                            # isa_options.md row 1 (5.24 MB at 4 B/cycle). Assumes VDOTK issues back to back
                            # (unrolled or counted), with no loop branch per word. Not yet measured.
SW_MAC_CYCLES = 4.0         # software attention, cycles per int8 MAC. isa_options.md: "~4 cycles/MAC
                            # (lw, lw, mul, add)". mul is 1 cycle on the DSP (core_facts.md, 12 Nov update).
                            # Does not include a load-use stall or loop overhead per MAC.
READOUT_LOOP_CYCLES = 5.0   # per output row, K-independent part (pointer bump, compare, branch, SETP).
READOUT_LANE_CYCLES = 6.0   # per output row per lane (ACCRD, mulh scale, shift, clamp, store).
                            # isa_options.md gives only the sum: "K x ACCRD + K x scale + loop ~ 35 cycles/row"
                            # at K = 5. The split 5 + 6K is an ASSUMPTION chosen so K = 5 gives 35.
DIV_CYCLES = 32.0           # one divide on MCycle. core_facts.md (measured 12 Nov: 136 divides x 32 cycles).
# Assumptions (no source in the docs; instruction counts for a plain C loop, to be replaced by xsim numbers):
KV_WORD_CYCLES = 1.0        # kv_stream: cycles per KV-cache word; assumed equal to a weight word (O1 reuses the port).
RMSNORM_ELEM_CYCLES = 12.0  # per element: square + accumulate (~4) then scale, clamp, store (~8).
RMSNORM_FIXED_CYCLES = 100.0  # per norm: integer inverse square root; includes one divide (DIV_CYCLES) + setup.
ROPE_ELEM_CYCLES = 8.0      # per element of q and k: table load, 2 mul, add, shift, store.
SOFTMAX_ELEM_CYCLES = 20.0  # per score: max scan (~4), table exp + sum (~10), normalise + store (~6).
SOFTMAX_ROW_CYCLES = 60.0   # per softmax row (one head, one token): one divide for 1/sum + setup.
RESID_ELEM_CYCLES = 6.0     # residual add per element: lb, lb, add, clamp, sb, loop share.
COPY_ELEM_CYCLES = 2.0      # per byte: embedding row copy into the bank, k and v copy into the KV cache.
ARGMAX_ELEM_CYCLES = 5.0    # per logit: load, compare, branch, loop. (isa_options O3 says the argmax is
                            # 3-5% of a pass; this constant gives about 1-2%. Left as is, see the report.)
MATRIX_SETUP_CYCLES = 20.0  # per weight matrix: SETK / SETP, pointers, call overhead.
PASS_FIXED_CYCLES = 200.0   # per pass: position bookkeeping, accept loop.
SW_STALL_FACTOR = 1.0       # multiplies every software part (not the streams). 1.0 = the core's branch and
                            # load-use stalls are ignored. ISA test measured: 72 load-use stalls, 44 x 2-cycle
                            # mispredicts in ~9.8K cycles (HANDOFF section 10), so about 1.02 on that program.

COSTS = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, float)}

# measured 13 Nov, 200 prompts (checkpoints/spec_run200.out): tokens per target pass for each gamma
MEASURED_TOKENS_PER_PASS = {1: 1.56, 2: 1.85, 3: 2.05, 4: 2.15, 6: 2.24, 8: 2.28}
DEFAULT_ALPHAS = [0.59, 0.7]          # 0.59 = measured per-guess agreement (13 Nov); 0.7 = the earlier assumption
DEFAULT_GAMMAS = [1, 2, 3, 4, 6, 8]
DEFAULT_CTXS = [32, 64, 128]
PARTS = ("weight_stream", "readout", "attention", "norms_rope_softmax", "other")


# ---------------- the model ----------------
def model_from(c, name):
    """Plain model description from the loaded toml: dim, layers, heads, mlp_mult, vocab."""
    m = c[name]
    return dict(name=name, dim=m["dim"], layers=m["layers"], heads=m["heads"],
                mlp_mult=m["mlp_mult"], vocab=c["tokenizer"]["vocab"])


def matrices(m):
    """(rows, cols) of every streamed matrix in one pass. Names follow model/model.py."""
    d, h = m["dim"], m["mlp_mult"] * m["dim"]
    per_layer = [(3 * d, d),   # attn.qkv
                 (d, d),       # attn.proj
                 (h, d),       # mlp.fc1
                 (d, h)]       # mlp.fc2
    return per_layer * m["layers"] + [(m["vocab"], d)]   # tied output layer = embedding table


def weight_bytes(m, weight_bits=8):
    return sum(r * c for r, c in matrices(m)) * weight_bits // 8


def pass_cycles(m, K, ctx, hw=None, attention=None, weight_bits=8, costs=None):
    """Estimated cycles for one pass of model m over K tokens with ctx tokens of context.

    Returns {weight_stream, readout, attention, norms_rope_softmax, other, total} (floats, cycles).
    """
    hw = hw or {}
    k = dict(COSTS, **(costs or {}))
    lanes = hw.get("lanes", 8)
    port_bytes = hw.get("weight_port_bits", 32) // 8
    attention = attention or hw.get("attention", "software")
    if attention not in ("software", "kv_stream"):
        raise ValueError(f"attention must be software or kv_stream, not {attention}")
    d, L, H = m["dim"], m["layers"], m["heads"]
    hd = d // H
    mats = matrices(m)
    repeats = math.ceil(K / lanes)            # K > lanes: stream everything again for the next group
    stall = k["SW_STALL_FACTOR"]
    row = k["READOUT_LOOP_CYCLES"] * repeats + k["READOUT_LANE_CYCLES"] * K   # one output row, all K lanes

    weight_stream = repeats * math.ceil(weight_bytes(m, weight_bits) / port_bytes) * k["WEIGHT_WORD_CYCLES"]
    readout = stall * sum(r for r, _ in mats) * row

    # Attention, per token and layer: scores = heads x ctx dot products of head_dim; value mix = heads x
    # head_dim outputs, each a sum over ctx. Both are ctx x dim MACs per layer.
    macs_per_token = L * 2 * ctx * d
    if attention == "software":
        attn = stall * K * macs_per_token * k["SW_MAC_CYCLES"]
    else:
        # O1: keys (ctx x dim bytes) and transposed values (dim x ctx bytes) stream once for all K lanes.
        kv_words = math.ceil(L * 2 * ctx * d / port_bytes)
        kv_rows = L * (H * ctx + d)           # one score per (head, position); one output per value row
        attn = repeats * kv_words * k["KV_WORD_CYCLES"] + stall * kv_rows * row

    norms = (2 * L + 1) * (d * k["RMSNORM_ELEM_CYCLES"] + k["RMSNORM_FIXED_CYCLES"])
    rope = L * 2 * d * k["ROPE_ELEM_CYCLES"]
    softmax = L * H * (ctx * k["SOFTMAX_ELEM_CYCLES"] + k["SOFTMAX_ROW_CYCLES"])
    nrs = stall * K * (norms + rope + softmax)

    per_token = (2 * L * d * k["RESID_ELEM_CYCLES"]                 # two residual adds per layer
                 + (d + L * 2 * d) * k["COPY_ELEM_CYCLES"]          # embedding row + k, v into the cache
                 + m["vocab"] * k["ARGMAX_ELEM_CYCLES"])            # pick the top logit
    other = stall * (K * per_token + repeats * len(mats) * k["MATRIX_SETUP_CYCLES"] + k["PASS_FIXED_CYCLES"])

    out = dict(weight_stream=weight_stream, readout=readout, attention=attn,
               norms_rope_softmax=nrs, other=other)
    out["total"] = sum(out.values())
    return out


def round_cycles(gamma, ctx, draft, target, **kw):
    """One speculative round: gamma draft passes at K = 1, then one target pass at K = gamma + 1."""
    return gamma * pass_cycles(draft, 1, ctx, **kw)["total"] + pass_cycles(target, gamma + 1, ctx, **kw)["total"]


def expected_tokens(alpha, gamma):
    """Tokens kept per round if each guess agrees with probability alpha: 1 + a + ... + a^gamma."""
    return sum(alpha ** i for i in range(gamma + 1))


def speedup(tokens_per_round, gamma, ctx, draft, target, **kw):
    """Plain cycles per token (target pass, K = 1) / speculative cycles per token."""
    plain = pass_cycles(target, 1, ctx, **kw)["total"]
    return plain / (round_cycles(gamma, ctx, draft, target, **kw) / tokens_per_round)


def speedup_weight_only(tokens_per_round, gamma, draft, target, lanes=8):
    """The older estimate in HANDOFF section 10 / sc.py show: weight stream only, nothing else counted."""
    cost = weight_bytes(draft) / weight_bytes(target)
    return tokens_per_round / (math.ceil((gamma + 1) / lanes) + gamma * cost)


# ---------------- report ----------------
def build(c, ctxs, gammas, alphas, measured, modes, costs, bd_gamma):
    draft, target = model_from(c, "draft"), model_from(c, "target")
    hw, wb = c["hw"], c["quant"]["weight_bits"]
    mhz = hw["clock_mhz"]

    def kw(mode):
        return dict(hw=hw, attention=mode, weight_bits=wb, costs=costs)

    rep = {
        "label": "estimated (cycle model, not measured; see tools/cyclemodel.py docstring)",
        "config": {"draft": draft, "target": target, "weight_bits": wb, "clock_mhz": mhz,
                   "hw": {x: hw[x] for x in ("lanes", "weight_port_bits")},
                   "attention_modes": modes, "ctx": ctxs, "gammas": gammas, "alphas": alphas,
                   "breakdown_gamma": bd_gamma},
        "constants": dict(COSTS, **costs),
        "weight_bytes": {"draft": weight_bytes(draft, wb), "target": weight_bytes(target, wb)},
        "breakdown": [], "speedup_alpha": [], "speedup_measured": [],
    }
    for mode in modes:
        for ctx in ctxs:
            for name, m, K in (("draft", draft, 1), ("target", target, 1), ("target", target, bd_gamma + 1)):
                p = pass_cycles(m, K, ctx, **kw(mode))
                rep["breakdown"].append(dict(attention=mode, ctx=ctx, model=name, K=K, cycles=p,
                                             ms=p["total"] / mhz / 1e3, estimated=True))
        for ctx in ctxs:
            for a in alphas:
                for g in gammas:
                    t = expected_tokens(a, g)
                    rep["speedup_alpha"].append(dict(
                        attention=mode, ctx=ctx, alpha=a, gamma=g, tokens_per_round=t,
                        speedup=speedup(t, g, ctx, draft, target, **kw(mode)),
                        speedup_weight_only=speedup_weight_only(t, g, draft, target, hw["lanes"]),
                        estimated=True))
            for g, t in sorted(measured.items()):
                rep["speedup_measured"].append(dict(
                    attention=mode, ctx=ctx, gamma=g, tokens_per_round=t,
                    tokens_source="measured 13 Nov, 200 prompts" if measured is MEASURED_TOKENS_PER_PASS
                    else "given on the command line",
                    speedup=speedup(t, g, ctx, draft, target, **kw(mode)),
                    speedup_weight_only=speedup_weight_only(t, g, draft, target, hw["lanes"]),
                    estimated=True))
    return rep


def fmt_m(x):
    return f"{x / 1e6:.3f}M"


def print_report(rep):
    cfg = rep["config"]
    ctxs, modes = cfg["ctx"], cfg["attention_modes"]
    print("SpecCore cycle model. ALL NUMBERS ARE ESTIMATES (not measured in xsim).")
    wbt, wbd = rep["weight_bytes"]["target"], rep["weight_bytes"]["draft"]
    print(f"  weights streamed per pass: target {wbt / 1e6:.2f} MB, draft {wbd / 1e6:.3f} MB "
          f"(draft / target = {wbd / wbt:.3f}); clock {cfg['clock_mhz']} MHz, lanes {cfg['hw']['lanes']}, "
          f"port {cfg['hw']['weight_port_bits']} bits")

    for mode in modes:
        print(f"\nPass breakdown, attention = {mode} (estimated cycles)")
        print(f"  {'ctx':>4} {'pass':<12}{'weights':>9}{'readout':>9}{'attention':>10}{'norm/rope/sm':>13}"
              f"{'other':>8}{'TOTAL':>9}{'ms':>7}")
        for r in rep["breakdown"]:
            if r["attention"] != mode:
                continue
            p = r["cycles"]
            print(f"  {r['ctx']:>4} {r['model'] + ' K=' + str(r['K']):<12}{fmt_m(p['weight_stream']):>9}"
                  f"{fmt_m(p['readout']):>9}{fmt_m(p['attention']):>10}{fmt_m(p['norms_rope_softmax']):>13}"
                  f"{fmt_m(p['other']):>8}{fmt_m(p['total']):>9}{r['ms']:>7.1f}")

    head = "".join(f"{'ctx ' + str(x):>9}" for x in ctxs)
    for mode in modes:
        for a in cfg["alphas"]:
            print(f"\nEstimated speedup vs plain decoding, attention = {mode}, alpha = {a} "
                  f"(tokens/round = 1 + a + ... + a^gamma)")
            print(f"  {'gamma':>5}{'tok/round':>10}{head}{'weights only':>14}")
            for g in cfg["gammas"]:
                rows = [r for r in rep["speedup_alpha"]
                        if r["attention"] == mode and r["alpha"] == a and r["gamma"] == g]
                print(f"  {g:>5}{rows[0]['tokens_per_round']:>10.2f}"
                      + "".join(f"{r['speedup']:>8.2f}x" for r in rows)
                      + f"{rows[0]['speedup_weight_only']:>13.2f}x")
    for mode in modes:
        rows = [r for r in rep["speedup_measured"] if r["attention"] == mode]
        if not rows:
            continue
        print(f"\nEstimated speedup with tokens/target pass from: {rows[0]['tokens_source']}; attention = {mode}")
        print(f"  {'gamma':>5}{'tok/pass':>10}{head}{'weights only':>14}")
        for g in sorted({r["gamma"] for r in rows}):
            rg = [r for r in rows if r["gamma"] == g]
            print(f"  {g:>5}{rg[0]['tokens_per_round']:>10.2f}" + "".join(f"{r['speedup']:>8.2f}x" for r in rg)
                  + f"{rg[0]['speedup_weight_only']:>13.2f}x")
        for ctx in ctxs:
            best = max((r for r in rows if r["ctx"] == ctx), key=lambda r: r["speedup"])
            print(f"  best gamma at ctx {ctx}: {best['gamma']} ({best['speedup']:.2f}x, estimated)")
    print("\n'weights only' = the older estimate (HANDOFF section 10): weight stream alone, draft = "
          f"{wbd / wbt:.3f} of a target pass per guess.")
    print("Estimates only. Ignored: DDR latency/refresh, FIFO stalls, cache effects, core stalls (SW_STALL_FACTOR).")


def selftest(c):
    """Checks against the numbers already written down in HANDOFF section 10 and isa_options.md."""
    c["draft"].update(dim=128, layers=2, heads=4, mlp_mult=4)
    c["target"].update(dim=256, layers=6, heads=8, mlp_mult=4)
    c["tokenizer"]["vocab"] = 2048
    d, t = model_from(c, "draft"), model_from(c, "target")
    hw = dict(lanes=8, weight_port_bits=32)
    p1 = pass_cycles(t, 1, 64, hw=hw, attention="software")
    p5 = pass_cycles(t, 5, 64, hw=hw, attention="software")
    assert weight_bytes(t) == 5_242_880 == sc.streamed_bytes(c["target"], 2048, 8)
    assert p1["weight_stream"] == p5["weight_stream"] == 1_310_720          # 1.31M, K-independent
    assert pass_cycles(t, 8, 64, hw=hw)["weight_stream"] == 1_310_720
    assert pass_cycles(t, 9, 64, hw=hw)["weight_stream"] == 2 * 1_310_720   # K > lanes: second stream
    assert weight_bytes(d) / weight_bytes(t) == 0.125
    assert sum(r for r, _ in matrices(t)) == 15_872                         # isa_options: 15.9K rows
    assert p5["readout"] == 15_872 * 35                                     # ~0.55M
    assert p5["attention"] == 5 * 196_608 * 4                               # ~3.9M
    assert abs(speedup_weight_only(expected_tokens(0.7, 4), 4, d, t) - 1.85) < 0.005
    assert abs(speedup_weight_only(1.93, 2, d, t) - 1.54) < 0.005
    assert abs(sum(v for k, v in p5.items() if k != "total") - p5["total"]) < 1e-6
    assert abs(expected_tokens(1.0, 4) - 5.0) < 1e-12
    print("SELFTEST PASS")


# ---------------- calibrated mode: pass costs measured in xsim (run_ref.c -DRR_KSWEEP) ----------------
# draft catch-up tokens per round (audit e3), measured 14 Nov, integer model, 20 prompts (checkpoints/qspec20.out)
MEASURED_CATCHUP = {1: 0.53, 2: 0.30, 4: 0.12}
CAL_BUCKETS = ("matvec", "attn", "normq", "other")


def read_ksweep(path):
    """{model: {K: {bucket: cycles, 'total': cycles}}, 'ctx': n0} from 'cycles ksweep' lines."""
    out, ctx = {}, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("cycles ksweep "):
            continue
        t = line.split()
        name, K = t[2], int(t[4])
        vals = {t[i]: int(t[i + 1]) for i in range(5, len(t) - 3, 2)}
        out.setdefault(name, {})[K] = vals
        ctx = int(line.rsplit(" ", 1)[1].rstrip(")"))
    if not out:
        sys.exit(f"no 'cycles ksweep' lines in {path}")
    return out, ctx


def fit_line(ks, ys):
    """Least squares y = S + P * K. Returns S, P, worst relative error of the fit."""
    n = len(ks)
    mk, my = sum(ks) / n, sum(ys) / n
    P = sum((k - mk) * (y - my) for k, y in zip(ks, ys)) / sum((k - mk) ** 2 for k in ks)
    S = my - P * mk
    err = max(abs(S + P * k - y) / y for k, y in zip(ks, ys) if y)
    return S, P, err


def calibrate(sweep):
    """{model: {bucket: (shared, per_lane, fit_err)}} plus 'total'."""
    cal = {}
    for name, byk in sweep.items():
        ks = sorted(byk)
        cal[name] = {b: fit_line(ks, [byk[k][b] for k in ks]) for b in CAL_BUCKETS + ("total",)}
    return cal


def cal_pass(cal, name, K):
    S, P, _ = cal[name]["total"]
    return S + P * K


def cal_round(cal, gamma, catchup):
    """gamma draft passes (the first also carries the catch-up tokens as extra lanes) + one K = gamma+1 verify."""
    d = gamma * cal_pass(cal, "draft", 1) + catchup * cal["draft"]["total"][1]
    return d, cal_pass(cal, "target", gamma + 1)


def cal_report(path, tokens):
    sweep, ctx = read_ksweep(path)
    cal = calibrate(sweep)
    print(f"Calibrated from {path} (MEASURED pass costs in xsim, ctx {ctx} before the pass; "
          "speedups below = measured pass costs x measured tokens/round).")
    print(f"\n  {'model':<7}{'bucket':<8}" + "".join(f"{'K=' + str(k):>11}" for k in sorted(sweep['target']))
          + f"{'shared':>11}{'per lane':>11}{'fit err':>9}")
    for name in ("draft", "target"):
        for b in CAL_BUCKETS + ("total",):
            S, P, e = cal[name][b]
            print(f"  {name:<7}{b:<8}" + "".join(f"{sweep[name][k][b]:>11,}" for k in sorted(sweep[name]))
                  + f"{S:>11,.0f}{P:>11,.0f}{e * 100:>8.1f}%")
    for name in ("draft", "target"):
        S, P, _ = cal[name]["total"]
        print(f"  {name}: one extra lane costs {P / (S + P):.2f} of a K=1 pass (shared part {S / (S + P):.2f})")
    print(f"  draft / target, K=1: {cal_pass(cal, 'draft', 1) / cal_pass(cal, 'target', 1):.3f}")

    plain = cal_pass(cal, "target", 1)
    print(f"\n  {'gamma':>5}{'tok/round':>10}{'catch-up':>9}{'draft':>10}{'verify':>10}{'per token':>11}{'speedup':>9}")
    for g in sorted(tokens):
        cu = MEASURED_CATCHUP.get(g)
        if cu is None:
            continue
        d, v = cal_round(cal, g, cu)
        per_tok = (d + v) / tokens[g]
        print(f"  {g:>5}{tokens[g]:>10.2f}{cu:>9.2f}{d / 1e6:>9.2f}M{v / 1e6:>9.2f}M{per_tok / 1e6:>10.2f}M"
              f"{plain / per_tok:>8.2f}x")
    print(f"  plain: {plain / 1e6:.2f}M cycles per token (target, K=1)")
    # check against the measured gamma-2 golden run. Its rounds (run_ref.c -DRR_SPECLOG, PC):
    # (gamma, catch-up) = (2,0) (2,0) (2,0) (1,1) (0,0): near the end of the output gamma shrinks.
    # A round with gamma >= 1: first draft pass carries the catch-up lanes, then gamma-1 K=1 passes.
    pd = sum(cal_pass(cal, "draft", 1 + cu) + (g - 1) * cal_pass(cal, "draft", 1)
             for g, cu in GOLDEN_G2_ROUNDS if g)
    pv = sum(cal_pass(cal, "target", g + 1) for g, _ in GOLDEN_G2_ROUNDS)
    print(f"  model check, golden gamma-2 run: predicted draft {pd / 1e6:.2f}M + verify {pv / 1e6:.2f}M = "
          f"{(pd + pv) / 1e6:.2f}M cycles; measured 15 Nov (slow-path build, BP off, ctx 16-23): "
          "draft 6.89M + verify 38.99M = 45.88M")
    return cal


GOLDEN_G2_ROUNDS = [(2, 0), (2, 0), (2, 0), (1, 1), (0, 0)]


def cal_scaled(cal, cuts):
    """Copy of cal with per-lane parts scaled: cuts = {bucket: keep_fraction} for both models.
    'draft' as a bucket name scales the whole draft pass (a cheaper draft model)."""
    out = {}
    for name, bk in cal.items():
        out[name] = {}
        S_tot = P_tot = 0.0
        for b in CAL_BUCKETS:
            S, P, e = bk[b]
            P *= cuts.get(b, 1.0)
            if name == "draft":
                S, P = S * cuts.get("draft", 1.0), P * cuts.get("draft", 1.0)
            out[name][b] = (S, P, e)
            S_tot += S
            P_tot += P
        out[name]["total"] = (S_tot, P_tot, bk["total"][2])
    return out


def cal_speedup(cal, g, tokens):
    d, v = cal_round(cal, g, MEASURED_CATCHUP[g])
    return cal_pass(cal, "target", 1) / ((d + v) / tokens[g])


WHATIF = [  # (label, {bucket: fraction of the measured per-lane cost kept})
    ("measured (no change)", {}),
    ("matvec per-lane (readout+fill) -50%", {"matvec": 0.5}),
    ("matvec per-lane (readout+fill) -80%", {"matvec": 0.2}),
    ("norm/quant -50%", {"normq": 0.5}),
    ("norm/quant -80%", {"normq": 0.2}),
    ("attention -50%", {"attn": 0.5}),
    ("attention -80%", {"attn": 0.2}),
    ("readout+fill + norm/quant -80%", {"matvec": 0.2, "normq": 0.2}),
    ("all per-lane -80%", {"matvec": 0.2, "normq": 0.2, "attn": 0.2, "other": 0.2}),
    ("draft pass -50% (smaller draft)", {"draft": 0.5}),
    ("all per-lane -80% + draft -50%", {"matvec": 0.2, "normq": 0.2, "attn": 0.2, "other": 0.2, "draft": 0.5}),
]


def cal_whatif(cal, tokens):
    """Speedup if a per-lane bucket were cheaper. Plain decoding (K = 1) gets the same saving, so the
    gain is smaller than the saving on the verify pass alone. ESTIMATES built on measured costs."""
    gs = [g for g in sorted(tokens) if g in MEASURED_CATCHUP]
    print("\nWhat-if (ESTIMATE from the measured costs above; the plain K=1 pass gets the same saving):")
    print(f"  {'change':<38}{'plain/token':>12}" + "".join(f"{'g' + str(g):>8}" for g in gs))
    for label, cuts in WHATIF:
        c = cal_scaled(cal, cuts)
        print(f"  {label:<38}{cal_pass(c, 'target', 1) / 1e6:>11.2f}M"
              + "".join(f"{cal_speedup(c, g, tokens):>7.2f}x" for g in gs))


def cast(old, s):
    if isinstance(old, bool):
        return s.lower() in ("1", "true", "yes")
    if isinstance(old, int):
        return int(s)
    if isinstance(old, float):
        return float(s)
    return s


def main():
    ap = argparse.ArgumentParser(description="SpecCore cycle model (estimates only).")
    ap.add_argument("--json", metavar="PATH", help="also write everything as JSON")
    ap.add_argument("--alpha", type=float, nargs="+", help=f"per-guess agreement values (default {DEFAULT_ALPHAS})")
    ap.add_argument("--gamma", type=int, nargs="+",
                    help=f"gamma sweep (default {DEFAULT_GAMMAS}); a single value also sets the K = gamma+1 breakdown")
    ap.add_argument("--ctx", type=int, nargs="+", help=f"context lengths (default {DEFAULT_CTXS})")
    ap.add_argument("--attention", choices=("software", "kv_stream", "both"), default="both")
    ap.add_argument("--measured", nargs="*", metavar="GAMMA:TOKENS",
                    help="tokens per target pass per gamma, e.g. 2:1.93 4:2.21 (default: the 13 Nov set)")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="overrides: draft.dim=96 target.layers=4 tokenizer.vocab=2048 hw.lanes=4 "
                         "hw.weight_port_bits=64 hw.clock_mhz=65 cost.SW_MAC_CYCLES=6")
    ap.add_argument("--catchup", nargs="*", metavar="GAMMA:FRAC",
                    help="draft catch-up tokens per round per gamma (= fraction of rounds with every guess "
                         "accepted, from a guess log), e.g. 2:0.729 (default: the 14 Nov set)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calib", metavar="KSWEEP_TXT",
                    help="use MEASURED pass costs from a run_ref.c -DRR_KSWEEP core run instead of the constants")
    a = ap.parse_args()

    c = sc.load()
    if a.selftest:
        return selftest(c)
    if a.catchup:
        MEASURED_CATCHUP.clear()
        MEASURED_CATCHUP.update({int(g): float(f) for g, f in (x.split(":") for x in a.catchup)})
    if a.calib:
        tok = MEASURED_TOKENS_PER_PASS
        if a.measured:
            tok = {int(g): float(t) for g, t in (x.split(":") for x in a.measured)}
        cal = cal_report(a.calib, tok)
        return cal_whatif(cal, tok)
    costs = {}
    for item in a.set:
        key, _, val = item.partition("=")
        sec, _, name = key.partition(".")
        if sec == "cost":
            if name not in COSTS:
                sys.exit(f"unknown cost constant {name}; have: {', '.join(COSTS)}")
            costs[name] = float(val)
        elif sec in c and name in c[sec]:
            c[sec][name] = cast(c[sec][name], val)
        else:
            sys.exit(f"unknown override {key}")
    for name in ("draft", "target"):
        if c[name]["dim"] % c[name]["heads"]:
            sys.exit(f"{name}: dim not divisible by heads")

    measured = MEASURED_TOKENS_PER_PASS
    if a.measured:
        measured = {int(g): float(t) for g, t in (x.split(":") for x in a.measured)}
    gammas = a.gamma or DEFAULT_GAMMAS
    if a.gamma and not a.measured:
        measured_used = {g: t for g, t in measured.items() if g in gammas}
        measured = measured if len(measured_used) == len(measured) else measured_used
    bd_gamma = gammas[0] if a.gamma and len(a.gamma) == 1 else c["spec"]["gamma"]
    modes = ["software", "kv_stream"] if a.attention == "both" else [a.attention]
    rep = build(c, a.ctx or DEFAULT_CTXS, gammas, a.alpha or DEFAULT_ALPHAS, measured, modes, costs, bd_gamma)
    print_report(rep)
    if a.json:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
