"""Draft-length predictor bake-off (decision D12): replay a gamma = 8 guess log.

Speculative decoding round: the draft guesses up to gamma tokens one at a time, the
target checks them in one pass, guesses are kept up to the first mismatch, then the
target's own token is appended. A draft-length predictor decides, before each guess
is submitted, "keep guessing" or "stop and let the target verify" - a branch predictor
whose branch is "will my next guess be accepted?".

CAUSALITY RULE (what a predictor may look at)
  At the decision for slot i of a round the predictor may use ONLY:
    - the slot index i,
    - the draft's confidence for guess i (top-1 minus top-2 logit). The draft has
      already computed guess i; the decision is whether to SUBMIT it,
    - the last token (the one guess i follows),
    - outcomes of COMPLETED rounds: the global history of accept/reject bits.
  It does NOT know whether guesses 0..i-1 of the SAME round were accepted; that is
  only known after the target verifies. State (tables, weights, history) is updated
  once per round, after the round completes - online, as hardware would.

WHAT A PREDICTOR LEARNS FROM A ROUND (n = guesses submitted, run = leading accepts)
    - slots 0..min(n,run)-1 : accepted (1)
    - slot run, if run < n  : rejected (0)
    - slots after the first reject: nothing (the target never judged them)
    - the stop slot n, if n <= run: the guess the predictor declined. Its outcome is
      learned by comparing that guess with the token the target produced there
      ("shadow guess"). Without it a predictor that says stop never finds out it
      could have continued, and stays stuck. --no-shadow turns this off.
  The shadow guess / confidence peek at the stop slot is one draft pass whose guess
  is not submitted. The main metric does NOT charge for it (cost = n * c_draft + 1,
  as specified); the column "tok/cost+pk" charges one extra c_draft in every round
  where a predictor chose to stop before the slot limit.

REPLAY SCORING (per logged round with g slots, g = 8 except near the end of a prompt)
    tokens = min(n, run) + 1        cost = n * c_draft + 1      (n capped at g)
  n = 0 is plain decoding (1 token, cost 1). Rounds with no logged slots (one token
  left in the prompt) are reconstructed from the pos column and score 1 token, cost 1.
  APPROXIMATION: rounds are replayed independently. In a real run, stopping early
  shifts where the next round starts, so the following guesses would differ from the
  logged ones. The validation table shows how far this is from real runs.

DECISION ACCURACY: every continue/stop decision is marked in hindsight. "Continue" at
  slot i is right iff i < run (the guess is accepted); "stop" at slot n is right iff
  n >= run. A continue after the round's first mismatch counts as wrong.

Usage:
  python tools/bakeoff.py [--log logs/guesses_g8.csv] [--cost 0.125] [--json build/bakeoff.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMMA = 8
# spec.py, 20 prompts, real runs at gamma = n (tokens per target pass)
MEASURED_20 = {1: 1.57, 2: 1.93, 3: 2.05, 4: 2.21, 6: 2.33, 8: 2.37}
CONF_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
CONF_EDGES = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]      # thermometer code for the perceptron


# ---------------------------------------------------------------- log loading

def load_rounds(path, only_prompts=None):
    """Returns (rounds, info). A round is dict(prompt, slots=[(slot, conf, last_tok)], run).
    Incomplete prompts (file still being written) are dropped and reported."""
    per = {}                                    # prompt -> {round -> [rows]}
    bad = 0
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            try:
                p, rnd, slot, pos, last = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])
                acc = None if r[6] == "" else int(r[6])
                conf = float(r[7])
            except (ValueError, IndexError):
                bad += 1
                continue
            per.setdefault(p, {}).setdefault(rnd, []).append((slot, pos, last, acc, conf))
    prompts = {}
    for p, rds in per.items():
        out = []
        ok = True
        for rnd in sorted(rds):
            rows = sorted(rds[rnd])
            if [x[0] for x in rows] != list(range(len(rows))):
                ok = False
                break
            run = 0
            while run < len(rows) and rows[run][3] == 1:
                run += 1
            out.append(dict(prompt=p, start=rows[0][1], run=run, first=rows[0][2:],
                            slots=[(s, c, lt) for s, _, lt, _, c in rows]))
        if ok and out and sorted(rds) == list(range(len(out))):
            prompts[p] = out
        else:
            bad += 1
    if not prompts:
        sys.exit(f"no usable rounds in {path}")
    # every prompt generates up to the same end position; infer it, then drop prompts that
    # stop short (partial file) and add the unlogged 1-token round where one token was left
    ends = {p: r[-1]["start"] + r[-1]["run"] + 1 for p, r in prompts.items()}
    end = max(ends.values())
    start = min(r[0]["start"] for r in prompts.values())
    rounds, dropped, added, kept = [], [], 0, []
    for p in sorted(prompts):
        if only_prompts is not None and p not in only_prompts:
            continue
        if ends[p] == end - 1:
            prompts[p].append(dict(prompt=p, start=end - 1, run=0, slots=[]))
            added += 1
        elif ends[p] != end:
            dropped.append(p)
            continue
        kept.append(p)
        rounds += prompts[p]
    info = dict(path=str(path), prompts=len(kept), prompt_ids=kept, rounds=len(rounds),
                tokens_per_prompt=end - start, bad_rows=bad, dropped_prompts=dropped,
                zero_slot_rounds_added=added)
    return rounds, info


# ---------------------------------------------------------------- predictors
# features f = (slot, conf, last_tok). predict(f) -> True = submit this guess.
# update(obs) once per completed round; obs = [(f, bit)] for the slots whose outcome
# was learned (see docstring), in slot order.

def sat(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def tok_hash(tok, slot, bits):
    """Index from (last_tok, slot): fold the 11-bit token id, slot in the top 3 bits."""
    return (tok ^ (tok >> bits) ^ (slot << (bits - 3))) & ((1 << bits) - 1)


class Fixed:
    family = "fixed"; uses_conf = False; decides = False

    def __init__(self, n):
        self.n = n; self.name = f"fixed-{n}"
        self.bits = 0; self.bits_detail = "no state (constant n)"

    def predict(self, f): return f[0] < self.n
    def update(self, obs): pass


class ConfThreshold:
    family = "conf"; uses_conf = True; decides = True

    def __init__(self, t):
        self.t = t; self.name = f"conf>{t:g}"
        self.bits = 0; self.bits_detail = "no state (one comparator; threshold is a constant)"

    def predict(self, f): return f[1] > self.t
    def update(self, obs): pass


class Bimodal:
    """2-bit counters indexed by hash(last_tok, slot)."""
    family = "bimodal"; uses_conf = False; decides = True

    def __init__(self, idx_bits=10):
        self.b = idx_bits; self.t = [2] * (1 << idx_bits)
        self.name = "bimodal"
        self.bits = 2 << idx_bits
        self.bits_detail = f"{1 << idx_bits} x 2-bit counters"

    def predict(self, f): return self.t[tok_hash(f[2], f[0], self.b)] >= 2

    def update(self, obs):
        for f, bit in obs:
            i = tok_hash(f[2], f[0], self.b)
            self.t[i] = sat(self.t[i] + (1 if bit else -1), 0, 3)


class Gshare:
    """2-bit counters indexed by hash(last_tok, slot) XOR global outcome history."""
    family = "gshare"; uses_conf = False; decides = True

    def __init__(self, idx_bits=10, hist=8):
        self.b = idx_bits; self.h = hist; self.hist = 0
        self.t = [2] * (1 << idx_bits)
        self.name = "gshare"
        self.bits = (2 << idx_bits) + hist
        self.bits_detail = f"{1 << idx_bits} x 2-bit counters + {hist}-bit history"

    def _i(self, f): return tok_hash(f[2], f[0], self.b) ^ self.hist

    def predict(self, f): return self.t[self._i(f)] >= 2

    def update(self, obs):
        for f, bit in obs:                      # history is frozen during the round
            i = self._i(f)
            self.t[i] = sat(self.t[i] + (1 if bit else -1), 0, 3)
        for _, bit in obs:
            self.hist = ((self.hist << 1) | bit) & ((1 << self.h) - 1)


class Perceptron:
    """Jimenez-Lin perceptron. y = bias + sum(hist_i * w_i) + w_slot[slot] + sum(conf_j * w_j).
    hist_i = +-1 (last H outcome bits), slot is one-hot (adds one weight), confidence is a
    thermometer code over CONF_EDGES as +-1. Integer weights, adds only. Trains when the
    prediction was wrong or |y| <= theta. Rows selected by a hash of last_tok."""
    family = "perceptron"; uses_conf = True; decides = True

    def __init__(self, rows_bits=4, hist=12, wbits=8):
        self.rb = rows_bits; self.h = hist; self.hist = [-1] * hist
        self.nc = len(CONF_EDGES)
        self.nw = 1 + hist + GAMMA + self.nc
        self.w = [[0] * self.nw for _ in range(1 << rows_bits)]
        self.wmax = (1 << (wbits - 1)) - 1
        active = 1 + hist + 1 + self.nc
        self.theta = int(1.93 * active + 14)
        self.name = "perceptron"
        self.bits = (1 << rows_bits) * self.nw * wbits + hist
        self.bits_detail = (f"{1 << rows_bits} rows x {self.nw} weights (1 bias + {hist} history + "
                            f"{GAMMA} slot + {self.nc} confidence) x {wbits} bits + {hist}-bit history; "
                            f"theta {self.theta}")

    def _x(self, f):
        """(row, [(weight index, +-1 input)])"""
        slot, conf, tok = f
        x = [(0, 1)] + [(1 + i, h) for i, h in enumerate(self.hist)]
        x.append((1 + self.h + slot, 1))
        base = 1 + self.h + GAMMA
        x += [(base + j, 1 if conf > e else -1) for j, e in enumerate(CONF_EDGES)]
        return (tok ^ (tok >> self.rb) ^ (tok >> 2 * self.rb)) & ((1 << self.rb) - 1), x

    def predict(self, f):
        r, x = self._x(f)
        w = self.w[r]
        return sum(w[i] * v for i, v in x) >= 0

    def update(self, obs):
        for f, bit in obs:
            r, x = self._x(f)
            w = self.w[r]
            y = sum(w[i] * v for i, v in x)
            if (y >= 0) != bool(bit) or abs(y) <= self.theta:
                d = 1 if bit else -1
                for i, v in x:
                    w[i] = sat(w[i] + d * v, -self.wmax - 1, self.wmax)
        for _, bit in obs:
            self.hist = [1 if bit else -1] + self.hist[:-1]


class TageLite:
    """Bimodal base + tagged tables with geometric history lengths. The longest table whose
    tag matches provides the prediction; 2-bit usefulness counters guard allocation."""
    family = "tage"; uses_conf = False; decides = True

    def __init__(self, base_bits=9, idx_bits=7, tag_bits=8, lengths=(4, 8, 16)):
        self.bb = base_bits; self.ib = idx_bits; self.tb = tag_bits; self.L = lengths
        self.base = [2] * (1 << base_bits)
        n = 1 << idx_bits
        self.ctr = [[4] * n for _ in lengths]       # 3-bit, >= 4 means continue
        self.tag = [[-1] * n for _ in lengths]      # -1 = empty (a valid bit in hardware)
        self.u = [[0] * n for _ in lengths]
        self.hist = 0; self.hmax = max(lengths); self.ticks = 0
        self.name = "tage-lite"
        per = 3 + tag_bits + 2
        self.bits = (2 << base_bits) + len(lengths) * n * per + self.hmax
        self.bits_detail = (f"base {1 << base_bits} x 2-bit + {len(lengths)} tables x {n} entries x "
                            f"{per} bits (3 ctr + {tag_bits} tag + 2 useful), history lengths "
                            f"{'/'.join(map(str, lengths))} + {self.hmax}-bit history")

    @staticmethod
    def _fold(h, length, bits):
        h &= (1 << length) - 1
        out = 0
        while h:
            out ^= h & ((1 << bits) - 1)
            h >>= bits
        return out

    def _keys(self, f):
        slot, _, tok = f
        key = tok | (slot << 11)
        ks = []
        for t, length in enumerate(self.L):
            i = (key ^ (key >> self.ib) ^ self._fold(self.hist, length, self.ib) ^ (t * 0x2B)) & ((1 << self.ib) - 1)
            g = (key ^ (key >> 3) ^ self._fold(self.hist, length, self.tb)
                 ^ (self._fold(self.hist, length, self.tb - 1) << 1)) & ((1 << self.tb) - 1)
            ks.append((i, g))
        return tok_hash(tok, slot, self.bb), ks

    def _lookup(self, f):
        bi, ks = self._keys(f)
        hits = [t for t, (i, g) in enumerate(ks) if self.tag[t][i] == g]
        basep = self.base[bi] >= 2
        prov = hits[-1] if hits else -1
        pred = self.ctr[prov][ks[prov][0]] >= 4 if hits else basep
        alt = self.ctr[hits[-2]][ks[hits[-2]][0]] >= 4 if len(hits) > 1 else basep
        return bi, ks, prov, pred, alt

    def predict(self, f): return self._lookup(f)[3]

    def update(self, obs):
        for f, bit in obs:
            bi, ks, prov, pred, alt = self._lookup(f)
            d = 1 if bit else -1
            if prov < 0:
                self.base[bi] = sat(self.base[bi] + d, 0, 3)
            else:
                i = ks[prov][0]
                self.ctr[prov][i] = sat(self.ctr[prov][i] + d, 0, 7)
                if pred != alt:
                    self.u[prov][i] = sat(self.u[prov][i] + (1 if pred == bool(bit) else -1), 0, 3)
            if pred != bool(bit) and prov < len(self.L) - 1:
                for t in range(prov + 1, len(self.L)):
                    i, g = ks[t]
                    if self.u[t][i] == 0:
                        self.tag[t][i] = g; self.ctr[t][i] = 4 if bit else 3
                        break
                else:
                    for t in range(prov + 1, len(self.L)):
                        i = ks[t][0]
                        self.u[t][i] -= 1 if self.u[t][i] else 0
            self.ticks += 1
            if self.ticks % 4096 == 0:              # age the usefulness counters
                self.u = [[v >> 1 for v in row] for row in self.u]
        for _, bit in obs:
            self.hist = ((self.hist << 1) | bit) & ((1 << self.hmax) - 1)


class Oracle:
    """Knows run; submits exactly n = run. Upper bound, not buildable."""
    family = "oracle"; uses_conf = False; decides = False
    name = "oracle"; bits = None; bits_detail = "not buildable (knows the answer)"


def make_predictors():
    ps = [Fixed(n) for n in range(GAMMA + 1)]
    ps += [ConfThreshold(t) for t in CONF_SWEEP]
    ps += [Bimodal(), Gshare(), Perceptron(), TageLite(), Oracle()]
    return ps


# ---------------------------------------------------------------- replay

class Tally:
    def __init__(self): self.rounds = self.tokens = self.n = self.peeks = self.dec = self.right = 0

    def add(self, n, run, g, peek):
        self.rounds += 1; self.tokens += min(n, run) + 1; self.n += n; self.peeks += peek
        self.dec += n + (n < g); self.right += min(n, run) + (n < g and n >= run)

    def result(self, c, lane=0.0):
        # submitted guess: draft pass c + one more verify lane (lane); peeked-but-unsubmitted: c only
        if not self.rounds:
            return None
        return dict(rounds=self.rounds,
                    tokens_per_target_pass=self.tokens / self.rounds,
                    tokens_per_cost=self.tokens / (self.rounds + (c + lane) * self.n),
                    tokens_per_cost_peek_charged=self.tokens / (self.rounds + (c + lane) * self.n + c * self.peeks),
                    avg_n=self.n / self.rounds,
                    decision_accuracy=self.right / self.dec if self.dec else None)


def replay(pred, rounds, half_from, c, shadow=True, lane=0.0):
    whole, half = Tally(), Tally()
    oracle = isinstance(pred, Oracle)
    for k, r in enumerate(rounds):
        slots, run = r["slots"], r["run"]
        g = len(slots)
        if oracle:
            n = run
        else:
            n = 0
            while n < g and pred.predict(slots[n]):     # no feedback inside the round
                n += 1
        peek = int(pred.decides and n < g)
        whole.add(n, run, g, peek)
        if k >= half_from:
            half.add(n, run, g, peek)
        if not oracle:
            m = min(n, run)
            obs = [(slots[i], 1) for i in range(m)]
            if m < g and (m < n or shadow):             # verified reject, or the shadow guess
                obs.append((slots[m], 1 if m < run else 0))
            pred.update(obs)
    return whole.result(c, lane), half.result(c, lane)


# ---------------------------------------------------------------- data notes

def diagnostics(rounds):
    """Accept rates over judged guesses of the gamma = 8 log (what signal is there to find)."""
    def rate(pairs):
        return [dict(key=k, n=n, accept=a / n) for k, (a, n) in pairs if n]
    by_slot, by_conf, by_prev = {}, {}, {}
    edges = [0.0] + CONF_EDGES
    runs = [0] * (GAMMA + 1)
    prev = None
    judged = acc = 0
    for r in rounds:
        if r["slots"]:
            runs[r["run"]] += 1
        for i, (slot, conf, _) in enumerate(r["slots"]):
            if i > r["run"]:
                break
            bit = int(i < r["run"])
            judged += 1; acc += bit
            b = sum(conf > e for e in CONF_EDGES)
            keys = ((by_slot, slot), (by_conf, f"{edges[b]:g}+"))
            if i == 0 and prev is not None:
                keys += ((by_prev, f"prev run {min(prev, 3)}{'+' if prev >= 3 else ''}"),)
            for d, k in keys:
                v = d.setdefault(k, [0, 0]); v[0] += bit; v[1] += 1
        prev = r["run"] if r["slots"] else None
    return dict(judged_guesses=judged, accept_rate=acc / max(judged, 1),
                run_length_histogram=runs,
                by_slot=rate(sorted(by_slot.items())),
                by_conf=rate(sorted(by_conf.items(), key=lambda kv: float(kv[0][:-1]))),
                slot0_by_prev_run=rate(sorted(by_prev.items())))


def measured_from_sibling(log, n, main_rounds, prompt_ids):
    """tokens per target pass of the real run at gamma = n on the same prompts, read from
    guesses_g<n>.csv next to the log (or in logs/). None if missing, partial, or the prompts
    differ (checked: every prompt's first guess must match the main log's)."""
    for d in (Path(log).parent, ROOT / "logs"):
        p = d / f"guesses_g{n}.csv"
        if not p.exists():
            continue
        try:
            rounds, info = load_rounds(p, set(prompt_ids))
        except (SystemExit, OSError):
            continue
        if info["prompts"] != len(prompt_ids):
            continue                                    # partial / different prompt set
        first = {}
        for r in main_rounds:
            first.setdefault(r["prompt"], r.get("first"))
        seen = {}
        for r in rounds:
            seen.setdefault(r["prompt"], r.get("first"))
        if seen != first:
            continue
        return info["prompts"] * info["tokens_per_prompt"] / info["rounds"]
    return None


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", default=str(ROOT / "logs" / "guesses_g8.csv"))
    ap.add_argument("--cost", type=float, default=0.125, help="c_draft: one draft pass in target passes")
    ap.add_argument("--lane", type=float, default=0.0,
                    help="extra verify cost per submitted guess, in target passes (measured 16 Nov: 0.43; "
                         "0 = the old weights-only assumption)")
    ap.add_argument("--json", default=str(ROOT / "build" / "bakeoff.json"))
    ap.add_argument("--no-shadow", action="store_true",
                    help="predictors do not learn the outcome of the guess they declined")
    a = ap.parse_args()
    c = a.cost

    rounds, info = load_rounds(a.log)
    pids = info["prompt_ids"]
    half_prompt = pids[len(pids) // 2]
    half_from = next(k for k, r in enumerate(rounds) if r["prompt"] >= half_prompt)

    print(f"log {info['path']}")
    print(f"  {info['prompts']} prompts, {info['rounds']} rounds "
          f"({info['zero_slot_rounds_added']} unlogged 1-token rounds reconstructed), "
          f"{info['tokens_per_prompt']} tokens per prompt")
    if info["bad_rows"] or info["dropped_prompts"]:
        print(f"  WARNING: {info['bad_rows']} unreadable rows/prompts, incomplete prompts dropped: "
              f"{info['dropped_prompts']} (file may be mid-rewrite)")
    print(f"  c_draft = {c:g}, lane = {a.lane:g} (cost = 1 + n * (c_draft + lane); a peek costs c_draft); second half = prompts {half_prompt}.. ({len(rounds) - half_from} rounds); "
          f"shadow guess {'off' if a.no_shadow else 'on'}")
    print("  Rounds are replayed independently (approximation: an early stop would shift where the")
    print("  next round starts). Predictors learn online, state updated after each completed round.")

    results = []
    for p in make_predictors():
        whole, half = replay(p, rounds, half_from, c, shadow=not a.no_shadow, lane=a.lane)
        results.append(dict(name=p.name, family=p.family, uses_conf=p.uses_conf,
                            bits=p.bits, bits_detail=p.bits_detail, all=whole, second_half=half))
    results.sort(key=lambda r: -r["all"]["tokens_per_cost"])

    def cells(m):
        acc = "    -" if m["decision_accuracy"] is None else f"{100 * m['decision_accuracy']:5.1f}"
        return (f"{m['tokens_per_target_pass']:8.3f} {m['tokens_per_cost']:8.3f} "
                f"{m['tokens_per_cost_peek_charged']:11.3f} {m['avg_n']:6.2f} {acc}")

    hdr = f"{'tok/pass':>8} {'tok/cost':>8} {'tok/cost+pk':>11} {'avg n':>6} {'acc%':>5}"
    print(f"\nSorted by tokens per cost unit, whole log. tok/cost+pk also charges the unsubmitted")
    print("draft pass at the stop slot (see docstring). fixed-0 = plain decoding = 1.000.")
    print(f"{'':<12} {'bits':>6} | {'whole log':^42} | {'second half':^42}")
    print(f"{'predictor':<12} {'':>6} | {hdr} | {hdr}")
    for r in results:
        bits = "-" if r["bits"] is None else str(r["bits"])
        print(f"{r['name']:<12} {bits:>6} | {cells(r['all'])} | {cells(r['second_half'])}")

    print("\nState (bit budget):")
    for r in sorted(results, key=lambda r: r["name"]):
        if r["family"] not in ("fixed", "conf"):
            print(f"  {r['name']:<11} {'-' if r['bits'] is None else r['bits']:>6}  {r['bits_detail']}")
    print("  fixed-n, conf>t   0  no state")

    by_name = {r["name"]: r for r in results}
    validation = []
    print("\nValidation: fixed-n replay vs real spec.py runs at gamma = n (tokens per target pass).")
    print("Differences come from the independence approximation (and, for the 20-prompt column,")
    print("from a different prompt set if this log is not the 20-prompt one).")
    print(f"{'gamma':>5} {'replay':>8} {'meas(20p)':>10} {'diff':>7} {'meas(same prompts)':>19} {'diff':>7}")
    for n in sorted(MEASURED_20):
        rep = by_name[f"fixed-{n}"]["all"]["tokens_per_target_pass"]
        m20 = MEASURED_20[n]
        ms = rep if n == GAMMA else measured_from_sibling(a.log, n, rounds, pids)
        validation.append(dict(gamma=n, replay=rep, measured_20_prompts=m20, measured_same_prompts=ms))
        same = f"{ms:19.3f} {rep - ms:+7.3f}" if ms is not None else f"{'n/a':>19} {'':>7}"
        print(f"{n:>5} {rep:8.3f} {m20:10.2f} {rep - m20:+7.3f} {same}")
    print("  meas(same prompts) is read from guesses_g<n>.csv (next to --log, else logs/) when it is")
    print("  complete for the same prompts; at gamma = 8 the replay IS the real run (diff 0 by construction).")

    d = diagnostics(rounds)
    print(f"\nData notes ({d['judged_guesses']} judged guesses, accept rate {d['accept_rate']:.3f}):")
    print("  run length 0..8 (rounds): " + " ".join(map(str, d["run_length_histogram"])))
    for title, key in (("accept rate by slot", "by_slot"), ("accept rate by confidence", "by_conf"),
                       ("slot-0 accept rate by previous round's run", "slot0_by_prev_run")):
        print(f"  {title}: " + "  ".join(f"{x['key']}: {x['accept']:.2f} (n={x['n']})" for x in d[key]))

    out = dict(log=info, c_draft=c, lane=a.lane, gamma=GAMMA, shadow_guess=not a.no_shadow,
               second_half_from_prompt=half_prompt,
               notes=["rounds replayed independently (approximation)",
                      "tokens = min(n, run) + 1; cost = n * c_draft + 1",
                      "tokens_per_cost_peek_charged adds one c_draft per round where a predictor "
                      "chose to stop before the slot limit",
                      "conf>t thresholds are a sweep over the same data, not a held-out choice"],
               predictors=results, validation=validation, diagnostics=d)
    jp = Path(a.json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {jp}")


if __name__ == "__main__":
    main()
