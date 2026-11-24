"""SpecCore Live: type a prompt, watch speculative decoding run round by round.

    python tools/live_demo.py            then open http://127.0.0.1:8765
    python tools/live_demo.py --selftest check the streamed loop against model/qmodel.py qspec

What runs: the integer models (model/qmodel.py), i.e. exactly the core's arithmetic (bit-exact with the
C reference and the RTL on every golden prompt). Tokens, draft guesses, accept/reject decisions and the
target's choices are real. The RTL itself is not run live (xsim needs ~2 minutes per token).
Core cycles: MEASURED per-pass costs on the core (build/ksweep_accq_bp_core.txt, shared + per-lane fit at
ctx 15); the attention part is scaled linearly with the context length (estimate). Plain decoding (target
only, greedy) runs after each request and must give the same text, which the page shows.
Local only (binds 127.0.0.1). Stop with Ctrl+C."""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "tools")]
import qmodel  # noqa: E402
import cyclemodel as cm  # noqa: E402

CTX_MAX = 128          # trained context length
MHZ = 122.7            # post-route clock (OOC), measured
LOCK = threading.Lock()  # one generation at a time (models hold KV caches)


def load_costs():
    sweep, ctx = cm.read_ksweep(ROOT / "build/ksweep_accq_bp_core.txt")
    cal = cm.calibrate(sweep)
    return ctx, {m: {b: cal[m][b][:2] for b in cm.CAL_BUCKETS} for m in ("draft", "target")}


CAL_CTX, COST = load_costs()


def pass_cost(model, k, ctx):
    """Cycles of one pass with k lanes at context ctx; attention scaled linearly from the calibration ctx."""
    out = {}
    for b, (s, p) in COST[model].items():
        v = s + p * k
        if b == "attn":
            v *= max(ctx, 1) / CAL_CTX
        out[b] = max(0.0, v)
    out["total"] = sum(out.values())
    return out


def top(logits, n=5):
    lg = np.asarray(logits, dtype=np.float64) / 4096.0          # Q12 -> logit units
    pr = np.exp(lg - lg.max()); pr /= pr.sum()
    idx = np.argsort(-lg)[:n]
    return [[int(i), round(float(pr[i]), 4)] for i in idx]


class Pieces:
    def __init__(self):
        from tokenizers import ByteLevelBPETokenizer
        d = ROOT / "model/tokenizer/bpe2048"
        self.tok = ByteLevelBPETokenizer(str(d / "vocab.json"), str(d / "merges.txt"))
        self.cache = {}

    def __call__(self, t):
        t = int(t)
        if t not in self.cache:
            self.cache[t] = self.tok.decode([t])
        return self.cache[t]

    def encode(self, text):
        return self.tok.encode(text).ids


def spec_stream(dm, tm, prompt, n_new, gamma, gate=None):
    """Same algorithm as qmodel.qspec (peek gate), yielding one dict per round with the logits detail."""
    dm.reset(); tm.reset()
    seq = list(prompt)
    end = len(prompt) + n_new
    tm.feed(seq[:-1]); dm.feed(seq[:-1])
    while len(seq) < end:
        g = min(gamma, end - len(seq) - 1)
        behind = len(seq) - 1 - dm.n
        if behind > 0:
            dm.feed(seq[dm.n:len(seq) - 1])
        guesses, gl, d_last, dpass, peeked = [], [], seq[-1], 0, False
        while len(guesses) < g:
            lg = dm.step(d_last); dpass += 1
            top2 = np.partition(lg, -2)[-2:]
            t, mg = int(np.argmax(lg)), int(top2[1] - top2[0])     # exactly qmodel.feed_m
            if gate is not None and guesses and mg <= gate:
                peeked = True
                break
            guesses.append(t); gl.append(dict(top=top(lg, 3), margin=mg)); d_last = t
        g = len(guesses)
        ctx = len(seq)
        base = tm.n
        tl = [tm.step(t) for t in [seq[-1]] + guesses]
        want = [int(np.argmax(x)) for x in tl]
        n_ok = 0
        while n_ok < g and guesses[n_ok] == want[n_ok]:
            n_ok += 1
        seq += guesses[:n_ok] + [want[n_ok]]
        tm.truncate(base + 1 + n_ok)
        dm.truncate(min(dm.n, len(seq) - 1))
        yield dict(guesses=guesses, draft=gl, target=[top(x) for x in tl[:n_ok + 1]], n_ok=n_ok, fix=want[n_ok],
                   behind=max(behind, 0), dpass=dpass, peeked=peeked, ctx=ctx, done=len(seq) - len(prompt))
    return


class Handler(BaseHTTPRequestHandler):
    pieces = None

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self.send(200, (ROOT / "artifacts/demo/live.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/generate":
            return self.generate(parse_qs(u.query))
        self.send(404, b"not found", "text/plain")

    def event(self, kind, data):
        self.wfile.write(f"event: {kind}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())
        self.wfile.flush()

    def generate(self, q):
        P = self.pieces
        text = q.get("prompt", ["def add(a, b):"])[0]
        mode = q.get("mode", ["g2"])[0]
        gamma, gate = {"g1": (1, None), "g2": (2, None), "g3": (3, None), "g4": (4, None), "gate": (4, 4096)}.get(mode, (2, None))
        prompt = P.encode(text)[-64:] or P.encode("\n")
        n_new = max(1, min(int(q.get("n", ["96"])[0]), CTX_MAX - len(prompt)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not LOCK.acquire(timeout=60):
            return self.event("fail", dict(msg="Another generation is running; try again in a moment."))
        try:
            self.event("start", dict(prompt=[[t, P(t)] for t in prompt], n_new=n_new, gamma=gamma, gate=gate,
                                     cal_ctx=CAL_CTX, mhz=MHZ))
            dm, tm = qmodel.QModel("draft"), qmodel.QModel("target")
            out, cyc = [], 0.0
            for r in spec_stream(dm, tm, prompt, n_new, gamma, gate):
                dr = 0.0
                if r["dpass"] == 0 and r["behind"]:
                    dr += pass_cost("draft", r["behind"], r["ctx"])["total"]
                for i in range(r["dpass"]):
                    dr += pass_cost("draft", 1 + (r["behind"] if i == 0 else 0), r["ctx"] + i)["total"]
                K = len(r["guesses"]) + 1
                vc = pass_cost("target", K, r["ctx"])
                cyc += dr + vc["total"]
                new = r["guesses"][:r["n_ok"]] + [r["fix"]]
                out += new
                plain_so_far = sum(pass_cost("target", 1, len(prompt) + i)["total"] for i in range(min(len(out), n_new)))
                r.update(plain_cycles=round(plain_so_far), draft_cycles=round(dr), verify=dict((k, round(v)) for k, v in vc.items()), K=K,
                         cycles=round(cyc), pieces={str(t): P(t) for t in set(r["guesses"] + [r["fix"]] +
                         [a for x in r["target"] + [d["top"] for d in r["draft"]] for a, _ in x])})
                self.event("round", r)
            out = out[:n_new]
            plain = qmodel.qgreedy(tm, prompt, n_new)              # target alone, greedy
            plain_cyc = sum(pass_cost("target", 1, len(prompt) + i)["total"] for i in range(n_new))
            self.event("done", dict(identical=plain == out, plain_cycles=round(plain_cyc), spec_cycles=round(cyc)))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            LOCK.release()


def selftest():
    P = Pieces()
    sp, _, _ = qmodel.float_models()
    ok = True
    for seed, plen, n in ((1, 16, 8), (7, 8, 16), (21, 16, 40)):
        p = sp.prompts(1, plen, seed=seed)[0]
        for gamma, gate in ((2, None), (4, 4096)):
            ref, _ = qmodel.qspec(qmodel.QModel("draft"), qmodel.QModel("target"), p, n, gamma, gate)
            got = []
            for r in spec_stream(qmodel.QModel("draft"), qmodel.QModel("target"), p, n, gamma, gate):
                got += r["guesses"][:r["n_ok"]] + [r["fix"]]
            same = got[:n] == ref
            ok &= same
            print(f"seed {seed} gamma {gamma} gate {gate}: {'same as qspec' if same else 'DIFFERENT'}")
    print("selftest", "PASS" if ok else "FAIL", "|", repr("".join(P(t) for t in P.encode("def f(x):"))))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    Handler.pieces = Pieces()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"SpecCore Live on http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
    if not a.no_open:
        import webbrowser
        threading.Timer(1.0, webbrowser.open, (f"http://127.0.0.1:{a.port}",)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
