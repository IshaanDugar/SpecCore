"""Checks for model.py, one per worksheet step.

Usage:  python model/check.py 3     # runs checks for steps 1..3
"""
import math
import sys

import torch

sys.path.insert(0, "model")
import model as M  # noqa: E402

torch.manual_seed(0)
TINY = M.Config(vocab=2048, dim=64, n_layers=2, n_heads=4, ctx=32)


def step1():
    x = torch.randn(2, 5, 8)
    assert x.shape == (2, 5, 8)
    print("  step 1 is reading only: tensors and shapes. Nothing to check.")


def step2():
    n = M.RMSNorm(2)
    y = n(torch.tensor([[3.0, 4.0]]))
    want = torch.tensor([[0.8485, 1.1314]])
    assert torch.allclose(y, want, atol=1e-3), f"RMSNorm([3,4]) gave {y.tolist()}, want {want.tolist()}"
    assert any(p.requires_grad for p in n.parameters()), "RMSNorm needs a learnable weight (nn.Parameter)"
    x = torch.randn(2, 7, 64)
    assert M.RMSNorm(64)(x).shape == x.shape


def step3():
    a = M.Attention(TINY)
    x = torch.randn(1, 10, TINY.dim)
    y = a(x)
    assert y.shape == x.shape, f"shape {tuple(y.shape)}, want {tuple(x.shape)}"
    for mod in a.modules():
        if isinstance(mod, torch.nn.Linear):
            assert mod.bias is None, "no biases: use bias=False"
    x2 = x.clone()
    x2[:, -1] += 1.0                      # change only the LAST token
    y2 = a(x2)
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5), \
        "earlier tokens changed when only the last token changed: the causal mask is missing or wrong"
    assert not torch.allclose(y[:, -1], y2[:, -1]), "last token output should change"


def step4():
    D = 32
    q, k = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)
    rq = M.rope(q, start=5)
    assert rq.shape == q.shape
    assert torch.allclose(rq.norm(), q.norm(), atol=1e-4), "rope must not change vector length"
    assert torch.allclose(M.rope(q, start=0), q, atol=1e-6), "position 0 must not rotate"
    d1 = (M.rope(q, 7) * M.rope(k, 3)).sum()
    d2 = (M.rope(q, 17) * M.rope(k, 13)).sum()
    assert torch.allclose(d1, d2, atol=1e-4), "q.k must depend only on the position DIFFERENCE (7-3 = 17-13)"
    step3()  # attention must still pass with rope inside
    print("  (make sure Attention now applies rope to q and k)")


def step5():
    m = M.MLP(TINY)
    x = torch.randn(2, 5, TINY.dim)
    assert m(x).shape == x.shape
    n = sum(p.numel() for p in m.parameters())
    assert n == 8 * TINY.dim ** 2, f"MLP params {n}, want 8*dim^2 = {8 * TINY.dim ** 2} (two C x 4C matrices, no bias)"


def step6():
    b = M.Block(TINY)
    x = torch.randn(2, 5, TINY.dim)
    assert b(x).shape == x.shape


def step7():
    m = M.Model(TINY)
    idx = torch.randint(0, TINY.vocab, (2, 16))
    logits, loss = m(idx)
    assert logits.shape == (2, 16, TINY.vocab) and loss is None
    _, loss = m(idx, idx)
    want = math.log(TINY.vocab)
    assert abs(loss.item() - want) < 1.0, \
        f"loss at start {loss.item():.2f}, want about ln(vocab) = {want:.2f}. Hint: init weights small (std 0.02)"
    for name, cfg, lo, hi in [("target", M.TARGET, 5.0e6, 5.5e6), ("draft", M.DRAFT, 0.6e6, 0.7e6)]:
        n = sum(p.numel() for p in M.Model(cfg).parameters())
        assert lo < n < hi, f"{name} has {n/1e6:.2f}M params, want {lo/1e6}-{hi/1e6}M. Is the output layer tied to the embedding?"
        print(f"  {name}: {n/1e6:.3f}M params")


def step8():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = M.Model(TINY).to(dev)
    idx = torch.randint(0, TINY.vocab, (4, TINY.ctx + 1), device=dev)
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for i in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"  step {i:3d}  loss {loss.item():.3f}")
    print(f"  final loss {loss.item():.3f} on {dev}")
    assert loss.item() < 0.5, "could not memorise one batch: something in the model or the loss is off"


STEPS = [step1, step2, step3, step4, step5, step6, step7, step8]

if __name__ == "__main__":
    upto = int(sys.argv[1]) if len(sys.argv) > 1 else len(STEPS)
    for i, fn in enumerate(STEPS[:upto], 1):
        try:
            fn()
            print(f"step {i}: PASS")
        except NotImplementedError as e:
            print(f"step {i}: not written yet ({e})")
            break
        except AssertionError as e:
            print(f"step {i}: FAIL - {e}")
            break
