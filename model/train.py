"""Train the draft or target model on tokenized code.

Usage:
  python model/train.py --model draft  --hours 1.5
  python model/train.py --model target --hours 2
Reads data/tok<vocab>/{train,val}.bin. Saves checkpoints/<model>.pt (best val loss) and logs to checkpoints/<model>.log.

Draft variants (D15): --name NAME writes checkpoints/NAME.pt / .log instead (never overwrites draft.pt),
--dim/--layers/--heads change the shape (head dim must stay 32 for the C code), --distil W trains
on (1-W) x cross-entropy + W x KL(target || draft) with the trained target as teacher.
  python model/train.py --model draft --name d64x2 --dim 64 --heads 2 --tokens 228e6
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import model as M  # noqa: E402


def batches(data, B, T, dev, gen):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix.tolist()])
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


@torch.no_grad()
def evaluate(m, data, B, T, dev, n=40):
    m.eval()
    gen = torch.Generator().manual_seed(123)  # same val batches every time
    tot = 0.0
    for _ in range(n):
        x, y = batches(data, B, T, dev, gen)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            _, loss = m(x, y)
        tot += loss.item()
    m.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["draft", "target"], required=True)
    ap.add_argument("--vocab", type=int, default=None, help="default: speccore.toml")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens (default tokens_per_param x params, speccore.toml)")
    ap.add_argument("--batch", type=int, default=None, help="default: speccore.toml")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hours", type=float, default=None, help="hard time cap (default: speccore.toml)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--name", default=None, help="checkpoint/log name (default: the model name)")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--distil", type=float, default=0.0, help="weight of KL(target || model), 0 = off")
    ap.add_argument("--teacher", default="checkpoints/target.pt", help="--distil: teacher checkpoint (target shape)")
    ap.add_argument("--hard", action="store_true", help="--distil term = cross-entropy on the teacher's argmax token (greedy acceptance) instead of KL")
    ap.add_argument("--init", default=None, help="warm-start: load weights from this checkpoint (same shape)")
    args = ap.parse_args()
    name = args.name or args.model
    if name == "draft" and (args.dim or args.layers or args.heads or args.distil):
        sys.exit("a changed draft needs --name (draft.pt is the reference checkpoint)")
    tc = M._cfg["train"]
    args.vocab = args.vocab or M._cfg["tokenizer"]["vocab"]
    args.batch = args.batch or tc["batch"]
    args.hours = args.hours or tc[f"{args.model}_hours"]

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = M.Config(**(M.DRAFT if args.model == "draft" else M.TARGET).__dict__)
    cfg.vocab = args.vocab
    cfg.dim = args.dim or cfg.dim
    cfg.n_layers = args.layers or cfg.n_layers
    cfg.n_heads = args.heads or cfg.n_heads
    assert cfg.dim // cfg.n_heads == 32, "head dim must be 32 (QR_D in ref/qref.h)"
    T, B = cfg.ctx, args.batch

    train = np.memmap(f"data/tok{args.vocab}/train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(f"data/tok{args.vocab}/val.bin", dtype=np.uint16, mode="r")

    m = M.Model(cfg).to(dev)
    if args.init:
        m.load_state_dict(torch.load(args.init, map_location=dev)["model"])
    n_params = sum(p.numel() for p in m.parameters())
    tokens = args.tokens or tc["tokens_per_param"] * n_params
    steps = int(tokens // (B * T))
    lr = args.lr or (2e-3 if args.model == "draft" else 1e-3)
    warm = min(500, steps // 20)

    decay = [p for p in m.parameters() if p.dim() >= 2]
    no_decay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))

    def lr_at(s):
        if s < warm:
            return lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    os.makedirs("checkpoints", exist_ok=True)
    log = open(f"checkpoints/{name}.log", "a")
    teacher = None
    if args.distil:
        teacher = M.Model(M.TARGET).to(dev)
        teacher.load_state_dict(torch.load(args.teacher, map_location=dev)["model"])
        teacher.eval()

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"{name} ({cfg.dim}x{cfg.n_layers}, {cfg.n_heads} heads, distil {args.distil}{' hard' if args.hard else ''}{' init ' + args.init if args.init else ''}): {n_params/1e6:.3f}M params, {len(train)/1e6:.0f}M train tokens available, "
        f"{steps} steps x {B*T} tokens = {steps*B*T/1e6:.0f}M tokens, lr {lr}, device {dev}")

    gen = torch.Generator().manual_seed(0)
    best = float("inf")
    t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = batches(train, B, T, dev, gen)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits, loss = m(x, y)
            if teacher is not None:
                with torch.no_grad():
                    tl, _ = teacher(x)
                if args.hard:
                    kl = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tl.argmax(-1).view(-1))
                else:
                    kl = F.kl_div(F.log_softmax(logits.float(), -1), F.log_softmax(tl.float(), -1),
                                  log_target=True, reduction="batchmean") / T
                loss = (1 - args.distil) * loss + args.distil * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        if s % 100 == 0:
            el = time.time() - t0
            say(f"step {s}/{steps} loss {loss.item():.3f} lr {lr_at(s):.2e} {el/60:.1f} min")
        last = s == steps - 1 or (time.time() - t0) > args.hours * 3600
        if s % args.eval_every == 0 or last:
            vl = evaluate(m, val, B, T, dev)
            say(f"  val loss {vl:.3f}")
            if vl < best:
                best = vl
                torch.save({"cfg": cfg.__dict__, "model": m.state_dict(), "step": s, "val_loss": vl},
                           f"checkpoints/{name}.pt")
        if last:
            break
    say(f"done: best val loss {best:.3f}, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
