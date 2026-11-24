"""SpecCore model: small Llama-style decoder for Python code.

Explained block by block in artifacts/primer03_model.html and docs/explain/model_worksheet.md.
Check with:  python model/check.py        (steps 1-8)

Sizes come from speccore.toml ([draft], [target]). Shape names used everywhere:
  B = batch (sequences at once), T = tokens per sequence, C = dim (numbers per token)
  H = heads, D = C // H (numbers per head)

Checkpoint contract (keep these names if you rewrite this file, so trained weights still load):
  tok_emb.weight                      (vocab, C)   also the output layer (tied)
  blocks.{i}.norm1.weight             (C,)
  blocks.{i}.attn.qkv.weight          (3C, C)      rows: q, then k, then v
  blocks.{i}.attn.proj.weight         (C, C)
  blocks.{i}.norm2.weight             (C,)
  blocks.{i}.mlp.fc1.weight           (4C, C)
  blocks.{i}.mlp.fc2.weight           (C, 4C)
  norm_f.weight                       (C,)
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sc  # noqa: E402


@dataclass
class Config:
    vocab: int = 2048
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ctx: int = 128        # max tokens the model sees
    mlp_mult: int = 4     # MLP hidden size = mlp_mult * dim


_cfg = sc.load()
TARGET = Config(**sc.model_cfg(_cfg, "target"))
DRAFT = Config(**sc.model_cfg(_cfg, "draft"))


# ---------------- Step 2: RMSNorm ----------------
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2 over last dim) + eps) * weight.  Shape in = shape out."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------- Step 4: RoPE ----------------
def rope(x, start: int = 0):
    """Rotate pairs of numbers in x by an angle that depends on position.

    x: (B, H, T, D), D even. Token t gets position start + t.
    Pairs are neighbours: (x0, x1), (x2, x3), ...  Pair i turns by pos * 10000^(-2i/D).
    """
    B, H, T, D = x.shape
    i = torch.arange(D // 2, device=x.device, dtype=torch.float32)
    freq = 10000.0 ** (-2 * i / D)                                   # (D/2,)
    pos = torch.arange(start, start + T, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]                               # (T, D/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()              # (B, H, T, D/2) each
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2).type_as(x)      # re-interleave pairs


# ---------------- Step 3 (+4): causal self-attention ----------------
class Attention(nn.Module):
    """x (B, T, C) -> (B, T, C). Each token mixes in information from itself and EARLIER tokens only."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.h, self.d = cfg.n_heads, cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        mask = torch.tril(torch.ones(cfg.ctx, cfg.ctx, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)        # not saved in checkpoints

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                       # each (B, T, C)
        q, k, v = (t.view(B, T, self.h, self.d).transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        q, k = rope(q), rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)          # (B, H, T, T) relevance scores
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))      # no looking at later tokens
        att = F.softmax(att.float(), dim=-1).type_as(v)              # each row sums to 1
        y = att @ v                                                  # weighted sum of values
        y = y.transpose(1, 2).reshape(B, T, C)                       # glue heads back together
        return self.proj(y)


# ---------------- Step 5: MLP ----------------
class MLP(nn.Module):
    """x (B, T, C) -> (B, T, C): Linear C->4C, ReLU, Linear 4C->C. No biases."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.mlp_mult * cfg.dim, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_mult * cfg.dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------- Step 6: Block ----------------
class Block(nn.Module):
    """x = x + attn(norm1(x));  x = x + mlp(norm2(x))."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.dim), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.dim), MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------- Step 7: the whole model ----------------
class Model(nn.Module):
    """Token IDs (B, T) -> logits (B, T, vocab), and the loss if targets are given."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.dim)
        # Small random start (std 0.02). Layers that write into the residual path start smaller,
        # so the sum over many layers doesn't blow up at the start of training (GPT-2 trick).
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * cfg.n_layers) if name.endswith(("proj.weight", "fc2.weight")) else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        """Returns (logits, loss). loss is None when targets is None."""
        x = self.tok_emb(idx)                                        # (B, T, C): look up one row per token
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = x @ self.tok_emb.weight.T                           # (B, T, vocab): tied output layer
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
