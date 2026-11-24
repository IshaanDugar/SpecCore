"""Export a trained checkpoint to the integer form the core runs (D4-D6).

    python model/export.py                  # both models -> checkpoints/export/<name>.npz
    python model/export.py --models target

What changes from the float model (each is checked by model/qmodel.py --agree):
  1. Every weight matrix becomes int8 with one scale per row ("per-row scale").
     W8[i, j] = round(W[i, j] / s_i), s_i = max_j |W[i, j]| / 127.
  2. Each RMSNorm gain g is folded into the matrix that follows it (W' = W * g per column),
     so the core never multiplies by g. That un-ties the output layer: the output matrix is
     tok_emb * norm_f.weight, the embedding table stays tok_emb (two int8 tables).
  3. Scales are integer pairs (m, e): value = m * 2^-e, m in [2^29, 2^30) (D5, mult + shift).
  4. Tables the core needs: RoPE cos/sin in Q14 and a 256-entry 2^(-i/256) table in Q15 for
     the integer softmax, plus the constant log2(e)/sqrt(D) as a scale.

Arrays in the .npz (each matrix X has X.w int8 (rows, cols), X.m and X.e (rows,) row scales):
  emb                       embedding table (V, C)
  L{i}.qkv  (3C, C)         norm1 gain folded in; rows q, then k, then v
  L{i}.proj (C, C)
  L{i}.fc1  (4C, C)         norm2 gain folded in
  L{i}.fc2  (C, 4C)
  out       (V, C)          norm_f gain folded in
  rope_cos/rope_sin (ctx, D/2) int32 Q14, exp2_lut (256,) int32 Q15
  attn_m/attn_e             log2(e)/sqrt(D) as a scale
  cfg                       [vocab, dim, layers, heads, ctx, mlp_mult]
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from model import DRAFT, TARGET  # noqa: E402

OUT = Path(os.environ["SC_EXPORT"]) if os.environ.get("SC_EXPORT") else ROOT / "checkpoints" / "export"
M_LO, M_HI = 1 << 29, 1 << 30


def float_scale(x):
    """Positive float -> (m, e) with m * 2^-e <= x, m in [2^29, 2^30). x = 0 -> (0, 0)."""
    if x <= 0:
        return 0, 0
    e = 29 - math.floor(math.log2(x))
    m = math.floor(x * 2.0 ** e)
    while m >= M_HI:
        e -= 1; m = math.floor(x * 2.0 ** e)
    while m < M_LO:
        e += 1; m = math.floor(x * 2.0 ** e)
    return m, e


def quant_rows(w):
    """float (R, C) -> int8 (R, C), m (R,) int64, e (R,) int64. Per-row scale, symmetric, +-127."""
    w = w.astype(np.float64)
    amax = np.abs(w).max(axis=1)
    m = np.zeros(len(w), np.int64); e = np.zeros(len(w), np.int64)
    q = np.zeros(w.shape, np.int8)
    for i, a in enumerate(amax):
        if a == 0:
            continue
        m[i], e[i] = float_scale(a / 127.0)
        s = m[i] * 2.0 ** -int(e[i])                        # use the stored scale, not the ideal one
        q[i] = np.clip(np.rint(w[i] / s), -127, 127).astype(np.int8)
    return q, m, e


def tables(cfg):
    D = cfg.dim // cfg.n_heads
    i = np.arange(D // 2, dtype=np.float64)
    freq = 10000.0 ** (-2 * i / D)                          # same as model.rope
    ang = np.arange(cfg.ctx, dtype=np.float64)[:, None] * freq[None, :]
    cos = np.rint(np.cos(ang) * (1 << 14)).astype(np.int32)
    sin = np.rint(np.sin(ang) * (1 << 14)).astype(np.int32)
    lut = np.rint(32768.0 * 2.0 ** (-np.arange(256) / 256.0)).astype(np.int32)
    am, ae = float_scale(math.log2(math.e) / math.sqrt(D))
    return cos, sin, lut, am, ae


def export(name, cfg, ckpt=None):
    ck = torch.load(ROOT / "checkpoints" / f"{ckpt or name}.pt", map_location="cpu")
    sd = {k: v.double().numpy() for k, v in ck["model"].items()}
    arr = {}

    def put(key, w):
        q, m, e = quant_rows(w)
        arr[key + ".w"], arr[key + ".m"], arr[key + ".e"] = q, m, e

    put("emb", sd["tok_emb.weight"])
    for L in range(cfg.n_layers):
        p = f"blocks.{L}."
        put(f"L{L}.qkv", sd[p + "attn.qkv.weight"] * sd[p + "norm1.weight"][None, :])
        put(f"L{L}.proj", sd[p + "attn.proj.weight"])
        put(f"L{L}.fc1", sd[p + "mlp.fc1.weight"] * sd[p + "norm2.weight"][None, :])
        put(f"L{L}.fc2", sd[p + "mlp.fc2.weight"])
    put("out", sd["tok_emb.weight"] * sd["norm_f.weight"][None, :])
    cos, sin, lut, am, ae = tables(cfg)
    arr.update(rope_cos=cos, rope_sin=sin, exp2_lut=lut, attn_m=np.int64(am), attn_e=np.int64(ae),
               cfg=np.array([cfg.vocab, cfg.dim, cfg.n_layers, cfg.n_heads, cfg.ctx, cfg.mlp_mult]))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", **arr)
    n8 = sum(v.size for k, v in arr.items() if v.dtype == np.int8)
    ns = sum(v.size for k, v in arr.items() if k.endswith(".m"))
    print(f"{name}: {n8:,} int8 weights + {ns:,} row scales -> {OUT / (name + '.npz')}")
    return arr


MAGIC = 0x31514353                                          # "SCQ1"
HDR_WORDS = 16                                              # fixed header, then per-layer offsets


def model_blob(z):
    """One model as bytes (little-endian, every part 4-byte aligned). Layout (u32 words):
      0 magic  1 vocab  2 dim  3 layers  4 heads  5 ctx  6 mlp_mult  7 attn_m  8 attn_e
      9 off emb  10 off out  11 off rope_cos  12 off rope_sin  13 off exp2_lut  14 total bytes  15 0
      16 + 4*L + {0,1,2,3}: off qkv, proj, fc1, fc2 of layer L
    Matrix at off: u32 rows, u32 cols, u32 m[rows], i32 e[rows], i8 w[rows*cols] (row-major,
    so a row is a run of 32-bit words = the VDOTK weight stream).
    Tables: rope cos, sin as i16 (ctx, D/2); exp2 table as u16 (256)."""
    vocab, C, L, H, ctx, mlp = (int(t) for t in z["cfg"])
    parts, off = [], [HDR_WORDS * 4 + 16 * L]

    def add(b):
        b = bytes(b) + b"\0" * (-len(b) % 4)
        parts.append(b); o = off[0]; off[0] += len(b); return o

    def mat(k):
        w, m, e = z[k + ".w"], z[k + ".m"], z[k + ".e"]
        return add(np.array(w.shape, "<u4").tobytes() + m.astype("<u4").tobytes()
                   + e.astype("<i4").tobytes() + w.astype(np.int8).tobytes())

    o_emb, o_out = mat("emb"), mat("out")
    o_cos = add(z["rope_cos"].astype("<i2").tobytes())
    o_sin = add(z["rope_sin"].astype("<i2").tobytes())
    o_lut = add(z["exp2_lut"].astype("<u2").tobytes())
    lay = []
    for i in range(L):
        lay += [mat(f"L{i}.qkv"), mat(f"L{i}.proj"), mat(f"L{i}.fc1"), mat(f"L{i}.fc2")]
    hdr = [MAGIC, vocab, C, L, H, ctx, mlp, int(z["attn_m"]), int(z["attn_e"]),
           o_emb, o_out, o_cos, o_sin, o_lut, off[0], 0] + lay
    return np.array(hdr, "<u4").tobytes() + b"".join(parts)


def write_image():
    """checkpoints/export/model.img = [u32 off target, u32 off draft, 0, 0] + target blob + draft blob.
    The core sees it at DDR_BASE (0x2000_0000); the PC build reads the file."""
    t = model_blob(np.load(OUT / "target.npz"))
    d = model_blob(np.load(OUT / "draft.npz"))
    img = np.array([16, 16 + len(t), 0, 0], "<u4").tobytes() + t + d
    (OUT / "model.img").write_bytes(img)
    print(f"model.img: {len(img):,} bytes (target {len(t):,}, draft {len(d):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["draft", "target"])
    ap.add_argument("--no-img", action="store_true", help="skip writing model.img")
    ap.add_argument("--ckpt", default=None, help="checkpoint name to export (one model only); output dir via env SC_EXPORT")
    a = ap.parse_args()
    for n in a.models:
        export(n, {"draft": DRAFT, "target": TARGET}[n], a.ckpt)
    if not a.no_img:
        write_image()


if __name__ == "__main__":
    main()
