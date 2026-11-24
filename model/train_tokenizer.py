"""Train byte-level BPE tokenizers at several vocab sizes and report the numbers for decision D2.

Numbers reported per vocab size (measured on data/raw/val.txt):
  - characters per token (higher = each token covers more code)
  - output-layer bytes per pass (vocab x dim, int8) for draft (dim 128) and target (dim 256)

Usage: python model/train_tokenizer.py --sample_mb 50
"""
import argparse
import json
import os

from tokenizers import ByteLevelBPETokenizer

EOT = "<|endoftext|>"


def read_mb(path, mb):
    with open(path, encoding="utf-8") as f:
        return f.read(mb * 1_000_000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096")
    ap.add_argument("--sample_mb", type=int, default=50)
    ap.add_argument("--out", default="model/tokenizer")
    args = ap.parse_args()

    sample = read_mb("data/raw/train.txt", args.sample_mb)
    val = read_mb("data/raw/val.txt", 2)
    sample_path = "data/raw/_tok_sample.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample)

    rows = []
    for v in [int(s) for s in args.sizes.split(",")]:
        tok = ByteLevelBPETokenizer()
        tok.train([sample_path], vocab_size=v, min_frequency=2, special_tokens=[EOT], show_progress=False)
        d = f"{args.out}/bpe{v}"
        os.makedirs(d, exist_ok=True)
        tok.save_model(d)
        n_tok = sum(len(tok.encode(part).ids) for part in val.split(EOT))
        cpt = len(val) / n_tok
        rows.append({"vocab": v, "chars_per_token": round(cpt, 3),
                     "draft_out_layer_KB": v * 128 // 1024, "target_out_layer_KB": v * 256 // 1024})
        print(rows[-1], flush=True)
    with open(f"{args.out}/report.json", "w") as f:
        json.dump(rows, f, indent=2)
    os.remove(sample_path)


if __name__ == "__main__":
    main()
