"""Tokenize data/raw/{train,val}.txt with a trained BPE tokenizer into uint16 token files.

Usage: python model/prepare.py --vocab 2048
Output: data/tok<vocab>/train.bin, val.bin (numpy uint16)
"""
import argparse
import os

import numpy as np
from tokenizers import ByteLevelBPETokenizer

EOT = "<|endoftext|>"


def encode_file(tok, src, dst, eot_id, chunk_files=2000):
    out = open(dst, "wb")
    n = 0
    buf = []

    def flush():
        nonlocal n
        ids = []
        for enc in tok.encode_batch(buf):
            ids.extend(enc.ids)
            ids.append(eot_id)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(out)
        n += len(arr)
        buf.clear()

    with open(src, encoding="utf-8") as f:
        doc = []
        for line in f:
            if line.strip() == EOT:
                buf.append("".join(doc))
                doc = []
                if len(buf) >= chunk_files:
                    flush()
                    print(f"  {dst}: {n/1e6:.1f}M tokens", flush=True)
            else:
                doc.append(line)
    if buf:
        flush()
    out.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048)
    args = ap.parse_args()
    d = f"model/tokenizer/bpe{args.vocab}"
    tok = ByteLevelBPETokenizer(f"{d}/vocab.json", f"{d}/merges.txt")
    tok.add_special_tokens([EOT])
    eot_id = tok.token_to_id(EOT)
    assert eot_id is not None and args.vocab <= 65535
    out = f"data/tok{args.vocab}"
    os.makedirs(out, exist_ok=True)
    for split in ["val", "train"]:
        n = encode_file(tok, f"data/raw/{split}.txt", f"{out}/{split}.bin", eot_id)
        print(f"{split}: {n:,} tokens (eot id {eot_id})")


if __name__ == "__main__":
    main()
