"""Download a permissive-licence subset of codeparrot-clean (Python files from GitHub).

Output: data/raw/train.txt and data/raw/val.txt, files separated by EOT.

Usage: python model/data.py --mb 300
"""
import argparse
import os
import random

from datasets import load_dataset

EOT = "<|endoftext|>"
# Per-file licence column on the dataset card. Keep permissive licences only.
KEEP = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=300, help="target size of train text in MB")
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--max_file_kb", type=int, default=64, help="skip huge files")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    rng = random.Random(0)
    target = args.mb * 1_000_000
    n_train = n_val = files = skipped = 0
    with open(f"{args.out}/train.txt", "w", encoding="utf-8") as ft, \
         open(f"{args.out}/val.txt", "w", encoding="utf-8") as fv:
        for row in ds:
            text = row["content"]
            if row["license"] not in KEEP or len(text) > args.max_file_kb * 1024 or EOT in text:
                skipped += 1
                continue
            chunk = text.rstrip() + "\n" + EOT + "\n"
            if rng.random() < args.val_frac:
                fv.write(chunk); n_val += len(chunk)
            else:
                ft.write(chunk); n_train += len(chunk)
            files += 1
            if files % 20000 == 0:
                print(f"{files} files, train {n_train/1e6:.0f} MB, val {n_val/1e6:.1f} MB, skipped {skipped}", flush=True)
            if n_train >= target:
                break
    print(f"done: {files} files, train {n_train/1e6:.1f} MB, val {n_val/1e6:.2f} MB, skipped {skipped}")


if __name__ == "__main__":
    main()
