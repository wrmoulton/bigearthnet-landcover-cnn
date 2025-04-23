#!/usr/bin/env python3
"""
filter_and_sample.py

1. Uses metadata 'country' column to pick Portugal patches.
2. Copies Portugal .pt files to out_portugal/.
3. Samples N global (non-Portugal) patches to out_global/.
4. --delete removes all other .pt files in src-dir.
"""

import argparse, random, shutil
from pathlib import Path

import pandas as pd

def main(args):
    # Load metadata
    md = pd.read_parquet(args.metadata, engine="pyarrow")
    # Filter Portugal patch IDs
    portugal_md = md[md['country'] == "Portugal"]
    portugal_ids = set(portugal_md['patch_id'].astype(str))
    print(f"✔️  Portugal patches: {len(portugal_ids)}")

    # List all .pt files in src-dir
    src = Path(args.src_dir)
    all_files = list(src.rglob("*.pt"))
    print(f"🔎 Found {len(all_files)} total .pt files")

    # Partition into Portugal vs others
    portugal_files = [f for f in all_files if f.stem in portugal_ids]
    other_files    = [f for f in all_files if f.stem not in portugal_ids]
    print(f" Portugal files:     {len(portugal_files)}")
    print(f" Non‑Portugal files: {len(other_files)}")

    # Prepare output dirs
    out_pt = Path(args.out_portugal); out_pt.mkdir(parents=True, exist_ok=True)
    out_gl = Path(args.out_global);   out_gl.mkdir(parents=True, exist_ok=True)

    # Copy Portugal files
    for f in portugal_files:
        shutil.copy2(f, out_pt / f.name)

    # Sample N global files
    n = args.n_global
    if n > len(other_files):
        raise ValueError(f"Requested {n} global samples but only {len(other_files)} available")
    sampled = random.sample(other_files, n)
    for f in sampled:
        shutil.copy2(f, out_gl / f.name)

    print(f"  Copied {len(portugal_files)} Portugal files to {out_pt}")
    print(f"  Copied {n} global samples to {out_gl}")

    # Delete the rest, if requested
    if args.delete:
        to_delete = set(all_files) - set(portugal_files) - set(sampled)
        print(f"  Deleting {len(to_delete)} files from {src}")
        for f in to_delete:
            f.unlink()
        print("  Deletion complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir",      required=True,
                   help="Root folder of all .pt files")
    p.add_argument("--metadata",     required=True,
                   help="Path to metadata.parquet")
    p.add_argument("--out-portugal", required=True,
                   help="Where to copy Portugal .pt files")
    p.add_argument("--out-global",   required=True,
                   help="Where to copy sampled global .pt files")
    p.add_argument("--n-global",     type=int, required=True,
                   help="Number of global (non-Portugal) patches to sample")
    p.add_argument("--delete",       action="store_true",
                   help="Delete all other .pt files in src-dir after copying")
    args = p.parse_args()

    random.seed(42)
    main(args)
