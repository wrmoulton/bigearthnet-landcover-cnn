#!/usr/bin/env python3
"""
Estimate land‑cover change in Portugal between two years across multiple categories
using your fine‑tuned CNN, and produce a bar chart of the deltas.

Usage:
  python landcover_change.py \
    --metadata    data/metadata.parquet \
    --portugal-dir data/portugal_subset \
    --checkpoint  best_ft.pt \
    --year-a      2017 \
    --year-b      2018 \
    --batch-size  64
"""
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from finetune_portugal import build_model  # your model factory
import matplotlib.pyplot as plt

# ─── Dataset ────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, data_dir, patch_ids, md, class_to_idx):
        self.data_dir     = Path(data_dir)
        self.ids          = patch_ids
        self.md           = md.set_index("patch_id")
        self.class_to_idx = class_to_idx
        self.num_classes  = len(class_to_idx)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        x   = torch.load(self.data_dir / f"{pid}.pt")
        y   = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in self.md.at[pid, "labels"]:
            if lbl in self.class_to_idx:
                y[self.class_to_idx[lbl]] = 1.0
        return x, y

# ─── Compute raw %‑cover for all classes ──────────────────────────────────────
def compute_pct_all(model, device, data_dir, ids, md, class_to_idx, batch_size=64):
    ds     = PatchDataset(data_dir, ids, md, class_to_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    totals = torch.zeros(len(class_to_idx), device=device)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x     = x.to(device)
            logits= model(x)
            probs = torch.sigmoid(logits)
            totals += probs.sum(dim=0)
    totals = totals.cpu().numpy()
    if totals.sum() == 0:
        raise RuntimeError("No predictions made!")
    return totals / totals.sum() * 100.0    # percent cover per class

# ─── Main ────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1) load metadata & class‑map
    md          = pd.read_parquet(args.metadata, engine="pyarrow")
    all_labels  = sorted({lbl for lbls in md.labels for lbl in lbls})
    class_to_idx= {c:i for i,c in enumerate(all_labels)}

    # 2) load your model
    model = build_model(len(all_labels)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 3) define the groups you want
    GROUPS = {
        "Forest":       ["Broad-leaved forest","Coniferous forest","Mixed forest","Transitional woodland, shrub"],
        "Urban":        ["Urban fabric","Industrial or commercial units"],
        "Agriculture":  ["Arable land","Permanent crops","Pastures","Complex cultivation patterns"],
        "Water":        ["Inland waters","Marine waters","Inland wetlands","Coastal wetlands"],
    }

    # helper: get all Portugal patch IDs for a year
    def ids_for_year(y):
        mask = (
            (md.country=="Portugal") &
            md.patch_id.str.startswith("S2") &
            (md.patch_id.str[11:15].astype(int)==y)
        )
        ids = md.loc[mask,"patch_id"].tolist()
        print(f"  • {len(ids)} patches in Portugal from {y}")
        return ids

    print(f"Gathering IDs for years {args.year_a} vs {args.year_b}…")
    ids_a = ids_for_year(args.year_a)
    ids_b = ids_for_year(args.year_b)
    print()

    # 4) compute full class distributions
    print("Computing percent‑cover for each year…")
    pct_all_a = compute_pct_all(model, device, args.portugal_dir, ids_a, md, class_to_idx, args.batch_size)
    pct_all_b = compute_pct_all(model, device, args.portugal_dir, ids_b, md, class_to_idx, args.batch_size)

    # 5) aggregate per group
    results = []
    for name, cats in GROUPS.items():
        idxs = [class_to_idx[c] for c in cats if c in class_to_idx]
        pa = pct_all_a[idxs].sum()
        pb = pct_all_b[idxs].sum()
        results.append((name, pa, pb, pb-pa))

    # 6) print table
    print("\n🏷️  Land‑cover change by group:")
    print(f"{'Group':15} {args.year_a:>6}   {args.year_b:>6}   Δ")
    for name, pa, pb, d in results:
        print(f"{name:15} {pa:6.2f}%   {pb:6.2f}%   {d:+6.2f}%")

    # 7) bar chart
    groups = [r[0] for r in results]
    delta  = [r[3] for r in results]
    plt.figure(figsize=(8,4))
    plt.bar(groups, delta, edgecolor='k')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.ylabel(f"% change {args.year_b} – {args.year_a}")
    plt.title(f"Land‑cover change in Portugal: {args.year_a}→{args.year_b}")
    plt.tight_layout()
    out_png = "landcover_change.png"
    plt.savefig(out_png)
    print(f"\n📊 Bar chart saved to {out_png}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",    required=True, help="data/metadata.parquet")
    p.add_argument("--portugal-dir",required=True, help="Portugal .pt folder")
    p.add_argument("--checkpoint",  required=True, help="best_ft.pt")
    p.add_argument("--year-a",      type=int, default=2017, help="first year")
    p.add_argument("--year-b",      type=int, default=2018, help="second year")
    p.add_argument("--batch-size",  type=int, default=64)
    args = p.parse_args()
    main(args)