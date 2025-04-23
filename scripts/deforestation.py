# deforestation.py
"""
Estimate forest cover change in Portugal between 2017 and 2018 using the fine-tuned CNN.

Usage:
  python deforestation.py \
    --metadata    data/metadata.parquet \
    --portugal-dir data/portugal_subset \
    --checkpoint  best_ft.pt
"""
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from finetune_portugal import build_model  # reuse your model factory

# ─── Dataset for arbitrary patch list ─────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, data_dir, patch_ids, md, class_to_idx, device):
        self.data_dir = Path(data_dir)
        self.ids = patch_ids
        self.md = md.set_index('patch_id')
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)
        self.device = device

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        x = torch.load(self.data_dir / f"{pid}.pt")
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in self.md.at[pid, 'labels']:
            if lbl in self.class_to_idx:
                y[self.class_to_idx[lbl]] = 1.0
        return x, y

# ─── Compute distribution over all classes ─────────────────────────────────────
def get_landcover_distribution(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    totals = torch.zeros(dataset.num_classes, device=device)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            totals += probs.sum(dim=0)
    if totals.sum() == 0:
        return np.zeros_like(totals.cpu().numpy())
    pct = (totals / totals.sum()) * 100.0
    return pct.cpu().numpy()

# ─── Main: filter by year & compute forest change ─────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA Available:", torch.cuda.is_available(), "-> Using", device)

    # load metadata
    md = pd.read_parquet(args.metadata, engine='pyarrow')
    all_labels = sorted({lbl for labels in md.labels for lbl in labels})
    class_to_idx = {c:i for i, c in enumerate(all_labels)}

    forest_classes = ["Broad-leaved forest", "Coniferous forest", "Mixed forest", "Transitional woodland, shrub"]
    forest_idx = [class_to_idx[c] for c in forest_classes if c in class_to_idx]

    model = build_model(len(all_labels)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    def pct_for_year(year):
        mask = (md.country == 'Portugal') & md.patch_id.apply(
            lambda pid: (pid.startswith("S2") and int(pid.split('_')[2][:4]) == year)
        )
        ids = md.loc[mask, 'patch_id'].astype(str).tolist()
        print(f"🧾 Number of patches from year {year}: {len(ids)}")
        ds = PatchDataset(args.portugal_dir, ids, md, class_to_idx, device)
        dist = get_landcover_distribution(model, ds, device, batch_size=args.batch_size)
        return dist[forest_idx].sum()

    pct_2017 = pct_for_year(2017)
    pct_2018 = pct_for_year(2018)
    change = pct_2018 - pct_2017

    print(f"\n🟢 Forest cover 2017: {pct_2017:.2f}%")
    print(f"🟤 Forest cover 2018: {pct_2018:.2f}%")
    print(f"📉 Estimated change: {change:+.2f}%")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata',     required=True, help='metadata.parquet')
    p.add_argument('--portugal-dir', required=True, help='Portugal .pt folder')
    p.add_argument('--checkpoint',   required=True, help='best_ft.pt')
    p.add_argument('--batch-size',   type=int, default=64)
    args = p.parse_args()
    main(args)