
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from finetune_portugal import build_model, BigEarthNetDataset  # reuse your existing classes


def get_landcover_distribution(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()

    class_totals = torch.zeros(dataset.num_classes).to(device)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            class_totals += probs.sum(dim=0)
    print("Class totals:", class_totals)
    print("Total sum:", class_totals.sum())
    percentages = (class_totals / class_totals.sum()) * 100
    return percentages.cpu().numpy()


def main(args):
    
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load metadata and class map
    md = pd.read_parquet(args.metadata, engine="pyarrow")
    all_labels = sorted({lbl for lbls in md.labels for lbl in lbls})
    class_to_idx = {c: i for i, c in enumerate(all_labels)}

    # Rebuild dataset
    portugal_dataset = BigEarthNetDataset(
        args.portugal_dir, md, lambda c: c == "Portugal", class_to_idx, device=device
    )

    # Load model
    model = build_model(len(class_to_idx)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Compute distribution
    distribution = get_landcover_distribution(model, portugal_dataset, device)

    print("\n📊 Land Cover Prediction Distribution (Portugal):")
    for cls, pct in zip(class_to_idx.keys(), distribution):
        print(f"{cls:30} {pct:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Path to metadata.parquet")
    parser.add_argument("--portugal-dir", required=True, help="Path to Portugal .pt directory")
    parser.add_argument("--checkpoint", required=True, help="Path to best_ft.pt")
    args = parser.parse_args()

    main(args)
