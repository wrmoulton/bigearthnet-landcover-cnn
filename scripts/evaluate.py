#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from finetune_portugal import BigEarthNetDataset, build_model  

def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.binary_cross_entropy_with_logits(logits, y).item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_pred.append(preds)
            all_true.append(y.cpu().numpy())
    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    loss = total_loss / len(loader.dataset)
    acc  = accuracy_score(y_true.flatten(), y_pred.flatten())
    f1   = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    cm0  = confusion_matrix(y_true[:,0], y_pred[:,0], labels=[0,1])
    return loss, acc, f1, cm0

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",     required=True)
    p.add_argument("--portugal-dir", required=True)
    p.add_argument("--ft-ckpt",      required=True)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--val-frac",     type=float, default=0.2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = pd.read_parquet(args.metadata, engine="pyarrow")
    all_labels = sorted({lbl for lbls in md.labels for lbl in lbls})
    class_to_idx = {c:i for i,c in enumerate(all_labels)}

    # rebuild Portugal dataset and split 70/20/10
    full_ds = BigEarthNetDataset(args.portugal_dir, md, lambda c: c=="Portugal", class_to_idx)
    n = len(full_ds)
    n_train = int(0.7*n)
    n_val   = int(0.2*n)
    n_test  = n - n_train - n_val
    train_val, test_ds = random_split(full_ds, [n_train+n_val, n_test],
                                      generator=torch.Generator().manual_seed(42))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # load model and weights
    model = build_model(len(all_labels)).to(device)
    model.load_state_dict(torch.load(args.ft_ckpt, map_location=device))

    loss, acc, f1, cm0 = evaluate(model, test_loader, device)
    print(f"🔹 Test Loss = {loss:.4f}")
    print(f"🔹 Test Accuracy = {acc:.4f}")
    print(f"🔹 Test F1 = {f1:.4f}")
    print("🔹 Confusion Matrix (class0):")
    print(cm0)
