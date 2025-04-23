#!/usr/bin/env python3
import argparse, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ─── Dataset ────────────────────────────────────────────────────────────────
class BigEarthNetDataset(Dataset):
    def __init__(self, data_dir, metadata_df, country_filter, class_to_idx, device = torch.device("cpu")):
        self.data_dir = Path(data_dir)
        self.md       = metadata_df.set_index("patch_id")

        # filter metadata by country
        mask    = metadata_df["country"].apply(country_filter)
        desired = metadata_df.loc[mask, "patch_id"].astype(str)

        # only keep IDs for which .pt exists
        present = {p.stem for p in self.data_dir.glob("*.pt")}
        self.ids = [pid for pid in desired if pid in present]

        # warn if any got dropped
        dropped = set(desired) - set(self.ids)
        if dropped:
            print(f"[warn] Dropped {len(dropped)} missing files (e.g. {list(dropped)[:3]})")

        self.class_to_idx = class_to_idx
        self.num_classes  = len(class_to_idx)
        self.device = device

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        x = torch.load(self.data_dir / f"{pid}.pt", weights_only=True)
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in self.md.at[pid, "labels"]:
            if lbl in self.class_to_idx:
                y[self.class_to_idx[lbl]] = 1.0
        return x, y

# ─── Model Factory ──────────────────────────────────────────────────────────
def build_model(num_classes):
    m = resnet50(weights=None)
    m.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, num_classes)
    )
    return m

# ─── Metrics ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_pred.append(preds)
            all_true.append(y.cpu().numpy())
    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    avg_loss = total_loss / len(loader.dataset)
    # micro-averaged over all labels
    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    f1  = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    # Example confusion matrix for the first class
    cm0 = confusion_matrix(y_true[:,0], y_pred[:,0], labels=[0,1])
    return avg_loss, acc, f1, cm0

# ─── Training ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# ─── Main ────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # metadata + classes
    md = pd.read_parquet(args.metadata, engine="pyarrow")
    all_labels = sorted({lbl for lbls in md.labels for lbl in lbls})
    class_to_idx = {c:i for i,c in enumerate(all_labels)}
    num_classes = len(all_labels)
    print(f"{num_classes} classes loaded")

    # datasets
    global_ds = BigEarthNetDataset(args.global_dir, md, lambda c: c!="Portugal", class_to_idx)
    port_ds   = BigEarthNetDataset(args.portugal_dir, md, lambda c: c=="Portugal", class_to_idx)
    # 70/20/10 split of Portugal for train/val/test
    total = len(port_ds)
    n_train = int(0.7 * total)
    n_val   = int(0.2 * total)
    n_test  = total - n_train - n_val

    train_ds_full, val_ds, test_ds = random_split(
        port_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
)
    # splits
    def split(ds, frac):
        total = len(ds)
        nv = int(total * frac)
        nt = total - nv
        return random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(42))

    g_tr, g_val = split(global_ds, args.val_frac)
    p_train = DataLoader(train_ds_full, batch_size=args.batch_size, shuffle=True,num_workers=4, pin_memory=True)
    p_valld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,num_workers=4, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,num_workers=4, pin_memory=True)


    print(f"Global train/val: {len(g_tr)}/{len(g_val)}")
    print(f"Portugal train/val/test: {len(train_ds_full)}/{len(val_ds)}/{len(test_ds)}")


    # loaders
    g_train = DataLoader(g_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    g_valld = DataLoader(g_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #p_train = DataLoader(p_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #p_valld = DataLoader(p_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model + scaler
    model = build_model(num_classes).to(device)
    scaler = GradScaler()

    # Stage 1: Global pre-train
    opt = optim.Adam(model.parameters(), lr=args.pre_lr)
    best_gloss = float('inf')
    for e in range(1, args.pre_epochs+1):
        t0 = time.time()
        tl = train_epoch(model, g_train, opt, scaler, device)
        vl, vacc, vf1, cm0 = evaluate(model, g_valld, device)
        print(f"[Global E{e}/{args.pre_epochs}] train={tl:.4f} val={vl:.4f} acc={vacc:.4f} f1={vf1:.4f}")
        print(f" ConfMat[class0]:\n{cm0}")
        print(f" Time={(time.time()-t0)/60:.1f} min")
        if vl < best_gloss:
            best_gloss = vl
            torch.save(model.state_dict(), args.pre_ckpt)

    # Stage 2: Portugal fine-tune
    model.load_state_dict(torch.load(args.pre_ckpt, map_location=device))
    for n,p in model.named_parameters():
        if not n.startswith('fc'):
            p.requires_grad=False
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ft_lr)
    best_ploss = float('inf')
    for e in range(1, args.ft_epochs+1):
        t0 = time.time()
        tl = train_epoch(model, p_train, opt, scaler, device)
        vl, vacc, vf1, cm0 = evaluate(model, p_valld, device)
        print(f"[Portugal E{e}/{args.ft_epochs}] train={tl:.4f} val={vl:.4f} acc={vacc:.4f} f1={vf1:.4f}")
        print(f" ConfMat[class0]:\n{cm0}")
        print(f" Time={(time.time()-t0)/60:.1f} min")
        if vl < best_ploss:
            best_ploss = vl
            torch.save(model.state_dict(), args.ft_ckpt)

    print("Done. Best Global val:", best_gloss, "Best Portugal val:", best_ploss)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",     required=True, help="data/metadata.parquet")
    p.add_argument("--global-dir",   required=True, help="data/global_subset")
    p.add_argument("--portugal-dir", required=True, help="data/portugal_subset")
    p.add_argument("--pre-epochs", type=int, default=10)
    p.add_argument("--ft-epochs",  type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pre-lr",     type=float, default=1e-3)
    p.add_argument("--ft-lr",      type=float, default=1e-4)
    p.add_argument("--val-frac",   type=float, default=0.2)
    p.add_argument("--pre-ckpt",   default="best_pre.pt")
    p.add_argument("--ft-ckpt",    default="best_ft.pt")
    args = p.parse_args()

    random.seed(42)
    torch.backends.cudnn.benchmark = True
    main(args)

