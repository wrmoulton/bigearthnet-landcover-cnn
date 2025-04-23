
"""
Fine-tune pre-trained ResNet50 on Portugal BigEarthNet patches with K‑Fold validation.

Usage:
  python finetune_portugal_kfold.py \
    --metadata     data/metadata.parquet \
    --portugal-dir data/portugal_subset \
    --pre-ckpt     best_pre.pt \
    --folds        5 \
    --ft-epochs    20 \
    --batch-size   32 \
    --lr           1e-4 \
    --weight-decay 1e-4
"""
import argparse, time, random
from pathlib import Path
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet50
from torch.cuda.amp import autocast
try:
    from torch.amp import GradScaler  
except ImportError:
    from torch.cuda.amp import GradScaler
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

# ─── Dataset ────────────────────────────────────────────────────────────────
class BigEarthNetDataset(Dataset):
    def __init__(self, data_dir, metadata_df, class_to_idx, device):
        self.data_dir = Path(data_dir)
        self.md = metadata_df.set_index('patch_id')
        # Portugal only
        port = metadata_df[metadata_df['country']=='Portugal']['patch_id'].astype(str)
        present = {p.stem for p in self.data_dir.glob('*.pt')}
        self.ids = [pid for pid in port if pid in present]
        dropped = set(port) - set(self.ids)
        if dropped:
            print(f"[warn] Dropped {len(dropped)} missing files (e.g. {list(dropped)[:3]})")
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)
        self.device = device

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        x = torch.load(self.data_dir / f"{pid}.pt")
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in self.md.at[pid,'labels']:
            if lbl in self.class_to_idx:
                y[self.class_to_idx[lbl]] = 1.0
        return x, y

# ─── Model Factory ──────────────────────────────────────────────────────────
def build_model(num_classes, dropout=0.3):
    m = resnet50(weights=None)
    m.conv1 = nn.Conv2d(10,64,kernel_size=7,stride=2,padding=3,bias=False)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    return m

# ─── Training / Evaluation ───────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train(); total_loss=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        with autocast():
            logits=model(x)
            loss=F.binary_cross_entropy_with_logits(logits,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)

def eval_loader(model, loader, device):
    model.eval(); all_true,all_pred=[],[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=model(x)
            preds=(torch.sigmoid(logits)>0.5).cpu().numpy()
            all_pred.append(preds)
            all_true.append(y.cpu().numpy())
    y_true=np.vstack(all_true); y_pred=np.vstack(all_pred)
    acc=accuracy_score(y_true.flatten(),y_pred.flatten())
    f1=f1_score(y_true.flatten(),y_pred.flatten(),zero_division=0)
    return acc,f1

# ─── Main ────────────────────────────────────────────────────────────────────
def main(args):
    best_f1 = -1
    best_model_state = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:',device)
    # load metadata
    md = pd.read_parquet(args.metadata, engine='pyarrow')
    # class map
    all_labels=sorted({lbl for labels in md.labels for lbl in labels})
    class_to_idx={c:i for i,c in enumerate(all_labels)}
    # dataset
    dataset = BigEarthNetDataset(args.portugal_dir, md, class_to_idx, device)
    print('Total Portugal patches:', len(dataset))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_metrics = []
    for fold,(train_idx,val_idx) in enumerate(kf.split(range(len(dataset))),1):
        print(f'=== Fold {fold}/{args.folds} ===')
        # subsets
        tr_sub = Subset(dataset, train_idx)
        vl_sub = Subset(dataset, val_idx)
        tr_loader = DataLoader(tr_sub, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        vl_loader = DataLoader(vl_sub, batch_size=args.batch_size, shuffle=False,num_workers=4, pin_memory=True)
        # fresh copy and freeze backbone
        model = build_model(len(all_labels), dropout=args.dropout).to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(device='cuda')
        # train
        for epoch in range(1,args.ft_epochs+1):
            t0=time.time()
            loss = train_one_epoch(model, tr_loader, optimizer, scaler, device)
            acc,f1 = eval_loader(model, vl_loader, device)
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()
            print(f'Epoch {epoch}/{args.ft_epochs}  train_loss={loss:.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}  ({(time.time()-t0)/60:.1f}m)')
        fold_metrics.append((acc,f1))
    avg_acc = np.mean([m[0] for m in fold_metrics])
    avg_f1  = np.mean([m[1] for m in fold_metrics])
    print(f'>>> KFold mean val_acc={avg_acc:.4f}  val_f1={avg_f1:.4f}')
    torch.save(best_model_state, 'best_kfold_scratch.pt')
    print(' Best model saved as best_kfold_scratch.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',    required=True)
    p.add_argument('--portugal-dir',required=True)
    p.add_argument('--folds',       type=int, default=5)
    p.add_argument('--ft-epochs',   type=int, default=20)
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight-decay',type=float, default=1e-4)
    p.add_argument('--dropout',     type=float, default=0.3)
    args=p.parse_args()
    random.seed(42)
    main(args)
