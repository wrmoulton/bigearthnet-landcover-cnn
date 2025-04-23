import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from finetune_portugal import build_model, BigEarthNetDataset
import pandas as pd
from torch.utils.data import DataLoader

# 1) load metadata & labels
md = pd.read_parquet("data/metadata.parquet", engine="pyarrow")
all_labels = sorted({lbl for lbls in md.labels for lbl in lbls})
class_to_idx = {c:i for i,c in enumerate(all_labels)}

# 2) rebuild model & load checkpoint
model = build_model(len(all_labels))
model.load_state_dict(torch.load("best_pre.pt", map_location="cpu"))
model.eval()

# 3) create your Portugal or global val loader exactly as before,
ds = BigEarthNetDataset("data/global_subset", md, lambda c: c!="Portugal", class_to_idx)
_, val_ds = torch.utils.data.random_split(
    ds,
    [int(len(ds)*0.8), len(ds) - int(len(ds)*0.8)],
    generator=torch.Generator().manual_seed(42)
)
loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

# 4) collect preds & trues
all_true, all_pred = [], []
with torch.no_grad():
    for x,y in loader:
        logits = model(x)
        preds = (torch.sigmoid(logits)>0.5).numpy()
        all_pred.append(preds)
        all_true.append(y.numpy())
y_true = np.vstack(all_true)
y_pred = np.vstack(all_pred)

# 5) pick a class index that actually occurs, e.g. class 5
cls = 5
print(all_labels[cls], "positives in val:", y_true[:,cls].sum())
cm = confusion_matrix(y_true[:,cls], y_pred[:,cls], labels=[0,1])
print("Confusion matrix for", all_labels[cls], ":\n", cm)
