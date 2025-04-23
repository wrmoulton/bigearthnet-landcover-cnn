#!/usr/bin/env python3
import argparse, random
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from finetune_portugal import build_model

# ─── Load a single patch and convert to RGB ────────────────────────────────
def load_patch(pt_dir, patch_id):
    x = torch.load(Path(pt_dir) / f"{patch_id}.pt")  # (10, 120, 120)
    rgb = x[[3, 2, 1], :, :].numpy().transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return x.unsqueeze(0), (rgb * 255).astype(np.uint8)

# ─── Compute CAM and overlay ────────────────────────────────────────────────
def visualize_cam(patch_id, model, device, pt_dir, class_to_idx, target_class):
    x, rgb = load_patch(pt_dir, patch_id)
    input_tensor = x.to(device)

    # Forward pass and extract CAM
    model.zero_grad()
    cam_extractor = GradCAM(model, target_layer="layer4")
    logits = model(input_tensor)
    cams = cam_extractor(class_idx=class_to_idx[target_class], scores=logits)
    cam = cams[0].squeeze().cpu().numpy()


    # Resize CAM to match image
    cam_resized = cv2.resize(cam, (120, 120))
    cam_img = Image.fromarray((cam_resized * 255 / cam_resized.max()).astype(np.uint8))
    rgb_img = Image.fromarray(rgb)

    # Overlay heatmap
    overlay = overlay_mask(rgb_img, cam_img, alpha=0.5)

    return rgb_img, overlay

# ─── Main ───────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load metadata and pick random patches
    md = pd.read_parquet(args.metadata, engine="pyarrow")
    md = md[md.country == "Portugal"]
    md["year"] = md.patch_id.str[11:15].astype(int)

    ids_2017 = md[md.year == 2017]["patch_id"].tolist()
    ids_2018 = md[md.year == 2018]["patch_id"].tolist()

    if not ids_2017 or not ids_2018:
        raise RuntimeError("No patches found for 2017 or 2018.")

    pid_2017 = random.choice(ids_2017)
    pid_2018 = random.choice(ids_2018)
    print(f"Selected 2017 patch: {pid_2017}")
    print(f"Selected 2018 patch: {pid_2018}")

    # Load model
    model = build_model(num_classes=19).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Label index map
    class_to_idx = {l:i for i, l in enumerate(sorted([
        "Continuous urban fabric", "Discontinuous urban fabric",
        "Industrial or commercial units", "Pastures", "Broad-leaved forest",
        "Coniferous forest", "Mixed forest", "Natural grasslands", "Moors and heathland",
        "Sclerophyllous vegetation", "Transitional woodland, shrub", "Beaches, dunes, sands",
        "Bare rock", "Sparsely vegetated areas", "Inland marshes", "Peat bogs",
        "Water bodies", "Permanently irrigated land", "Complex cultivation patterns"
    ]))}

    # Visualize both
    raw1, cam1 = visualize_cam(pid_2017, model, device, args.portugal_dir, class_to_idx, args.forest_label)
    raw2, cam2 = visualize_cam(pid_2018, model, device, args.portugal_dir, class_to_idx, args.forest_label)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0][0].imshow(raw1); axes[0][0].set_title(f"RGB: {pid_2017}"); axes[0][0].axis("off")
    axes[0][1].imshow(cam1); axes[0][1].set_title(f"Grad-CAM: {pid_2017}"); axes[0][1].axis("off")
    axes[1][0].imshow(raw2); axes[1][0].set_title(f"RGB: {pid_2018}"); axes[1][0].axis("off")
    axes[1][1].imshow(cam2); axes[1][1].set_title(f"Grad-CAM: {pid_2018}"); axes[1][1].axis("off")
    plt.suptitle(f"Grad-CAM Activation for '{args.forest_label}'")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--portugal-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--forest-label", default="Broad-leaved forest")
    args = parser.parse_args()
    main(args)
