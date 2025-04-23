#!/usr/bin/env python
"""
Preprocess BigEarthNet‑S2:
  • Recursively finds valid patch folders with Sentinel-2 bands
  • Converts them into (10, 120, 120) tensors scaled to [0, 1]
  • Saves each .pt file in a mirrored folder under --output-path
  • Streams one patch at a time (doesn't build full list in memory)
"""

import os, glob, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# --- configuration ----------------------------------------------------------
BAND_ORDER = ['B02', 'B03', 'B04',        # 10‑m
              'B05', 'B06', 'B07',        # 20‑m red-edge
              'B08',                      # 10‑m NIR
              'B11', 'B12',               # 20‑m SWIR
              'B8A']                      # 20‑m NIR narrow
UPSAMPLE_20M = {'B05', 'B06', 'B07', 'B11', 'B12', 'B8A'}

# ---------------------------------------------------------------------------

def load_band(path: Path, upscale: bool) -> np.ndarray:
    img = Image.open(path)
    if upscale:
        img = img.resize((120, 120), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)

def tensorize_patch(patch_dir: Path) -> torch.Tensor:
    bands = []
    for band in BAND_ORDER:
        matches = list(patch_dir.glob(f'*_{band}.tif'))
        if not matches:
            raise FileNotFoundError(f'Missing band {band}')
        band_arr = load_band(matches[0], upscale=(band in UPSAMPLE_20M))
        bands.append(band_arr)
    stack = np.stack(bands, axis=0) / 10_000.0
    return torch.tensor(stack, dtype=torch.float32)

def find_patch_folders(root: Path):
    """Generator that yields each folder containing *_B02.tif"""
    for dirpath, dirnames, filenames in os.walk(root):
        if any(name.endswith('_B02.tif') for name in filenames):
            yield Path(dirpath)

def run(dataset_path: Path, output_path: Path, max_patches=None):
    good, bad = 0, 0
    patch_stream = find_patch_folders(dataset_path)

    if max_patches:
        patch_stream = (p for i, p in enumerate(patch_stream) if i < max_patches)

    for patch_dir in tqdm(patch_stream, desc="Processing patches", unit="patch"):
        flat_name = patch_dir.name + ".pt"
        out_file = output_path / flat_name
        output_path.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            continue

        try:
            tensor = tensorize_patch(patch_dir)
            torch.save(tensor, out_file)
            good += 1
        except Exception as e:
            bad += 1
            print(f"[skip] {relative_subpath} – {e}")

    print(f"\n Finished preprocessing\n   Saved: {good}\n   Skipped: {bad}\n")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True,
                        help='Path to BigEarthNet-S2 root folder')
    parser.add_argument('--output-path', required=True,
                        help='Where to save .pt files (mirrors folder structure)')
    parser.add_argument('--max-patches', type=int,
                        help='Optional cap for number of patches to process')
    args = parser.parse_args()

    run(Path(args.dataset_path).expanduser(),
        Path(args.output_path).expanduser(),
        args.max_patches)
