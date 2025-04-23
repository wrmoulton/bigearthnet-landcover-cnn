# BigEarthNet Portugal Land Cover Analysis 🌍

This project fine-tunes a ResNet-50 model to detect land cover changes in Portugal using BigEarthNet-S2 satellite data. We evaluated forest loss, urban expansion, and other land use changes between 2017 and 2018.
##  Dataset Setup

You must download the [BigEarthNet-S2](https://bigearth.net/#download) dataset:

1. Go to https://bigearth.net/#download and request access.
2. Once approved, download the **Portugal subset** or full dataset.
3. Extract `.pt` patches into a folder like `data/portugal_subset`.
4. Place the `metadata.parquet` file into `data/` (or update your paths accordingly).

##  Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt

```
## How To Run
1. Train with K-Fold:
```bash
 python k-fold-portugal-finetune.py --metadata data/metadata.parquet --portugal-dir data/portugal_subset --folds 5 --ft-epochs 20 --checkpoint best_kfold.pt
 ```
2. Evaluate Model:
```bash
python test_model.py --metadata data/metadata.parquet --portugal-dir data/portugal_subset --ft-ckpt best_kfold.pt
```
3. Land Coverage Analysis:
```bash
python landcover_change.py --metadata data/metadata.parquet --portugal-dir data/portugal_subset --checkpoint best_kfold.pt --year-a 2017 --year-b 2018 --categories "Broad-leaved forest" "Urban fabric"
```
4. Grad-CAM Visualization:
```bash
python gradcam_visualize.py --checkpoint best_kfold.pt --portugal-dir data/portugal_subset --metadata data/metadata.parquet
```
## Results Summary
Test F1 Score: 0.90 (K-Fold)

Forest Cover Decline: -2.46%

Urban Growth Identified via Grad-CAM

## Author: William Moulton - UCF Machine Learning Grad Class Project
