# Colab Run Guide

Use this to confirm training + inference on free Google Colab.

## 1) Runtime Setup

In Colab:
- Runtime -> Change runtime type -> Hardware accelerator -> GPU

## 2) Clone + Install

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd diffusion-system
pip install -r requirements.txt
```

## 3) Download COCO Captions Dataset

```bash
mkdir -p data
wget -q -O data/train2017.zip http://images.cocodataset.org/zips/train2017.zip
wget -q -O data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
wget -q -O data/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q data/train2017.zip -d data
unzip -q data/val2017.zip -d data
unzip -q data/annotations_trainval2017.zip -d data
```

If you want a faster first run, use `--coco_split val` in training.

## 4) Smoke Test

```bash
python test.py
```

Expected output includes tensor shapes for input/noisy/predicted noise.

## 5) Training (COCO, 10 epochs)

```bash
python train.py --dataset coco --coco_split train --epochs 10 --batch_size 32 --timesteps 200 --image_size 64 --num_workers 2 --sample_every 2 --device cuda
```

This writes:
- `outputs/last.pt`
- `outputs/sample_<run_timestamp>_epoch_XXX_<timestamp>.png` after every epoch
- `outputs/loss_curve_<run_timestamp>_epoch_XXX_<timestamp>.png` after every epoch
- `outputs/metrics_<run_timestamp>.csv` with train and val loss history

## 6) Inference (5 Prompts)

```bash
python sample.py \
  --checkpoint outputs/last.pt \
  --prompts "a photo of a cat" "a photo of a ship" "a photo of a dog" "a photo of a truck" "a photo of an airplane" \
  --output outputs/generated.png \
  --device cuda
```

## 7) Download Artifacts

From the left file panel in Colab, download:
- `outputs/last.pt`
- `outputs/generated.png`
