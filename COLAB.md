# Colab Run Guide

Use this to confirm training + inference on free Google Colab.

## 1) Runtime Setup

In Colab:
- Runtime -> Change runtime type -> Hardware accelerator -> GPU

## 2) Clone + Install

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Diffusion-system
pip install -r requirements.txt
```

## 3) Smoke Test

```bash
python test.py
```

Expected output includes tensor shapes for input/noisy/predicted noise.

## 4) Training (Colab-Friendly)

```bash
python train.py --dataset stl10 --epochs 1 --batch_size 32 --timesteps 150 --image_size 64 --num_workers 2 --sample_every 1 --device cuda
```

This writes:
- `outputs/last.pt`
- `outputs/sample_epoch_001.png`

## 5) Inference

```bash
python sample.py \
  --checkpoint outputs/last.pt \
  --prompts "a photo of a cat" "a photo of a ship" "a photo of a dog" "a photo of a truck" "a photo of an airplane" \
  --output outputs/generated.png \
  --device cuda
```

## 6) Download Artifacts

From the left file panel in Colab, download:
- `outputs/last.pt`
- `outputs/generated.png`
