# Scratch Text-to-Image Diffusion (Project Showcase)

This repository contains a **from-scratch, educational text-to-image diffusion model** in PyTorch.

It includes:
- Character-level text tokenizer + GRU text encoder
- Small conditional U-Net noise predictor
- DDPM forward/reverse process implementation
- Classifier-free guidance (CFG)
- Training on COCO captions (default), with optional STL-10/CIFAR-10
- Per-epoch validation evaluation and loss curve tracking
- Per-epoch timestamped sample generations, checkpoints, and metrics logs

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Train

```bash
python train.py --dataset coco --coco_split train --epochs 10 --batch_size 32 --timesteps 300 --image_size 64 --num_workers 2
```

Artifacts:
- `outputs/last.pt` checkpoint
- `outputs/ckpt_<run_timestamp>_epoch_XXX_<timestamp>.pt`
- `outputs/sample_<run_timestamp>_epoch_XXX_<timestamp>.png`
- `outputs/loss_curve_<run_timestamp>_epoch_XXX_<timestamp>.png`
- `outputs/metrics_<run_timestamp>.csv`

## 3) Generate from text

```bash
python sample.py --checkpoint outputs/last.pt --prompts "a photo of a cat" "a photo of a ship" "a photo of a dog" "a photo of a truck" "a photo of an airplane" --output outputs/generated.png
```

## Colab Quick Confirm

Use [COLAB.md](./COLAB.md) for a free Colab-ready training and inference workflow.

## Notes
- This is a compact educational implementation, not a production-quality large model.
- For a faster demo on CPU, lower `--timesteps` (for example `100`) and epochs.
- Best results come from training on GPU.
