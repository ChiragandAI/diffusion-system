# Scratch Text-to-Image Diffusion (Project Showcase)

This repository contains a **from-scratch, educational text-to-image diffusion model** in PyTorch.

It includes:
- Character-level text tokenizer + GRU text encoder
- Small conditional U-Net noise predictor
- DDPM forward/reverse process implementation
- Classifier-free guidance (CFG)
- Training on CIFAR-10 with auto-generated text prompts

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Train

```bash
python train.py --epochs 20 --batch_size 64 --timesteps 300 --num_workers 2
```

Artifacts:
- `outputs/last.pt` checkpoint
- `outputs/sample_epoch_XXX.png` preview generations

## 3) Generate from text

```bash
python sample.py --checkpoint outputs/last.pt --prompts "a photo of a cat" "a photo of a ship" --output outputs/generated.png
```

## Colab Quick Confirm

Use [COLAB.md](./COLAB.md) for a free Colab-ready training and inference workflow.

## Notes
- This is a compact educational implementation, not a production-quality large model.
- For a faster demo on CPU, lower `--timesteps` (for example `100`) and epochs.
- Best results come from training on GPU.
