# Scratch Text-to-Image Diffusion (Project Showcase)

This repository contains a **from-scratch, educational text-to-image diffusion model** in PyTorch.

It includes:
- Character-level text tokenizer + BiGRU attention text encoder
- Deeper FiLM-conditioned U-Net with bottleneck self-attention
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
python train.py --dataset coco --coco_split train --epochs 40 --batch_size 32 --val_batch_size 32 --timesteps 400 --image_size 64 --num_workers 8 --lr 1e-4 --device cuda --amp --compile_model --channels_last
```

Artifacts:
- `outputs/run_<timestamp_and_config>/last.pt` checkpoint
- `outputs/run_<timestamp_and_config>/ckpt_<run_timestamp>_epoch_XXX_<timestamp>.pt`
- `outputs/run_<timestamp_and_config>/sample_<run_timestamp>_epoch_XXX_<timestamp>.png`
- `outputs/run_<timestamp_and_config>/loss_curve_<run_timestamp>_epoch_XXX_<timestamp>.png`
- `outputs/run_<timestamp_and_config>/metrics_<run_timestamp>.csv`
- `outputs/latest_run.txt` pointer to the most recent run directory

Resume training example:

```bash
python train.py --dataset coco --epochs 60 --resume_checkpoint outputs/run_<your_run_name>/last.pt --device cuda --amp --compile_model --channels_last
```

## 3) Generate from text

```bash
python sample.py --checkpoint outputs/last.pt --prompts "a photo of a cat" "a photo of a ship" "a photo of a dog" "a photo of a truck" "a photo of an airplane" --output outputs/generated.png --cfg_scale 6.0 --device cuda --amp --channels_last
```

## Colab Quick Confirm

Use [COLAB.md](./COLAB.md) for a free Colab-ready training and inference workflow.

## Notes
- This is a compact educational implementation, not a production-quality large model.
- For a faster demo on CPU, lower `--timesteps` (for example `100`) and epochs.
- Best results come from training on GPU.
