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

## 5) Training (COCO, Better Defaults)

```bash
python train.py --dataset coco --coco_split train --epochs 40 --batch_size 32 --val_batch_size 32 --timesteps 400 --image_size 64 --num_workers 8 --lr 1e-4 --device cuda --amp --compile_model --channels_last
```

This writes:
- `outputs/run_<timestamp_and_config>/last.pt`
- `outputs/run_<timestamp_and_config>/sample_<run_timestamp>_epoch_XXX_<timestamp>.png` after every epoch
- `outputs/run_<timestamp_and_config>/loss_curve_<run_timestamp>_epoch_XXX_<timestamp>.png` after every epoch
- `outputs/run_<timestamp_and_config>/metrics_<run_timestamp>.csv` with train and val loss history
- `outputs/latest_run.txt` pointer to newest run directory

## 6) Inference (5 Prompts)

```bash
python sample.py \
  --checkpoint outputs/last.pt \
  --prompts "a photo of a cat" "a photo of a ship" "a photo of a dog" "a photo of a truck" "a photo of an airplane" \
  --cfg_scale 6.0 \
  --output outputs/generated.png \
  --device cuda \
  --amp --channels_last
```

## 7) Download Artifacts

From the left file panel in Colab, download:
- `outputs/last.pt`
- `outputs/generated.png`

## 8) Resume Training Later

```bash
python train.py --dataset coco --epochs 60 --resume_checkpoint outputs/run_<your_run_name>/last.pt --device cuda --amp --compile_model --channels_last
```

If the checkpoint already completed 40 epochs, this continues from epoch 41 to 60.
