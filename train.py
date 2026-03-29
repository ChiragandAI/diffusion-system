import argparse
import csv
import os
from datetime import datetime
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from diffusion_scratch.data import build_text_dataset, build_val_text_dataset
from diffusion_scratch.diffusion import DiffusionScheduler, sample_ddpm
from diffusion_scratch.text_encoder import CharTokenizer, TinyTextEncoder
from diffusion_scratch.unet import TinyConditionalUNet


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny text-to-image diffusion model from scratch.")
    p.add_argument("--dataset", type=str, default="coco", choices=["coco", "stl10", "cifar10"])
    p.add_argument("--coco_split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=400)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--drop_text_prob", type=float, default=0.05)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--text_emb_dim", type=int, default=192)
    p.add_argument("--text_hidden_dim", type=int, default=384)
    p.add_argument("--base_channels", type=int, default=96)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--preview_cfg_scale", type=float, default=6.0)
    p.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision on CUDA.")
    p.add_argument("--no-amp", action="store_false", dest="amp", help="Disable automatic mixed precision.")
    p.add_argument("--compile_model", action="store_true", default=True, help="Use torch.compile for speed.")
    p.add_argument("--no-compile_model", action="store_false", dest="compile_model")
    p.add_argument("--channels_last", action="store_true", default=True, help="Use NHWC memory format on CUDA.")
    p.add_argument("--no-channels_last", action="store_false", dest="channels_last")
    p.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def maybe_display_image(path: str):
    try:
        from IPython.display import Image, display

        display(Image(filename=path))
    except Exception:
        pass


@torch.no_grad()
def evaluate_val_loss(unet, text_encoder, tokenizer, scheduler, loader, device: str, use_amp: bool, channels_last: bool):
    unet.eval()
    text_encoder.eval()

    total_loss = 0.0
    num_steps = 0

    for images, captions in loader:
        images = images.to(device, non_blocking=device.startswith("cuda"))
        if channels_last and device.startswith("cuda"):
            images = images.to(memory_format=torch.channels_last)
        t = scheduler.sample_timesteps(images.size(0))
        x_t, noise = scheduler.add_noise(images, t)

        token_ids = torch.stack([tokenizer.encode(c) for c in captions]).to(device)
        amp_ctx = get_autocast_ctx(device, use_amp)
        with amp_ctx:
            text_cond = text_encoder(token_ids)
            pred_noise = unet(x_t, t, text_cond)
            loss = F.mse_loss(pred_noise, noise)
        total_loss += loss.item()
        num_steps += 1

    return total_loss / max(1, num_steps)


def plot_losses(epochs, train_losses, val_losses, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train vs Val Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def configure_cuda_runtime(args):
    if not args.device.startswith("cuda"):
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_autocast_ctx(device: str, use_amp: bool):
    if not (use_amp and device.startswith("cuda")):
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def build_grad_scaler(device: str, use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device="cuda", enabled=use_amp and device.startswith("cuda"))
    return torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    configure_cuda_runtime(args)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv_path = os.path.join(args.output_dir, f"metrics_{run_timestamp}.csv")

    device = args.device
    tokenizer = CharTokenizer(max_length=args.max_text_len)
    print(
        "runtime:",
        f"device={device}",
        f"amp={args.amp and device.startswith('cuda')}",
        f"compile={args.compile_model}",
        f"channels_last={args.channels_last and device.startswith('cuda')}",
        f"grad_accum={args.gradient_accumulation_steps}",
    )

    train_dataset = build_text_dataset(
        name=args.dataset,
        root=args.data_dir,
        image_size=args.image_size,
        coco_split=args.coco_split,
    )
    val_dataset = build_val_text_dataset(
        name=args.dataset,
        root=args.data_dir,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    text_encoder = TinyTextEncoder(
        vocab_size=tokenizer.vocab_size,
        emb_dim=args.text_emb_dim,
        hidden_dim=args.text_hidden_dim,
    ).to(device)
    unet = TinyConditionalUNet(
        in_channels=3,
        base_channels=args.base_channels,
        cond_dim=args.text_hidden_dim,
        time_dim=args.text_hidden_dim,
    ).to(device)

    if args.channels_last and device.startswith("cuda"):
        unet = unet.to(memory_format=torch.channels_last)

    if args.compile_model and hasattr(torch, "compile"):
        unet = torch.compile(unet)
        text_encoder = torch.compile(text_encoder)

    scheduler = DiffusionScheduler(timesteps=args.timesteps, device=device)

    optimizer = optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    use_amp = args.amp and device.startswith("cuda")
    scaler = build_grad_scaler(device, use_amp)

    preview_prompts = [
        "a photo of a cat",
        "a photo of a ship",
        "a photo of a dog",
        "a photo of a truck",
        "a photo of an airplane",
    ]

    epoch_ids = []
    train_losses = []
    val_losses = []

    start_epoch = 1
    global_step = 0

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        unet.load_state_dict(ckpt["unet"])
        text_encoder.load_state_dict(ckpt["text_encoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        run_timestamp = ckpt.get("run_timestamp", run_timestamp)
        metrics_csv_path = os.path.join(args.output_dir, f"metrics_{run_timestamp}.csv")

        prev_train_losses = ckpt.get("train_losses", [])
        prev_val_losses = ckpt.get("val_losses", [])
        if prev_train_losses and prev_val_losses:
            epoch_ids = list(range(1, len(prev_train_losses) + 1))
            train_losses = list(prev_train_losses)
            val_losses = list(prev_val_losses)

        print(
            f"Resumed from {args.resume_checkpoint} at epoch={start_epoch} "
            f"(completed={start_epoch - 1}, global_step={global_step})"
        )

    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "timestamp", "train_loss", "val_loss", "sample_path", "plot_path"])

    if start_epoch > args.epochs:
        print(
            f"Nothing to train: start_epoch={start_epoch} is greater than target epochs={args.epochs}. "
            "Increase --epochs to continue training."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        unet.train()
        text_encoder.train()

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, captions) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=device.startswith("cuda"))
            if args.channels_last and device.startswith("cuda"):
                images = images.to(memory_format=torch.channels_last)
            t = scheduler.sample_timesteps(images.size(0))
            x_t, noise = scheduler.add_noise(images, t)

            token_ids = torch.stack([tokenizer.encode(c) for c in captions]).to(device)

            if args.drop_text_prob > 0:
                drop_mask = torch.rand(images.size(0), device=device) < args.drop_text_prob
                token_ids[drop_mask] = tokenizer.encode("").to(device)

            amp_ctx = get_autocast_ctx(device, use_amp)
            with amp_ctx:
                text_cond = text_encoder(token_ids)
                pred_noise = unet(x_t, t, text_cond)
                loss = F.mse_loss(pred_noise, noise)
                scaled_loss = loss / max(1, args.gradient_accumulation_steps)

            scaler.scale(scaled_loss).backward()

            if step % max(1, args.gradient_accumulation_steps) == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(unet.parameters()) + list(text_encoder.parameters()),
                        args.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            global_step += 1

            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={running_loss / step:.4f}")

        if len(train_loader) % max(1, args.gradient_accumulation_steps) != 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(unet.parameters()) + list(text_encoder.parameters()),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = evaluate_val_loss(
            unet,
            text_encoder,
            tokenizer,
            scheduler,
            val_loader,
            device,
            use_amp=use_amp,
            channels_last=args.channels_last,
        )
        lr_scheduler.step()

        epoch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = os.path.join(args.output_dir, f"sample_{run_timestamp}_epoch_{epoch:03d}_{epoch_timestamp}.png")
        plot_path = os.path.join(args.output_dir, f"loss_curve_{run_timestamp}_epoch_{epoch:03d}_{epoch_timestamp}.png")
        ckpt_path = os.path.join(args.output_dir, f"ckpt_{run_timestamp}_epoch_{epoch:03d}_{epoch_timestamp}.pt")

        ckpt = {
            "unet": unet.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "dataset": args.dataset,
            "coco_split": args.coco_split,
            "timesteps": args.timesteps,
            "image_size": args.image_size,
            "tokenizer_max_length": tokenizer.max_length,
            "text_emb_dim": args.text_emb_dim,
            "text_hidden_dim": args.text_hidden_dim,
            "base_channels": args.base_channels,
            "global_step": global_step,
            "epoch": epoch,
            "run_timestamp": run_timestamp,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": train_losses + [train_loss],
            "val_losses": val_losses + [val_loss],
        }
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))

        sampled = sample_ddpm(
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompts=preview_prompts,
            image_size=args.image_size,
            cfg_scale=args.preview_cfg_scale,
            device=device,
        )
        grid = make_grid(sampled, nrow=min(3, len(preview_prompts)))
        save_image(grid, sample_path)

        epoch_ids.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        plot_losses(epoch_ids, train_losses, val_losses, plot_path)

        with open(metrics_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_timestamp, f"{train_loss:.6f}", f"{val_loss:.6f}", sample_path, plot_path])

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6e} sample={sample_path} plot={plot_path}"
        )

        # In notebook environments (e.g., Colab), this displays the evolution graph and current sample.
        maybe_display_image(plot_path)
        maybe_display_image(sample_path)

    print(f"training finished, metrics logged at: {metrics_csv_path}")


if __name__ == "__main__":
    main()
