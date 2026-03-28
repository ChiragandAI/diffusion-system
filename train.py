import argparse
import csv
import os
from datetime import datetime

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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--drop_text_prob", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def maybe_display_image(path: str):
    try:
        from IPython.display import Image, display

        display(Image(filename=path))
    except Exception:
        pass


@torch.no_grad()
def evaluate_val_loss(unet, text_encoder, tokenizer, scheduler, loader, device: str):
    unet.eval()
    text_encoder.eval()

    total_loss = 0.0
    num_steps = 0

    for images, captions in loader:
        images = images.to(device)
        t = scheduler.sample_timesteps(images.size(0))
        x_t, noise = scheduler.add_noise(images, t)

        token_ids = torch.stack([tokenizer.encode(c) for c in captions]).to(device)
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv_path = os.path.join(args.output_dir, f"metrics_{run_timestamp}.csv")

    device = args.device
    tokenizer = CharTokenizer(max_length=48)

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
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    text_encoder = TinyTextEncoder(vocab_size=tokenizer.vocab_size, emb_dim=128, hidden_dim=256).to(device)
    unet = TinyConditionalUNet(in_channels=3, base_channels=64, cond_dim=256, time_dim=256).to(device)
    scheduler = DiffusionScheduler(timesteps=args.timesteps, device=device)

    optimizer = optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=args.lr)

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

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "timestamp", "train_loss", "val_loss", "sample_path", "plot_path"])

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        unet.train()
        text_encoder.train()

        running_loss = 0.0
        for step, (images, captions) in enumerate(train_loader, start=1):
            images = images.to(device)
            t = scheduler.sample_timesteps(images.size(0))
            x_t, noise = scheduler.add_noise(images, t)

            token_ids = torch.stack([tokenizer.encode(c) for c in captions]).to(device)

            if args.drop_text_prob > 0:
                drop_mask = torch.rand(images.size(0), device=device) < args.drop_text_prob
                token_ids[drop_mask] = tokenizer.encode("").to(device)

            text_cond = text_encoder(token_ids)
            pred_noise = unet(x_t, t, text_cond)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={running_loss / step:.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = evaluate_val_loss(unet, text_encoder, tokenizer, scheduler, val_loader, device)

        epoch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = os.path.join(args.output_dir, f"sample_{run_timestamp}_epoch_{epoch:03d}_{epoch_timestamp}.png")
        plot_path = os.path.join(args.output_dir, f"loss_curve_{run_timestamp}_epoch_{epoch:03d}_{epoch_timestamp}.png")
        ckpt_path = os.path.join(args.output_dir, f"ckpt_{run_timestamp}_epoch_{epoch:03d}_{epoch_timestamp}.pt")

        ckpt = {
            "unet": unet.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "dataset": args.dataset,
            "coco_split": args.coco_split,
            "timesteps": args.timesteps,
            "image_size": args.image_size,
            "tokenizer_max_length": tokenizer.max_length,
            "global_step": global_step,
            "epoch": epoch,
            "run_timestamp": run_timestamp,
            "train_loss": train_loss,
            "val_loss": val_loss,
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
            cfg_scale=5.0,
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
            f"sample={sample_path} plot={plot_path}"
        )

        # In notebook environments (e.g., Colab), this displays the evolution graph and current sample.
        maybe_display_image(plot_path)
        maybe_display_image(sample_path)

    print(f"training finished, metrics logged at: {metrics_csv_path}")


if __name__ == "__main__":
    main()
