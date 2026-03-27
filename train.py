import argparse
import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from diffusion_scratch.data import build_text_dataset
from diffusion_scratch.diffusion import DiffusionScheduler, sample_ddpm
from diffusion_scratch.text_encoder import CharTokenizer, TinyTextEncoder
from diffusion_scratch.unet import TinyConditionalUNet


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny text-to-image diffusion model from scratch.")
    p.add_argument("--dataset", type=str, default="coco", choices=["coco", "stl10", "cifar10"])
    p.add_argument("--coco_split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--drop_text_prob", type=float, default=0.1)
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    tokenizer = CharTokenizer(max_length=48)

    dataset = build_text_dataset(
        name=args.dataset,
        root=args.data_dir,
        image_size=args.image_size,
        coco_split=args.coco_split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
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

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        unet.train()
        text_encoder.train()

        running_loss = 0.0
        for step, (images, captions) in enumerate(loader, start=1):
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
                print(f"epoch={epoch} step={step}/{len(loader)} loss={running_loss / step:.4f}")

        epoch_loss = running_loss / max(1, len(loader))
        print(f"[epoch {epoch}] mean loss: {epoch_loss:.4f}")

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
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))

        if epoch % args.sample_every == 0:
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
            save_image(grid, os.path.join(args.output_dir, f"sample_epoch_{epoch:03d}.png"))

    print("training finished")


if __name__ == "__main__":
    main()
