import argparse
import os
import torch
from torchvision.utils import save_image, make_grid

from diffusion_scratch.diffusion import DiffusionScheduler, sample_ddpm
from diffusion_scratch.text_encoder import CharTokenizer, TinyTextEncoder
from diffusion_scratch.unet import TinyConditionalUNet


def parse_args():
    p = argparse.ArgumentParser(description="Generate images from a trained tiny text-to-image diffusion model.")
    p.add_argument("--checkpoint", type=str, default="./outputs/last.pt")
    p.add_argument("--prompts", type=str, nargs="+", required=True)
    p.add_argument("--output", type=str, default="./outputs/generated.png")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)

    tokenizer = CharTokenizer(max_length=ckpt.get("tokenizer_max_length", 48))
    text_encoder = TinyTextEncoder(vocab_size=tokenizer.vocab_size, emb_dim=128, hidden_dim=256).to(device)
    unet = TinyConditionalUNet(in_channels=3, base_channels=64, cond_dim=256, time_dim=256).to(device)

    text_encoder.load_state_dict(ckpt["text_encoder"])
    unet.load_state_dict(ckpt["unet"])

    scheduler = DiffusionScheduler(timesteps=ckpt.get("timesteps", 300), device=device)
    image_size = ckpt.get("image_size", 32)

    images = sample_ddpm(
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        prompts=args.prompts,
        image_size=image_size,
        cfg_scale=args.cfg_scale,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    grid = make_grid(images, nrow=min(4, len(args.prompts)))
    save_image(grid, args.output)
    print(f"saved generated image grid to: {args.output}")


if __name__ == "__main__":
    main()
