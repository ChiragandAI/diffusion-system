import argparse
import os
from contextlib import nullcontext
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
    p.add_argument("--cfg_scale", type=float, default=6.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--channels_last", action="store_true", default=True)
    p.add_argument("--no-channels_last", action="store_false", dest="channels_last")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)

    tokenizer = CharTokenizer(max_length=ckpt.get("tokenizer_max_length", 48))
    text_emb_dim = ckpt.get("text_emb_dim", 192)
    text_hidden_dim = ckpt.get("text_hidden_dim", 384)
    base_channels = ckpt.get("base_channels", 96)

    text_encoder = TinyTextEncoder(
        vocab_size=tokenizer.vocab_size,
        emb_dim=text_emb_dim,
        hidden_dim=text_hidden_dim,
    ).to(device)
    unet = TinyConditionalUNet(
        in_channels=3,
        base_channels=base_channels,
        cond_dim=text_hidden_dim,
        time_dim=text_hidden_dim,
    ).to(device)
    if args.channels_last and device.startswith("cuda"):
        unet = unet.to(memory_format=torch.channels_last)

    text_encoder.load_state_dict(ckpt["text_encoder"])
    unet.load_state_dict(ckpt["unet"])

    scheduler = DiffusionScheduler(timesteps=ckpt.get("timesteps", 300), device=device)
    image_size = ckpt.get("image_size", 32)

    use_amp = args.amp and device.startswith("cuda")
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else nullcontext()
    with amp_ctx:
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
