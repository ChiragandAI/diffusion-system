import argparse
import os
from contextlib import nullcontext
from datetime import datetime
import torch
from torchvision.utils import save_image, make_grid

from diffusion_scratch.diffusion import DiffusionScheduler, sample_ddpm
from diffusion_scratch.text_encoder import CharTokenizer, TinyTextEncoder
from diffusion_scratch.unet import TinyConditionalUNet


def parse_args():
    p = argparse.ArgumentParser(description="Generate images from a trained tiny text-to-image diffusion model.")
    p.add_argument("--checkpoint", type=str, default="./outputs/last.pt")
    p.add_argument("--prompts", type=str, nargs="+", required=True)
    p.add_argument("--output", type=str, default=None, help="If omitted, saves inside checkpoint run folder.")
    p.add_argument("--cfg_scale", type=float, default=6.0)
    p.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights from checkpoint if available.")
    p.add_argument("--no-use_ema", action="store_false", dest="use_ema")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--channels_last", action="store_true", default=True)
    p.add_argument("--no-channels_last", action="store_false", dest="channels_last")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def get_autocast_ctx(device: str, use_amp: bool):
    if not (use_amp and device.startswith("cuda")):
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def main():
    args = parse_args()
    device = args.device
    checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        latest_run_file = os.path.join("./outputs", "latest_run.txt")
        if os.path.exists(latest_run_file):
            with open(latest_run_file, "r") as f:
                latest_run = f.readline().strip()
            candidate = os.path.join(latest_run, "last.pt")
            if os.path.exists(candidate):
                checkpoint_path = candidate

    ckpt = torch.load(checkpoint_path, map_location=device)

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
    if args.use_ema and "ema_text_encoder" in ckpt and "ema_unet" in ckpt:
        text_encoder.load_state_dict(ckpt["ema_text_encoder"])
        unet.load_state_dict(ckpt["ema_unet"])

    scheduler = DiffusionScheduler(
        timesteps=ckpt.get("timesteps", 300),
        beta_schedule=ckpt.get("beta_schedule", "linear"),
        device=device,
    )
    image_size = ckpt.get("image_size", 32)
    prediction_target = ckpt.get("prediction_target", "epsilon")

    use_amp = args.amp and device.startswith("cuda")
    amp_ctx = get_autocast_ctx(device, use_amp)
    with amp_ctx:
        images = sample_ddpm(
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompts=args.prompts,
            image_size=image_size,
            cfg_scale=args.cfg_scale,
            prediction_target=prediction_target,
            device=device,
        )

    if args.output is None:
        ckpt_dir = os.path.dirname(checkpoint_path) or "."
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(ckpt_dir, f"generated_{ts}.png")
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid = make_grid(images, nrow=min(4, len(args.prompts)))
    save_image(grid, output_path)
    print(f"saved generated image grid to: {output_path}")


if __name__ == "__main__":
    main()
