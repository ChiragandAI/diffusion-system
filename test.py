"""Quick smoke test for the scratch text-to-image diffusion components."""

import torch

from diffusion_scratch.diffusion import DiffusionScheduler
from diffusion_scratch.text_encoder import CharTokenizer, TinyTextEncoder
from diffusion_scratch.unet import TinyConditionalUNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CharTokenizer()

    text_encoder = TinyTextEncoder(vocab_size=tokenizer.vocab_size).to(device)
    unet = TinyConditionalUNet().to(device)
    scheduler = DiffusionScheduler(timesteps=100, device=device)

    batch_size, image_size = 4, 32
    x0 = torch.randn(batch_size, 3, image_size, image_size, device=device)
    prompts = ["a photo of a cat", "a photo of a dog", "a photo of a ship", "a photo of a truck"]
    token_ids = torch.stack([tokenizer.encode(p) for p in prompts]).to(device)
    text_cond = text_encoder(token_ids)

    t = scheduler.sample_timesteps(batch_size)
    x_t, noise = scheduler.add_noise(x0, t)
    pred_noise = unet(x_t, t, text_cond)

    print("Input image shape:", x0.shape)
    print("Noisy image shape:", x_t.shape)
    print("True noise shape:", noise.shape)
    print("Predicted noise shape:", pred_noise.shape)


if __name__ == "__main__":
    main()
