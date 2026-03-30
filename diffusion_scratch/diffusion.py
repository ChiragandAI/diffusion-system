import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


class DiffusionScheduler:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "linear",
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps).to(device)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}. Use 'linear' or 'cosine'.")
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_hat = alpha_hat
        self.sqrt_alpha_hat = torch.sqrt(alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat)

    def to(self, device: str):
        self.device = device
        for name in [
            "betas",
            "alphas",
            "alpha_hat",
            "sqrt_alpha_hat",
            "sqrt_one_minus_alpha_hat",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise, noise


@torch.no_grad()
def sample_ddpm(
    unet,
    text_encoder,
    tokenizer,
    scheduler: DiffusionScheduler,
    prompts,
    image_size: int = 32,
    cfg_scale: float = 5.0,
    prediction_target: str = "epsilon",
    device: str = "cpu",
):
    unet.eval()
    text_encoder.eval()

    n = len(prompts)
    x = torch.randn(n, 3, image_size, image_size, device=device)

    token_ids = torch.stack([tokenizer.encode(p) for p in prompts]).to(device)
    null_ids = torch.stack([tokenizer.encode("") for _ in prompts]).to(device)

    cond = text_encoder(token_ids)
    uncond = text_encoder(null_ids)

    for i in reversed(range(scheduler.timesteps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        eps_uncond = unet(x, t, uncond)
        eps_cond = unet(x, t, cond)
        model_out = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        alpha = scheduler.alphas[i]
        alpha_hat = scheduler.alpha_hat[i]
        beta = scheduler.betas[i]
        sqrt_alpha_hat = scheduler.sqrt_alpha_hat[i]
        sqrt_one_minus_alpha_hat = scheduler.sqrt_one_minus_alpha_hat[i]

        if prediction_target == "epsilon":
            eps = model_out
        elif prediction_target == "v":
            eps = sqrt_alpha_hat * model_out + sqrt_one_minus_alpha_hat * x
        else:
            raise ValueError(f"Unsupported prediction_target: {prediction_target}. Use 'epsilon' or 'v'.")

        if i > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = (1.0 / torch.sqrt(alpha)) * (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * eps) + torch.sqrt(beta) * z

    x = x.clamp(-1, 1)
    return (x + 1) / 2
