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
def build_text_conditioning(text_encoder, tokenizer, prompts, device: str, text_encoder_type: str = "tiny"):
    if text_encoder_type == "tiny":
        token_ids = torch.stack([tokenizer.encode(p) for p in prompts]).to(device)
        pooled, tokens, pad_mask = text_encoder(token_ids, return_sequence=True)
        return pooled, tokens, pad_mask

    if text_encoder_type == "hf":
        pooled, tokens, pad_mask = text_encoder.encode_texts(prompts, device=device)
        return pooled, tokens, pad_mask

    raise ValueError(f"Unsupported text_encoder_type: {text_encoder_type}. Use 'tiny' or 'hf'.")


@torch.no_grad()
def predict_eps(
    unet,
    x: torch.Tensor,
    t: torch.Tensor,
    cond_pooled: torch.Tensor,
    cond_tokens: torch.Tensor,
    cond_mask: torch.Tensor,
    uncond_pooled: torch.Tensor,
    uncond_tokens: torch.Tensor,
    uncond_mask: torch.Tensor,
    cfg_scale: float,
    prediction_target: str,
    scheduler: DiffusionScheduler,
):
    model_uncond = unet(x, t, uncond_pooled, uncond_tokens, uncond_mask)
    model_cond = unet(x, t, cond_pooled, cond_tokens, cond_mask)
    model_out = model_uncond + cfg_scale * (model_cond - model_uncond)

    a_hat_t = scheduler.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
    om_t = scheduler.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)

    if prediction_target == "epsilon":
        return model_out
    if prediction_target == "v":
        return a_hat_t * model_out + om_t * x
    raise ValueError(f"Unsupported prediction_target: {prediction_target}. Use 'epsilon' or 'v'.")


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
    text_encoder_type: str = "tiny",
    device: str = "cpu",
):
    unet.eval()
    text_encoder.eval()

    n = len(prompts)
    x = torch.randn(n, 3, image_size, image_size, device=device)

    cond = build_text_conditioning(text_encoder, tokenizer, prompts, device, text_encoder_type=text_encoder_type)
    uncond = build_text_conditioning(text_encoder, tokenizer, [""] * n, device, text_encoder_type=text_encoder_type)

    for i in reversed(range(scheduler.timesteps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = predict_eps(
            unet,
            x,
            t,
            cond_pooled=cond[0],
            cond_tokens=cond[1],
            cond_mask=cond[2],
            uncond_pooled=uncond[0],
            uncond_tokens=uncond[1],
            uncond_mask=uncond[2],
            cfg_scale=cfg_scale,
            prediction_target=prediction_target,
            scheduler=scheduler,
        )

        alpha = scheduler.alphas[i]
        alpha_hat = scheduler.alpha_hat[i]
        beta = scheduler.betas[i]

        z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = (1.0 / torch.sqrt(alpha)) * (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * eps) + torch.sqrt(beta) * z

    x = x.clamp(-1, 1)
    return (x + 1) / 2


@torch.no_grad()
def sample_ddim(
    unet,
    text_encoder,
    tokenizer,
    scheduler: DiffusionScheduler,
    prompts,
    image_size: int = 32,
    cfg_scale: float = 5.0,
    prediction_target: str = "epsilon",
    text_encoder_type: str = "tiny",
    steps: int = 50,
    eta: float = 0.0,
    device: str = "cpu",
):
    unet.eval()
    text_encoder.eval()

    n = len(prompts)
    x = torch.randn(n, 3, image_size, image_size, device=device)

    cond = build_text_conditioning(text_encoder, tokenizer, prompts, device, text_encoder_type=text_encoder_type)
    uncond = build_text_conditioning(text_encoder, tokenizer, [""] * n, device, text_encoder_type=text_encoder_type)

    steps = max(2, min(steps, scheduler.timesteps))
    t_seq = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()

    for idx in range(len(t_seq) - 1):
        t_cur = t_seq[idx]
        t_next = t_seq[idx + 1]
        t = torch.full((n,), int(t_cur.item()), device=device, dtype=torch.long)

        eps = predict_eps(
            unet,
            x,
            t,
            cond_pooled=cond[0],
            cond_tokens=cond[1],
            cond_mask=cond[2],
            uncond_pooled=uncond[0],
            uncond_tokens=uncond[1],
            uncond_mask=uncond[2],
            cfg_scale=cfg_scale,
            prediction_target=prediction_target,
            scheduler=scheduler,
        )

        alpha_t = scheduler.alpha_hat[t_cur]
        alpha_next = scheduler.alpha_hat[t_next]

        x0_pred = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

        sigma = (
            eta
            * torch.sqrt((1 - alpha_next) / (1 - alpha_t))
            * torch.sqrt(1 - alpha_t / alpha_next)
        )
        noise = torch.randn_like(x) if sigma.item() > 0 else torch.zeros_like(x)

        dir_xt = torch.sqrt(torch.clamp(1 - alpha_next - sigma**2, min=0.0)) * eps
        x = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise

    x = x.clamp(-1, 1)
    return (x + 1) / 2
