import math
import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class TinyConditionalUNet(nn.Module):
    """Small text-conditioned U-Net that predicts noise epsilon."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, cond_dim: int = 256, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels, cond_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)

        self.down2 = ResBlock(base_channels * 2, base_channels * 2, cond_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)

        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, cond_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, cond_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.res_up2 = ResBlock(base_channels * 4, base_channels * 2, cond_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.res_up1 = ResBlock(base_channels * 2, base_channels, cond_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_dim)
        cond = text_cond + self.time_mlp(t_emb)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, cond)

        x = self.downsample1(d1)
        d2 = self.down2(x, cond)

        x = self.downsample2(d2)
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.res_up2(x, cond)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.res_up1(x, cond)

        return self.out(x)
