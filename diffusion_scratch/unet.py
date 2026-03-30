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


def _num_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class FiLMResBlock(nn.Module):
    """Residual block with FiLM conditioning for strong timestep/text pooled control."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, (2 * in_ch) + (2 * out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        film = self.cond_proj(cond)
        s1, b1, s2, b2 = torch.split(
            film,
            [self.conv1.in_channels, self.conv1.in_channels, self.conv2.in_channels, self.conv2.in_channels],
            dim=-1,
        )

        h = self.norm1(x)
        h = h * (1 + s1.unsqueeze(-1).unsqueeze(-1)) + b1.unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(self.act(h))

        h = self.norm2(h)
        h = h * (1 + s2.unsqueeze(-1).unsqueeze(-1)) + b2.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class SpatialCrossAttention(nn.Module):
    """Cross-attend image tokens to text tokens."""

    def __init__(self, img_channels: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(img_channels), img_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=img_channels,
            num_heads=num_heads,
            kdim=text_dim,
            vdim=text_dim,
            batch_first=True,
        )
        self.out_proj = nn.Linear(img_channels, img_channels)

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor, text_pad_mask: torch.Tensor = None) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x
        x = self.norm(x)
        q = x.view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.attn(q, text_tokens, text_tokens, key_padding_mask=text_pad_mask, need_weights=False)
        q = q + self.out_proj(attn_out)
        x = q.transpose(1, 2).view(b, c, h, w)
        return x + res


class SpatialSelfAttention(nn.Module):
    """Self-attention over spatial tokens at bottleneck."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x
        x = self.norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.proj(attn_out)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x + res


class TinyConditionalUNet(nn.Module):
    """Deeper text-conditioned U-Net with cross-attention."""

    def __init__(self, in_channels: int = 3, base_channels: int = 96, cond_dim: int = 384, time_dim: int = 384):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        ch1 = base_channels
        ch2 = base_channels * 2
        ch4 = base_channels * 4

        self.in_conv = nn.Conv2d(in_channels, ch1, 3, padding=1)

        self.down1a = FiLMResBlock(ch1, ch1, cond_dim)
        self.down1b = FiLMResBlock(ch1, ch1, cond_dim)
        self.cross1 = SpatialCrossAttention(ch1, text_dim=cond_dim, num_heads=4)
        self.downsample1 = nn.Conv2d(ch1, ch2, 4, stride=2, padding=1)

        self.down2a = FiLMResBlock(ch2, ch2, cond_dim)
        self.down2b = FiLMResBlock(ch2, ch2, cond_dim)
        self.cross2 = SpatialCrossAttention(ch2, text_dim=cond_dim, num_heads=4)
        self.downsample2 = nn.Conv2d(ch2, ch4, 4, stride=2, padding=1)

        self.mid1 = FiLMResBlock(ch4, ch4, cond_dim)
        self.mid_self_attn = SpatialSelfAttention(ch4, num_heads=4)
        self.mid_cross_attn = SpatialCrossAttention(ch4, text_dim=cond_dim, num_heads=4)
        self.mid2 = FiLMResBlock(ch4, ch4, cond_dim)

        self.up2 = nn.ConvTranspose2d(ch4, ch2, 4, stride=2, padding=1)
        self.res_up2a = FiLMResBlock(ch2 + ch2, ch2, cond_dim)
        self.res_up2b = FiLMResBlock(ch2, ch2, cond_dim)
        self.cross_up2 = SpatialCrossAttention(ch2, text_dim=cond_dim, num_heads=4)

        self.up1 = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.res_up1a = FiLMResBlock(ch1 + ch1, ch1, cond_dim)
        self.res_up1b = FiLMResBlock(ch1, ch1, cond_dim)
        self.cross_up1 = SpatialCrossAttention(ch1, text_dim=cond_dim, num_heads=4)

        self.out = nn.Sequential(
            nn.GroupNorm(_num_groups(ch1), ch1),
            nn.SiLU(),
            nn.Conv2d(ch1, in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_pooled: torch.Tensor,
        text_tokens: torch.Tensor = None,
        text_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_dim)
        cond = text_pooled + self.time_mlp(t_emb)

        if text_tokens is None:
            text_tokens = text_pooled.unsqueeze(1)
            text_pad_mask = None

        x = self.in_conv(x)

        d1 = self.down1a(x, cond)
        d1 = self.down1b(d1, cond)
        d1 = self.cross1(d1, text_tokens, text_pad_mask)

        x = self.downsample1(d1)
        d2 = self.down2a(x, cond)
        d2 = self.down2b(d2, cond)
        d2 = self.cross2(d2, text_tokens, text_pad_mask)

        x = self.downsample2(d2)
        x = self.mid1(x, cond)
        x = self.mid_self_attn(x)
        x = self.mid_cross_attn(x, text_tokens, text_pad_mask)
        x = self.mid2(x, cond)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.res_up2a(x, cond)
        x = self.res_up2b(x, cond)
        x = self.cross_up2(x, text_tokens, text_pad_mask)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.res_up1a(x, cond)
        x = self.res_up1b(x, cond)
        x = self.cross_up1(x, text_tokens, text_pad_mask)

        return self.out(x)
