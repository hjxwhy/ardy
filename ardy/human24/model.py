"""Finite scalar quantized autoencoder for the 218-channel Human24 window."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .stable_fsq import MotionGrid


@dataclass(frozen=True)
class TokenizerConfig:
    patch: int = 4
    width: int = 256
    heads: int = 4
    layers: int = 4
    code_dim: int = 128
    levels: int = 64
    paper_backbone: bool = False
    motion_grid: bool = False
    grid_position_weight: float = 1.0
    fp32_position_encoding: bool = False  # Legacy checkpoints added positions in the projection dtype.


class CausalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg.width
        self.heads = cfg.heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4),
                                nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x):
        b, t, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, width))
        return x + self.ff(x)


class PostNormBlock(nn.Module):
    """Causal encoder layer with the post-norm, 1024-wide FFN of the released tokenizer."""
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, 1024), nn.GELU(), nn.Linear(1024, config.width))

    def forward(self, x):
        b, t, width = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = self.norm1(x + self.proj(attn.transpose(1, 2).reshape(b, t, width)))
        return self.norm2(h + self.ff(h))


class Human24FSQVAE(nn.Module):
    def __init__(self, config=TokenizerConfig()):
        super().__init__()
        self.config = config
        if config.levels < 2 or config.levels % 2 or config.width % config.heads or config.width % 2:
            raise ValueError("Expected even FSQ levels and even width divisible by heads")
        self.register_buffer("mean", torch.zeros(218))
        self.register_buffer("std", torch.ones(218))
        indices = torch.arange(0, config.width, 2, dtype=torch.float32)
        if config.fp32_position_encoding:
            frequencies = torch.pow(10000.0, -indices / config.width)
        else:
            frequencies = torch.exp(indices * (-math.log(10000.0) / config.width))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.patch_in = nn.Linear(218 * config.patch, config.width)

        def stack(out_dim):
            block = PostNormBlock if config.paper_backbone else CausalBlock
            layers = [block(config) for _ in range(config.layers)]
            if not config.paper_backbone:
                layers.append(nn.LayerNorm(config.width))
            return nn.Sequential(*layers, nn.Linear(config.width, out_dim))

        self.encoder = stack(config.code_dim)
        self.code_in = nn.Linear(config.code_dim, config.width)
        self.decoder = stack(config.patch * 218)
        self.motion_grid = (MotionGrid(config.patch, config.code_dim, config.levels, config.grid_position_weight)
                            if config.motion_grid else None)
        if self.motion_grid is not None:
            for output in (self.encoder[-1], self.decoder[-1]):
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)

    def add_time(self, h):
        phase = torch.arange(h.shape[1], device=h.device)[:, None] * self.frequencies
        positions = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        if self.config.fp32_position_encoding:
            return h + positions
        return h + positions.to(h.dtype)

    def encode(self, x):
        """Return quantized codes [B, ceil(L/patch), code_dim] from raw features."""
        if x.ndim != 3 or x.shape[-1] != 218 or x.shape[1] < 1:
            raise ValueError("Expected nonempty [B, L, 218] input")
        x = (x - self.mean) / self.std
        pad = -x.shape[1] % self.config.patch
        if pad:
            x = torch.cat((x, x[:, -1:].expand(-1, pad, -1)), dim=1)
        patches = x.reshape(x.shape[0], -1, self.config.patch * 218)
        z = self.encoder(self.add_time(self.patch_in(patches))).float()
        if self.motion_grid is not None:
            return self.motion_grid.encode(patches, z)
        # Even-level FSQ grid [-half, half - 1], centered so zero maps to zero.
        half = self.config.levels // 2
        radius = (self.config.levels - 1) * 1.001 / 2
        bounded = (z + math.atanh(0.5 / radius)).tanh() * radius - 0.5
        rounded = bounded.round().clamp(-half, half - 1)
        return (bounded + (rounded - bounded).detach()) / half

    def decode_raw(self, codes):
        h = self.decoder(self.add_time(self.code_in(codes)))
        raw = h.reshape(h.shape[0], -1, 218).float()
        if self.motion_grid is not None:
            return self.motion_grid.decode(codes, raw, self.mean, self.std)
        return raw

    def unpack(self, raw):
        """Convert decoder channels to physical units and four contact probabilities."""
        x = raw * self.std + self.mean
        return torch.cat((x[..., :213], raw[..., 213:217].sigmoid(), x[..., 217:]), -1)

    def decode(self, codes, length=None):
        raw = self.decode_raw(codes)
        if length is not None and not 0 < length <= raw.shape[1]:
            raise ValueError("length must fit the supplied codes")
        return self.unpack(raw[:, :length])

    def forward(self, x):
        return self.decode(self.encode(x), length=x.shape[1])

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(TokenizerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        return model.to(device).eval()
