"""Priming pixel backbone with DiT-style timestep-conditioned adaLN-Zero.

Conditioning follows the equations in facebookresearch/DiT/models.py:
https://github.com/facebookresearch/DiT/blob/main/models.py
The Priming patch embedder, RoPE attention, SwiGLU, and class-prefix tokens
remain in use; this is not the released vanilla DiT-B architecture.
"""
import math

import torch
from torch import nn

from .layers import SwiGLUMlp, apply_rotary_pos_emb, apply_rotary_pos_emb_partial
from .priming import OST, RoPEAttention


def modulate(value, shift, scale):
    return value * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTTimestepEmbedder(nn.Module):
    """Continuous [0, 1] noise levels mapped to the DiT 0--1000 scale."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size),
                                 nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def sinusoidal_embedding(timesteps, dimensions, max_period=10000):
        half = dimensions // 2
        frequencies = torch.arange(half, device=timesteps.device, dtype=torch.float32)
        frequencies = torch.exp(frequencies * (-math.log(max_period) / half))
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if dimensions % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, noise_levels):
        frequencies = self.sinusoidal_embedding(noise_levels * 1000., self.frequency_embedding_size)
        return self.mlp(frequencies)


class AdaLNBlock(nn.Module):
    """Six-way shift/scale/gate modulation with zero-initialized residual gates."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=8/3,
                 weight_init_constant=.32, rope_func=apply_rotary_pos_emb):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads,
                                  weight_init_constant=weight_init_constant, rope_func=rope_func)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio),
                             weight_init="scaled_variance", weight_init_constant=weight_init_constant)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, sequence, rope_freqs, condition):
        attention_shift, attention_scale, attention_gate, mlp_shift, mlp_scale, mlp_gate = (
            self.adaLN_modulation(condition).chunk(6, dim=-1))
        attended = self.attn(modulate(self.norm1(sequence), attention_shift, attention_scale), rope_freqs)
        sequence = sequence + attention_gate[:, None, :] * attended
        transformed = self.mlp(modulate(self.norm2(sequence), mlp_shift, mlp_scale))
        return sequence + mlp_gate[:, None, :] * transformed


class AdaLNFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        for layer in (self.adaLN_modulation[-1], self.linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, sequence, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(sequence), shift, scale))


class AdaLNOST(OST):
    """B/16 Priming backbone retaining its image path and adding DiT conditioning."""

    timestep_conditioned = True

    def __init__(self, *, depth=16, hidden_size=768, num_heads=12, mlp_ratio=8/3, **kwargs):
        kwargs.setdefault('num_cfg_tokens', 0)
        kwargs.setdefault('num_interval_tokens', 0)
        if kwargs['num_cfg_tokens'] or kwargs['num_interval_tokens']:
            raise ValueError('The adaLN timestep ablation does not use CFG/interval tokens')
        # Construct the unchanged patch/label/position path without allocating
        # temporary original transformer blocks which would immediately be replaced.
        super().__init__(depth=0, hidden_size=hidden_size, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, **kwargs)
        self.depth = depth
        self.t_embedder = DiTTimestepEmbedder(hidden_size)
        rope_func = apply_rotary_pos_emb_partial if kwargs.get('rope_2d', True) else apply_rotary_pos_emb
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                       weight_init_constant=kwargs.get('weight_init_constant', .32), rope_func=rope_func)
            for _ in range(depth)
        ])
        self.final_layer = AdaLNFinalLayer(hidden_size, self.patch_size, self.out_channels)

    def forward(self, x, y, *, t):
        if t.numel() != x.shape[0]:
            raise ValueError('Exactly one timestep per image is required')
        noise_levels = t.reshape(x.shape[0])
        condition = self.t_embedder(noise_levels) + self.y_embedder(y)
        sequence = self._build_sequence(x, y)
        for block in self.blocks:
            sequence = block(sequence, self.rope_freqs, condition)
        patches = self.final_layer(sequence[:, self.prefix_tokens:], condition)
        return self.unpatchify(patches)
