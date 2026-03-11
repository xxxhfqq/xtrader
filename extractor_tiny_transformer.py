from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_sdpa: bool) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能整除 n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.use_sdpa = bool(use_sdpa)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, Dh]

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal_mask = torch.tril(torch.ones((t, t), device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn_out = torch.matmul(attn, v)

        out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.proj(out)
        return self.proj_dropout(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float, use_sdpa: bool) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, use_sdpa=use_sdpa)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        *,
        window_size: int,
        market_feature_dim: int,
        account_feature_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
        pooling: str = "last",
        use_sdpa: bool = True,
        features_dim: int | None = None,
    ) -> None:
        out_dim = int(features_dim or d_model)
        super().__init__(observation_space, features_dim=out_dim)
        self.window_size = int(window_size)
        self.market_feature_dim = int(market_feature_dim)
        self.account_feature_dim = int(account_feature_dim)
        self.pooling = pooling

        self.seq_flat_dim = self.window_size * self.market_feature_dim
        expected_dim = self.seq_flat_dim + self.account_feature_dim
        obs_dim = int(observation_space.shape[0])
        if obs_dim != expected_dim:
            raise ValueError(
                f"obs 维度不匹配: obs_dim={obs_dim}, expected={expected_dim} "
                f"(window_size={self.window_size}, market_dim={self.market_feature_dim}, account_dim={self.account_feature_dim})"
            )

        self.input_proj = nn.Linear(self.market_feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.window_size, d_model))
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_sdpa=use_sdpa,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model + self.account_feature_dim, out_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        bsz = obs.shape[0]
        seq_flat = obs[:, : self.seq_flat_dim]
        account = obs[:, self.seq_flat_dim :]
        seq = seq_flat.view(bsz, self.window_size, self.market_feature_dim)

        x = self.input_proj(seq) + self.pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)

        if self.pooling == "mean":
            seq_repr = x.mean(dim=1)
        else:
            seq_repr = x[:, -1, :]

        feat = torch.cat([seq_repr, account], dim=-1)
        return self.out_proj(feat)

