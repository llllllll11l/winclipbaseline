import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(hidden_dim, embed_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        target_dtype = self.down_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        if x.dtype != residual.dtype:
            x = x.to(residual.dtype)
        residual_scale = self.residual_scale.to(x.dtype)
        return residual + residual_scale * x
