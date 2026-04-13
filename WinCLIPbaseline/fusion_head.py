import torch
from torch import nn


class FusionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, scores: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual_scores = scores
        target_dtype = self.fc1.weight.dtype
        if scores.dtype != target_dtype:
            scores = scores.to(target_dtype)

        logits = self.norm(scores)
        logits = self.fc1(logits)
        logits = self.act(logits)
        logits = self.dropout(logits)
        logits = self.fc2(logits)

        if valid_mask is not None:
            valid_mask = valid_mask.to(dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~valid_mask, -1e4)

        weights = torch.softmax(logits, dim=-1)
        if valid_mask is not None:
            weights = weights * valid_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if residual_scores.dtype != weights.dtype:
            residual_scores = residual_scores.to(weights.dtype)
        fused_scores = (weights * residual_scores).sum(dim=-1)
        return fused_scores.to(dtype=scores.dtype)
