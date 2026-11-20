"""Root Mean Square Layer Normalization."""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize by root mean square with a learned scale and no bias."""
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dimension <= 0 or eps <= 0:
            raise ValueError("dimension and eps must be positive")
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight
