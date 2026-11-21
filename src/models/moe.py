"""Top-2 Mixture-of-Experts feed-forward layer, implemented with PyTorch."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Top2MoE(nn.Module):
    """Token-wise top-2 expert routing with Switch-style load balancing.

    Auxiliary loss: ``n_experts * sum_i(f_i * P_i)``, where ``f_i`` is the
    fraction of selected routes assigned to expert ``i`` and ``P_i`` is the
    mean router probability for that expert.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4,
                 top_k: int = 2, dropout: float = 0.0,
                 auxiliary_loss_coefficient: float = 0.01) -> None:
        super().__init__()
        if min(d_model, d_ff, n_experts, top_k) <= 0 or top_k > n_experts:
            raise ValueError("dimensions must be positive and top_k <= n_experts")
        if auxiliary_loss_coefficient < 0:
            raise ValueError("auxiliary_loss_coefficient must be nonnegative")
        self.d_model, self.n_experts, self.top_k = d_model, n_experts, top_k
        self.auxiliary_loss_coefficient = auxiliary_loss_coefficient
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.GELU(),
                          nn.Linear(d_ff, d_model, bias=False), nn.Dropout(dropout))
            for _ in range(n_experts)
        ])
        self.last_routing_stats: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("input must have shape (..., sequence, d_model)")
        original_shape = x.shape
        flat = x.reshape(-1, self.d_model)
        if flat.shape[0] == 0:
            raise ValueError("input must contain at least one token")
        probabilities = F.softmax(self.router(flat).float(), dim=-1)
        top_probabilities, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_weights = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Expert Linear layers may autocast to fp16/bf16 while router math is
        # deliberately fp32. Aggregate in the activation dtype for index_add_.
        output = torch.zeros_like(flat, dtype=torch.get_autocast_dtype(flat.device.type)
                                  if torch.is_autocast_enabled(flat.device.type) else flat.dtype)
        route_counts = torch.bincount(top_indices.reshape(-1), minlength=self.n_experts)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, route_slot = torch.where(top_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_idx))
            weighted = expert_output * top_weights[token_idx, route_slot].to(expert_output.dtype).unsqueeze(-1)
            output.index_add_(0, token_idx, weighted)
        assignment_fraction = route_counts.to(probabilities.dtype) / (flat.shape[0] * self.top_k)
        mean_probability = probabilities.mean(dim=0)
        auxiliary_loss = self.auxiliary_loss_coefficient * self.n_experts * torch.sum(
            assignment_fraction.detach() * mean_probability)
        with torch.no_grad():
            self.last_routing_stats = {
                "token_counts": route_counts.detach(),
                "expert_utilization": (route_counts > 0).float().mean(),
                "routing_distribution": assignment_fraction.detach(),
                "router_entropy": -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean(),
            }
        return output.reshape(original_shape), auxiliary_loss
