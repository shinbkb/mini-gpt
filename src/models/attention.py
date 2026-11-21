"""Causal multi-head self-attention with RoPE."""
import torch
from torch import nn
from torch.nn import functional as F

from src.inference.kv_cache import KVCache
from src.models.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Self-attention. Uses PyTorch SDPA with its causal mask."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, *, cache: KVCache | None = None,
                layer_idx: int = 0, position_offset: int = 0) -> torch.Tensor:
        batch, sequence, channels = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        q = apply_rope(q, offset=position_offset)
        k = apply_rope(k, offset=position_offset)
        if cache is None:
            is_causal = True
        else:
            if sequence > 1 and cache.layers[layer_idx] is not None:
                raise ValueError("cached decode accepts one new token at a time after prefill")
            entry = cache.update(layer_idx, k, v)
            k, v = entry.key, entry.value
            # Cached calls are either a full causal prefill or single-token decode.
            # For prefill SDPA's causal mask applies; single-token decode sees all past K/V.
            is_causal = sequence > 1
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.out_proj(attended)
