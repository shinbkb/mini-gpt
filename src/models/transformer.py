"""Pre-normalized decoder-only Transformer block."""
from torch import nn

from src.models.attention import CausalSelfAttention
from src.inference.kv_cache import KVCache
from src.models.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """A causal attention and GELU feed-forward block with residual paths."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, *, cache: KVCache | None = None,
                layer_idx: int = 0, position_offset: int = 0):
        x = x + self.attention(self.attn_norm(x), cache=cache,
                               layer_idx=layer_idx, position_offset=position_offset)
        return x + self.feed_forward(self.ffn_norm(x))
