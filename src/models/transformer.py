"""Pre-normalized decoder-only Transformer block."""
from torch import nn

from src.models.attention import CausalSelfAttention
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

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))
