"""Pre-normalized decoder-only Transformer block."""
from torch import nn

from src.models.attention import CausalSelfAttention
from src.inference.kv_cache import KVCache
from src.models.rmsnorm import RMSNorm
from src.models.moe import Top2MoE


class TransformerBlock(nn.Module):
    """A causal attention and GELU feed-forward block with residual paths."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, *, n_experts: int = 0, top_k: int = 2,
                 auxiliary_loss_coefficient: float = 0.01) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = (Top2MoE(d_model, d_ff, n_experts, top_k, dropout,
                                      auxiliary_loss_coefficient) if n_experts else nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        ))

    def forward(self, x, *, cache: KVCache | None = None,
                layer_idx: int = 0, position_offset: int = 0):
        x = x + self.attention(self.attn_norm(x), cache=cache,
                               layer_idx=layer_idx, position_offset=position_offset)
        ff_input = self.ffn_norm(x)
        if isinstance(self.feed_forward, Top2MoE):
            ff_output, auxiliary_loss = self.feed_forward(ff_input)
            result = x + ff_output
            return (result, auxiliary_loss) if self.training else (result, auxiliary_loss)
        return x + self.feed_forward(ff_input), x.new_zeros(())
