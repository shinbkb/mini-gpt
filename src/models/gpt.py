"""Decoder-only Dense GPT language model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from src.inference.kv_cache import KVCache
from src.models.rmsnorm import RMSNorm
from src.models.transformer import TransformerBlock


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 8000
    block_size: int = 512
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    tie_weights: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layers, self.d_model,
               self.n_heads, self.d_ff) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_mapping(cls, config: dict) -> "GPTConfig":
        return cls(**{key: config[key] for key in asdict(cls()).keys() if key in config})


class GPT(nn.Module):
    """Autoregressive language model that returns logits for each input token."""
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, *, use_cache: bool = False,
                cache: KVCache | None = None) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[1] == 0 or input_ids.shape[1] > self.config.block_size:
            raise ValueError("input sequence length must be in [1, block_size]")
        if use_cache and cache is None:
            cache = KVCache(self.config.n_layers)
        if cache is not None and len(cache) + input_ids.shape[1] > self.config.block_size:
            raise ValueError("cached sequence length exceeds block_size")
        position_offset = len(cache) if cache is not None else 0
        x = self.embedding_dropout(self.token_embedding(input_ids))
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=layer_idx,
                      position_offset=position_offset)
        logits = self.lm_head(self.final_norm(x))
        return (logits, cache) if use_cache else logits

    def parameter_count(self) -> int:
        """Count unique trainable parameters (tied embedding/head counted once)."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, *,
                 temperature: float = 1.0, top_k: int | None = None,
                 eos_id: int | None = None, generator: torch.Generator | None = None,
                 use_cache: bool = True) -> torch.Tensor:
        """Sample autoregressively, optionally reusing per-layer attention K/V."""
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be nonnegative and temperature positive")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within vocabulary size")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain a nonempty prompt batch")
        self.eval()
        if input_ids.shape[1] > self.config.block_size:
            raise ValueError("prompt length cannot exceed block_size")
        output = input_ids
        cache = KVCache(self.config.n_layers) if use_cache else None
        cached_length = 0
        for _ in range(max_new_tokens):
            if use_cache:
                if cached_length >= self.config.block_size:
                    # Sliding-window generation changes RoPE positions back to zero.
                    cache.clear()
                    context = output[:, -self.config.block_size:]
                    logits = self(context, cache=cache)[:, -1, :] / temperature
                    cached_length = output.shape[1]
                else:
                    context = output[:, cached_length:]
                    logits = self(context, cache=cache)[:, -1, :] / temperature
                    cached_length = output.shape[1]
            else:
                context = output[:, -self.config.block_size:]
                logits = self(context)[:, -1, :] / temperature
            if top_k is not None:
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_token = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)
            output = torch.cat((output, next_token), dim=1)
            if eos_id is not None and torch.all(next_token == eos_id):
                break
        return output
