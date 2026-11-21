"""Per-layer key/value cache for autoregressive decoding."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKV:
    """K/V tensors shaped ``(batch, heads, sequence, head_dim)``."""
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    """Mutable cache with one entry per transformer attention layer."""
    def __init__(self, n_layers: int) -> None:
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.layers: list[LayerKV | None] = [None] * n_layers

    def __len__(self) -> int:
        lengths = {entry.key.shape[-2] for entry in self.layers if entry is not None}
        if len(lengths) > 1:
            raise RuntimeError("KV cache layers have inconsistent sequence lengths")
        return lengths.pop() if lengths else 0

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> LayerKV:
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError("layer_idx outside cache")
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("key and value must have matching (batch, heads, sequence, head_dim) shapes")
        previous = self.layers[layer_idx]
        if previous is not None:
            if previous.key.shape[:2] != key.shape[:2] or previous.key.shape[-1] != key.shape[-1]:
                raise ValueError("new K/V tensors are incompatible with cached batch/head dimensions")
            if previous.key.device != key.device or previous.key.dtype != key.dtype:
                raise ValueError("new K/V tensors must match cached device and dtype")
            key = torch.cat((previous.key, key), dim=-2)
            value = torch.cat((previous.value, value), dim=-2)
        entry = LayerKV(key, value)
        self.layers[layer_idx] = entry
        return entry

    def clear(self) -> None:
        self.layers = [None] * len(self.layers)
