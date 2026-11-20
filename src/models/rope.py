"""Rotary positional embeddings for attention query and key tensors."""
import torch


def apply_rope(x: torch.Tensor, *, base: float = 10_000.0, offset: int = 0) -> torch.Tensor:
    """Apply RoPE to ``(batch, heads, sequence, head_dim)`` tensor.

    The last dimension must be even. ``offset`` is included for future cached
    decoding, though KV cache is not part of M2.
    """
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape (batch, heads, sequence, head_dim)")
    head_dim = x.shape[-1]
    if head_dim % 2 or base <= 0 or offset < 0:
        raise ValueError("head_dim must be even, base positive and offset nonnegative")
    positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
    frequencies = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim
    angles = positions[:, None] * (base ** -frequencies)[None, :]
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)
