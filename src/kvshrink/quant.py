"""Asymmetric min-max integer quantization with grouping and bit packing.

A tensor is split into groups of `group_size` consecutive elements along `dim`.
Each group stores its min (zero point) and step size (scale); elements become
integer codes in [0, 2^bits - 1], packed 8 // bits codes per byte.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SUPPORTED_BITS = (2, 4, 8)


def pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 codes in [0, 2^bits) along the last dim, 8 // bits codes per byte."""
    per_byte = 8 // bits
    if q.shape[-1] % per_byte:
        raise ValueError(f"last dim {q.shape[-1]} not divisible by {per_byte}")
    q = q.reshape(*q.shape[:-1], q.shape[-1] // per_byte, per_byte)
    out = q[..., 0].clone()
    for i in range(1, per_byte):
        out |= q[..., i] << (i * bits)
    return out


def unpack(p: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of `pack`: uint8 bytes -> uint8 codes, last dim grows by 8 // bits."""
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    parts = [(p >> (i * bits)) & mask for i in range(per_byte)]
    return torch.stack(parts, dim=-1).flatten(-2)


@dataclass
class QuantizedTensor:
    """Packed codes plus per-group scale and zero point."""

    packed: torch.Tensor  # uint8, [..., n_groups, group_size * bits // 8]
    scale: torch.Tensor  # [..., n_groups, 1], original dtype
    zero: torch.Tensor  # [..., n_groups, 1], original dtype (group minimum)
    bits: int
    dim: int  # quantized axis in the original tensor
    group_size: int
    shape: tuple[int, ...]  # shape with `dim` moved last
    dtype: torch.dtype

    def memory_bytes(self) -> int:
        """Bytes held by codes, scales and zero points."""
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scale, self.zero))


def quantize(x: torch.Tensor, bits: int, group_size: int, dim: int = -1) -> QuantizedTensor:
    """Quantize `x` in groups of `group_size` along `dim`.

    Per-element error is at most half a quantization step of its group
    (plus rounding of the stored scale/zero to `x.dtype`).
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    dim = dim % x.ndim
    xt = x.movedim(dim, -1)
    n = xt.shape[-1]
    if n % group_size:
        raise ValueError(f"size {n} along dim {dim} not divisible by group_size {group_size}")
    if group_size % (8 // bits):
        raise ValueError(f"group_size {group_size} must be divisible by {8 // bits} for {bits}-bit packing")
    g = xt.reshape(*xt.shape[:-1], n // group_size, group_size).float()
    levels = 2**bits - 1
    mn = g.amin(-1, keepdim=True)
    mx = g.amax(-1, keepdim=True)
    scale = (mx - mn) / levels
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # constant group -> all codes 0
    # Round scale/zero to the storage dtype first so codes are consistent with what dequant sees.
    scale = scale.to(x.dtype).float()
    mn = mn.to(x.dtype).float()
    q = torch.round((g - mn) / scale).clamp_(0, levels).to(torch.uint8)
    return QuantizedTensor(
        packed=pack(q, bits),
        scale=scale.to(x.dtype),
        zero=mn.to(x.dtype),
        bits=bits,
        dim=dim,
        group_size=group_size,
        shape=tuple(xt.shape),
        dtype=x.dtype,
    )


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct a dense tensor with the original shape and dtype."""
    q = unpack(qt.packed, qt.bits).float()
    g = q * qt.scale.float() + qt.zero.float()
    return g.reshape(qt.shape).to(qt.dtype).movedim(-1, qt.dim)
