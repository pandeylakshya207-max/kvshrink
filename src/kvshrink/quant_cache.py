"""Quantized KV caches: 8-bit per-token and KIVI-style (per-channel keys, per-token values).

Each K or V stream of a layer has two parts:
  * a quantized store for older tokens (preallocated packed codes + scale/zero),
  * a full-precision window holding the most recent tokens.
New tokens enter the window; whenever it holds at least `residual + block` tokens,
the oldest whole blocks are quantized and moved to the store. `block` is
`group_size` tokens for per-channel quantization and 1 token for per-token.
"""
from __future__ import annotations

import torch

from kvshrink.cache import KVCache
from kvshrink.quant import QuantizedTensor, dequantize, quantize

MODES = ("token", "channel")


class _QuantStream:
    """Quantized store + full-precision window for one tensor (K or V) of one layer."""

    def __init__(
        self, batch: int, heads: int, head_dim: int, max_len: int, bits: int, mode: str,
        group_size: int, residual: int, dtype: torch.dtype, device: torch.device | str,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode}")
        if mode == "token" and head_dim % group_size:
            raise ValueError(f"head_dim {head_dim} not divisible by group_size {group_size}")
        if residual < 0:
            raise ValueError("residual must be >= 0")
        self.B, self.H, self.D = batch, heads, head_dim
        self.max_len, self.bits, self.mode = max_len, bits, mode
        self.group, self.residual, self.dtype = group_size, residual, dtype
        self.block = group_size if mode == "channel" else 1
        bytes_per_group = group_size * bits // 8
        if mode == "token":
            shape = (batch, heads, max_len, head_dim // group_size)
        else:
            shape = (batch, heads, head_dim, max_len // group_size)
        self.packed = torch.zeros(*shape, bytes_per_group, dtype=torch.uint8, device=device)
        self.scale = torch.zeros(*shape, 1, dtype=dtype, device=device)
        self.zero = torch.zeros(*shape, 1, dtype=dtype, device=device)
        self.n_quant = 0
        self.window = torch.zeros(batch, heads, 0, head_dim, dtype=dtype, device=device)

    def __len__(self) -> int:
        return self.n_quant + self.window.shape[2]

    def append(self, x: torch.Tensor) -> None:
        w = torch.cat([self.window, x.to(self.dtype)], dim=2)
        n_flush = max(0, w.shape[2] - self.residual) // self.block * self.block
        if n_flush:
            self._store(w[:, :, :n_flush])
        self.window = w[:, :, n_flush:].clone()  # clone so the large concat buffer is freed

    def _slices(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "token":
            return self.packed[:, :, :n], self.scale[:, :, :n], self.zero[:, :, :n]
        nb = n // self.group
        return self.packed[:, :, :, :nb], self.scale[:, :, :, :nb], self.zero[:, :, :, :nb]

    def _store(self, x: torch.Tensor) -> None:
        n, s = x.shape[2], self.n_quant
        if s + n > self.max_len:
            raise ValueError(f"quantized store overflow: {s + n} > {self.max_len}")
        if self.mode == "token":
            qt = quantize(x, self.bits, self.group, dim=-1)
            dst = (slice(None), slice(None), slice(s, s + n))
        else:
            qt = quantize(x, self.bits, self.group, dim=-2)
            b0 = s // self.group
            dst = (slice(None), slice(None), slice(None), slice(b0, b0 + n // self.group))
        self.packed[dst] = qt.packed
        self.scale[dst] = qt.scale
        self.zero[dst] = qt.zero
        self.n_quant += n

    def dense(self) -> torch.Tensor:
        """All stored tokens as a dense [B, H, T, D] tensor (dequantized + window)."""
        n = self.n_quant
        if n == 0:
            return self.window
        packed, scale, zero = self._slices(n)
        if self.mode == "token":
            qt = QuantizedTensor(packed, scale, zero, self.bits, 3, self.group,
                                 (self.B, self.H, n, self.D), self.dtype)
        else:
            qt = QuantizedTensor(packed, scale, zero, self.bits, 2, self.group,
                                 (self.B, self.H, self.D, n), self.dtype)
        return torch.cat([dequantize(qt), self.window], dim=2)

    def allocated_bytes(self) -> int:
        bufs = (self.packed, self.scale, self.zero, self.window)
        return sum(t.numel() * t.element_size() for t in bufs)

    def used_bytes(self) -> int:
        bufs = (*self._slices(self.n_quant), self.window)
        return sum(t.numel() * t.element_size() for t in bufs)


class QuantizedKVCache(KVCache):
    """KV cache whose older tokens are stored as packed low-bit integers."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        *,
        key_bits: int,
        value_bits: int,
        key_mode: str = "token",
        value_mode: str = "token",
        key_group: int | None = None,
        value_group: int | None = None,
        residual: int = 0,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        key_group = key_group or head_dim
        value_group = value_group or head_dim
        args = (batch_size, num_kv_heads, head_dim, max_len)
        self.k = [_QuantStream(*args, key_bits, key_mode, key_group, residual, dtype, device)
                  for _ in range(num_layers)]
        self.v = [_QuantStream(*args, value_bits, value_mode, value_group, residual, dtype, device)
                  for _ in range(num_layers)]
        self.num_layers = num_layers
        self.max_len = max_len

    @classmethod
    def from_config(
        cls, config, batch_size: int, max_len: int, dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu", **quant_kwargs,
    ) -> "QuantizedKVCache":
        """Build a cache sized for a Hugging Face model config."""
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(config.num_hidden_layers, batch_size, config.num_key_value_heads, head_dim,
                   max_len, dtype=dtype, device=device, **quant_kwargs)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.k[layer]) + k.shape[2] > self.max_len:
            raise ValueError(f"cache overflow: {len(self.k[layer]) + k.shape[2]} > {self.max_len}")
        # This step attends to past tokens as stored (compressed) and to its own new
        # tokens in full precision; the new tokens are compressed for later steps.
        kd, vd = self.k[layer].dense(), self.v[layer].dense()
        self.k[layer].append(k)
        self.v[layer].append(v)
        return torch.cat([kd, k.to(kd.dtype)], dim=2), torch.cat([vd, v.to(vd.dtype)], dim=2)

    def seq_len(self, layer: int = 0) -> int:
        return len(self.k[layer])

    def memory_bytes(self) -> int:
        return sum(s.allocated_bytes() for s in self.k + self.v)

    def used_bytes(self) -> int:
        return sum(s.used_bytes() for s in self.k + self.v)


def int8_per_token(num_layers: int, batch_size: int, num_kv_heads: int, head_dim: int,
                   max_len: int, **kw) -> QuantizedKVCache:
    """8-bit per-token keys and values, one group per head vector, no fp window."""
    return QuantizedKVCache(num_layers, batch_size, num_kv_heads, head_dim, max_len,
                            key_bits=8, value_bits=8, key_mode="token", value_mode="token", **kw)


def kivi(num_layers: int, batch_size: int, num_kv_heads: int, head_dim: int, max_len: int,
         bits: int, group_size: int = 32, residual: int = 128, **kw) -> QuantizedKVCache:
    """KIVI-style: per-channel keys, per-token values, recent `residual` tokens in fp."""
    return QuantizedKVCache(num_layers, batch_size, num_kv_heads, head_dim, max_len,
                            key_bits=bits, value_bits=bits, key_mode="channel", value_mode="token",
                            key_group=group_size, value_group=group_size, residual=residual, **kw)
