"""KV cache implementations.

All tensors use the layout [batch, num_kv_heads, seq_len, head_dim].
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class KVCache(ABC):
    """Interface shared by every cache (full precision, quantized, evicting)."""

    @abstractmethod
    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values for `layer` and return the full K, V for attention.

        Args:
            layer: decoder layer index.
            k, v: new entries, shape [B, H_kv, T_new, D].

        Returns:
            (K, V) as dense tensors in the model dtype, shape [B, H_kv, T_total, D].
        """

    @abstractmethod
    def seq_len(self, layer: int = 0) -> int:
        """Number of tokens currently stored for `layer`."""

    @abstractmethod
    def memory_bytes(self) -> int:
        """Bytes allocated for cache storage across all layers."""

    @abstractmethod
    def used_bytes(self) -> int:
        """Bytes occupied by tokens actually stored (excludes unused preallocation)."""


class StaticKVCache(KVCache):
    """Full-precision cache with storage preallocated up to `max_len` tokens."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        shape = (batch_size, num_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.lengths = [0] * num_layers
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.dtype = dtype

    @classmethod
    def from_config(
        cls,
        config,
        batch_size: int,
        max_len: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> "StaticKVCache":
        """Build a cache sized for a Hugging Face model config."""
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        return cls(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            head_dim,
            max_len,
            dtype,
            device,
        )

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.batch_size, self.num_kv_heads, self.head_dim)
        if k.shape != v.shape:
            raise ValueError(f"k {tuple(k.shape)} and v {tuple(v.shape)} differ")
        if (k.shape[0], k.shape[1], k.shape[3]) != expected:
            raise ValueError(f"expected [B,H,*,D]={expected}, got {tuple(k.shape)}")
        start = self.lengths[layer]
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"cache overflow: {end} > max_len {self.max_len}")
        self.k[layer][:, :, start:end] = k.to(self.dtype)
        self.v[layer][:, :, start:end] = v.to(self.dtype)
        self.lengths[layer] = end
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def seq_len(self, layer: int = 0) -> int:
        return self.lengths[layer]

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def used_bytes(self) -> int:
        per_token = 2 * self.batch_size * self.num_kv_heads * self.head_dim * self.k[0].element_size()
        return per_token * sum(self.lengths)
