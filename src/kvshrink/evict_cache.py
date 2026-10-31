"""Token-eviction KV caches with a fixed per-layer token budget.

StreamingLLM (Xiao et al., 2023): keep the first `sink` tokens ("attention sinks")
plus the most recent `budget - sink` tokens; evict everything in between.

Keys keep the RoPE position they were written with (no re-indexing of positions
inside the cache). This is valid while the total sequence stays within the
model's trained context length (32k for Qwen2.5), which covers every experiment here.
"""
from __future__ import annotations

import torch

from kvshrink.cache import KVCache


class StreamingLLMCache(KVCache):
    """Attention sinks + recent window, at most `budget` tokens per layer."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        budget: int,
        sink: int = 4,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0 <= sink < budget:
            raise ValueError(f"need 0 <= sink < budget, got sink={sink}, budget={budget}")
        empty = (batch_size, num_kv_heads, 0, head_dim)
        self.k = [torch.zeros(empty, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(empty, dtype=dtype, device=device) for _ in range(num_layers)]
        self.seen = [0] * num_layers
        self.budget, self.sink, self.dtype = budget, sink, dtype
        self.num_layers = num_layers

    @classmethod
    def from_config(
        cls, config, batch_size: int, budget: int, sink: int = 4,
        dtype: torch.dtype = torch.float16, device: torch.device | str = "cpu",
    ) -> "StreamingLLMCache":
        """Build a cache sized for a Hugging Face model config."""
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(config.num_hidden_layers, batch_size, config.num_key_value_heads, head_dim,
                   budget, sink, dtype, device)

    def _evict(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2]
        if n <= self.budget:
            return x
        recent = self.budget - self.sink
        return torch.cat([x[:, :, : self.sink], x[:, :, n - recent:]], dim=2)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # This step attends to everything currently kept plus its own new tokens;
        # eviction happens afterwards, for later steps.
        K = torch.cat([self.k[layer], k.to(self.dtype)], dim=2)
        V = torch.cat([self.v[layer], v.to(self.dtype)], dim=2)
        self.seen[layer] += k.shape[2]
        self.k[layer], self.v[layer] = self._evict(K), self._evict(V)
        return K, V

    def seq_len(self, layer: int = 0) -> int:
        return self.k[layer].shape[2]

    def tokens_seen(self, layer: int = 0) -> int:
        return self.seen[layer]

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def used_bytes(self) -> int:
        return self.memory_bytes()
