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


class H2OCache(KVCache):
    """Heavy-Hitter Oracle (Zhang et al., 2023): recent window + top accumulated-attention tokens.

    Each (sequence, KV head) keeps its own set of `budget` tokens: the `recent` most
    recent ones plus the `budget - recent` older ones with the largest accumulated
    attention mass. The runner computes explicit attention for caches with
    `needs_attention = True` and reports it through `observe()` after `update()`.
    """

    needs_attention = True

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        budget: int,
        recent: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0 <= recent <= budget or budget <= 0:
            raise ValueError(f"need 0 <= recent <= budget and budget > 0, got {recent}, {budget}")
        empty = (batch_size, num_kv_heads, 0, head_dim)
        self.k = [torch.zeros(empty, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(empty, dtype=dtype, device=device) for _ in range(num_layers)]
        self.score = [torch.zeros(empty[:3], dtype=torch.float32, device=device) for _ in range(num_layers)]
        self.seen = [0] * num_layers
        self.budget, self.recent, self.dtype = budget, recent, dtype
        self.num_layers = num_layers

    @classmethod
    def from_config(
        cls, config, batch_size: int, budget: int, recent: int,
        dtype: torch.dtype = torch.float16, device: torch.device | str = "cpu",
    ) -> "H2OCache":
        """Build a cache sized for a Hugging Face model config."""
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(config.num_hidden_layers, batch_size, config.num_key_value_heads, head_dim,
                   budget, recent, dtype, device)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.k[layer] = torch.cat([self.k[layer], k.to(self.dtype)], dim=2)
        self.v[layer] = torch.cat([self.v[layer], v.to(self.dtype)], dim=2)
        new = torch.zeros(k.shape[:3], dtype=torch.float32, device=k.device)
        self.score[layer] = torch.cat([self.score[layer], new], dim=2)
        self.seen[layer] += k.shape[2]
        return self.k[layer], self.v[layer]

    def observe(self, layer: int, attn: torch.Tensor) -> None:
        """Accumulate attention mass [B, H_kv, S] for the stored tokens, then evict to budget."""
        if attn.shape != self.score[layer].shape:
            raise ValueError(f"attn {tuple(attn.shape)} != stored {tuple(self.score[layer].shape)}")
        s = self.score[layer] + attn.float()
        n = s.shape[-1]
        if n <= self.budget:
            self.score[layer] = s
            return
        B, H = s.shape[:2]
        heavy = self.budget - self.recent
        top = s[..., : n - self.recent].topk(heavy, dim=-1).indices.sort(dim=-1).values
        tail = torch.arange(n - self.recent, n, device=s.device).expand(B, H, self.recent)
        idx = torch.cat([top, tail], dim=-1)
        gidx = idx[..., None].expand(-1, -1, -1, self.k[layer].shape[-1])
        self.k[layer] = self.k[layer].gather(2, gidx)
        self.v[layer] = self.v[layer].gather(2, gidx)
        self.score[layer] = s.gather(2, idx)

    def seq_len(self, layer: int = 0) -> int:
        return self.k[layer].shape[2]

    def tokens_seen(self, layer: int = 0) -> int:
        return self.seen[layer]

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v + self.score)

    def used_bytes(self) -> int:
        return self.memory_bytes()
