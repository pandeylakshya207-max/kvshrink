"""Fill any kvshrink cache to a given length with synthetic (zero) contents.

Used by capacity benchmarks: memory and compute of a decode step depend on how many
tokens each cache holds, not on their values, so the cache state after `n_tokens`
of prefill is reproduced directly instead of running a real prefill for hundreds
of sequences. Positions continue from `n_tokens` (tokens_seen == n_tokens).
"""
from __future__ import annotations

import torch

from kvshrink.cache import KVCache, StaticKVCache
from kvshrink.evict_cache import H2OCache, StreamingLLMCache
from kvshrink.quant_cache import QuantizedKVCache


def fill_synthetic(cache: KVCache, n_tokens: int) -> None:
    """Put `cache` in the state it would have after `n_tokens` tokens of prefill."""
    if isinstance(cache, StaticKVCache):
        if n_tokens > cache.max_len:
            raise ValueError(f"{n_tokens} > max_len {cache.max_len}")
        cache.lengths = [n_tokens] * cache.num_layers
    elif isinstance(cache, QuantizedKVCache):
        if n_tokens > cache.max_len:
            raise ValueError(f"{n_tokens} > max_len {cache.max_len}")
        for s in cache.k + cache.v:
            nq = max(0, n_tokens - s.residual) // s.block * s.block
            s.n_quant = nq
            s.window = torch.zeros(s.B, s.H, n_tokens - nq, s.D, dtype=s.dtype, device=s.packed.device)
    elif isinstance(cache, (StreamingLLMCache, H2OCache)):
        keep = min(n_tokens, cache.budget)
        for i in range(cache.num_layers):
            B, H, _, D = cache.k[i].shape
            dev = cache.k[i].device
            cache.k[i] = torch.zeros(B, H, keep, D, dtype=cache.dtype, device=dev)
            cache.v[i] = torch.zeros(B, H, keep, D, dtype=cache.dtype, device=dev)
            cache.seen[i] = n_tokens
            if isinstance(cache, H2OCache):
                cache.score[i] = torch.zeros(B, H, keep, dtype=torch.float32, device=dev)
    else:
        raise TypeError(f"unsupported cache type {type(cache).__name__}")
