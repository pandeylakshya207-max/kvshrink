"""Build any kvshrink cache from a config dict (one entry of `methods:` in a YAML file)."""
from __future__ import annotations

import torch

from kvshrink.cache import KVCache, StaticKVCache
from kvshrink.quant_cache import QuantizedKVCache


def make_cache(
    spec: dict, config, batch_size: int, max_len: int,
    dtype: torch.dtype = torch.float16, device: torch.device | str = "cpu",
) -> KVCache:
    """Create a cache from `spec`.

    Supported `type`s:
      full  - uncompressed StaticKVCache
      quant - QuantizedKVCache with explicit key/value bits, modes, groups, residual
      kivi  - per-channel keys, per-token values: bits, group_size, residual
    """
    kw = {k: v for k, v in spec.items() if k != "type"}
    kind = spec["type"]
    if kind == "full":
        return StaticKVCache.from_config(config, batch_size, max_len, dtype, device)
    if kind == "quant":
        return QuantizedKVCache.from_config(config, batch_size, max_len, dtype, device, **kw)
    if kind == "kivi":
        g = kw.get("group_size", 32)
        return QuantizedKVCache.from_config(
            config, batch_size, max_len, dtype, device,
            key_bits=kw["bits"], value_bits=kw["bits"], key_mode="channel", value_mode="token",
            key_group=g, value_group=g, residual=kw.get("residual", 128),
        )
    raise ValueError(f"unknown cache type: {kind}")
