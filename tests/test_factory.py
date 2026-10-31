import pytest
import torch
from transformers import Qwen2Config

from kvshrink.cache import StaticKVCache
from kvshrink.factory import make_cache
from kvshrink.quant_cache import QuantizedKVCache

CFG = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)


def test_full():
    c = make_cache({"type": "full"}, CFG, 2, 32, torch.float32)
    assert isinstance(c, StaticKVCache) and c.max_len == 32


def test_quant_int8():
    c = make_cache({"type": "quant", "key_bits": 8, "value_bits": 8}, CFG, 2, 32, torch.float32)
    assert isinstance(c, QuantizedKVCache)
    assert c.k[0].bits == 8 and c.k[0].mode == "token" and c.k[0].group == 16


def test_kivi():
    c = make_cache({"type": "kivi", "bits": 2, "group_size": 8, "residual": 16}, CFG, 2, 64, torch.float32)
    assert c.k[0].mode == "channel" and c.v[0].mode == "token"
    assert c.k[0].bits == c.v[0].bits == 2 and c.k[0].residual == 16 and c.k[0].group == 8


def test_unknown_raises():
    with pytest.raises(ValueError):
        make_cache({"type": "nope"}, CFG, 1, 8)
