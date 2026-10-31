import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvshrink.cache import StaticKVCache
from kvshrink.capacity import fill_synthetic
from kvshrink.factory import make_cache
from kvshrink.model import QwenRunner

L, B, H, D = 2, 2, 2, 8
SPECS = [
    {"type": "full"},
    {"type": "quant", "key_bits": 8, "value_bits": 8},
    {"type": "kivi", "bits": 2, "group_size": 4, "residual": 8},
    {"type": "streaming", "budget": 12, "sink": 2},
    {"type": "h2o", "budget": 12, "recent": 4},
]


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.mark.parametrize("spec", SPECS, ids=[s["type"] for s in SPECS])
def test_fill_then_decode(spec):
    model = tiny_model()
    runner = QwenRunner(model)
    cache = make_cache(spec, model.config, 3, 64, torch.float32)
    fill_synthetic(cache, 37)
    budget = spec.get("budget", 10**9)
    assert cache.seq_len(0) == min(37, budget) and cache.tokens_seen(0) == 37
    ids = torch.randint(0, 128, (3, 1))
    for _ in range(3):
        logits = runner.forward(ids, cache, last_only=True)
        assert logits.shape == (3, 1, 128) and torch.isfinite(logits).all()
    assert cache.tokens_seen(0) == 40 and cache.seq_len(0) == min(40, budget)


def test_fill_quant_window_sizes():
    model = tiny_model()
    cache = make_cache({"type": "kivi", "bits": 2, "group_size": 4, "residual": 8}, model.config, 1, 64)
    fill_synthetic(cache, 21)
    ks, vs = cache.k[0], cache.v[0]
    assert (ks.n_quant, ks.window.shape[2]) == (12, 9)
    assert (vs.n_quant, vs.window.shape[2]) == (13, 8)


def test_fill_overflow_and_type_errors():
    with pytest.raises(ValueError):
        fill_synthetic(StaticKVCache(L, B, H, D, 8), 9)
    with pytest.raises(TypeError):
        fill_synthetic(object(), 4)
