import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvshrink.evict_cache import StreamingLLMCache
from kvshrink.factory import make_cache
from kvshrink.model import QwenRunner

L, B, H, D = 2, 2, 2, 8


def stream(cache, chunks, seed=0):
    """Append chunks of random K/V to layer 0; return the concatenated references."""
    g = torch.Generator().manual_seed(seed)
    ks, vs = [], []
    for t in chunks:
        k, v = torch.randn(B, H, t, D, generator=g), torch.randn(B, H, t, D, generator=g)
        ks.append(k)
        vs.append(v)
        cache.update(0, k, v)
    return torch.cat(ks, 2), torch.cat(vs, 2)


def test_keeps_sinks_and_recent_within_budget():
    c = StreamingLLMCache(L, B, H, D, budget=10, sink=2, dtype=torch.float32)
    k, v = stream(c, [5] + [1] * 20)
    assert c.seq_len(0) == 10 and c.tokens_seen(0) == 25
    assert c.seq_len(1) == 0 and c.tokens_seen(1) == 0
    assert torch.equal(c.k[0], torch.cat([k[:, :, :2], k[:, :, -8:]], 2))
    assert torch.equal(c.v[0], torch.cat([v[:, :, :2], v[:, :, -8:]], 2))


def test_update_returns_kept_past_plus_new_tokens():
    c = StreamingLLMCache(L, B, H, D, budget=6, sink=2, dtype=torch.float32)
    k, _ = stream(c, [10])
    kept = torch.cat([k[:, :, :2], k[:, :, -4:]], 2)
    k2, v2 = torch.randn(B, H, 3, D), torch.randn(B, H, 3, D)
    K, _ = c.update(0, k2, v2)
    assert torch.equal(K, torch.cat([kept, k2], 2))
    assert c.seq_len(0) == 6 and c.tokens_seen(0) == 13
    assert torch.equal(c.k[0], torch.cat([K[:, :, :2], K[:, :, -4:]], 2))


def test_long_prefill_attends_fully_then_evicts():
    c = StreamingLLMCache(L, B, H, D, budget=8, sink=4, dtype=torch.float32)
    k, v = torch.randn(B, H, 20, D), torch.randn(B, H, 20, D)
    K, V = c.update(0, k, v)
    assert torch.equal(K, k) and torch.equal(V, v)
    assert torch.equal(c.k[0], torch.cat([k[:, :, :4], k[:, :, -4:]], 2))


def test_invalid_budget_raises():
    with pytest.raises(ValueError):
        StreamingLLMCache(L, B, H, D, budget=4, sink=4)


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def test_budget_covering_sequence_matches_static():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(6))
    cache = make_cache({"type": "streaming", "budget": 64, "sink": 4}, model.config, 2, 64, torch.float32)
    assert torch.equal(runner.greedy_generate(ids, 30, cache=cache), runner.greedy_generate(ids, 30))


def test_long_decode_stays_within_budget():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(7))
    cache = make_cache({"type": "streaming", "budget": 12, "sink": 2}, model.config, 2, 12, torch.float32)
    out = runner.greedy_generate(ids, 40, cache=cache)
    assert out.shape == (2, 50)
    assert all(cache.seq_len(i) == 12 for i in range(2))
    assert all(cache.tokens_seen(i) == 49 for i in range(2))
