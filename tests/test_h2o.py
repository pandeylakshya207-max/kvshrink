import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvshrink.evict_cache import H2OCache
from kvshrink.factory import make_cache
from kvshrink.model import QwenRunner

L, B, H, D = 2, 2, 2, 8


def test_keeps_top_scored_plus_recent():
    c = H2OCache(L, B, H, D, budget=6, recent=2, dtype=torch.float32)
    k, v = torch.randn(B, H, 10, D), torch.randn(B, H, 10, D)
    c.update(0, k, v)
    attn = torch.zeros(B, H, 10)
    attn[..., [1, 3, 4, 6]] = 1.0
    c.observe(0, attn)
    idx = [1, 3, 4, 6, 8, 9]
    assert torch.equal(c.k[0], k[:, :, idx]) and torch.equal(c.v[0], v[:, :, idx])
    assert torch.equal(c.score[0], attn[..., idx])
    assert c.seq_len(0) == 6 and c.tokens_seen(0) == 10


def test_each_kv_head_keeps_its_own_tokens():
    c = H2OCache(L, B, H, D, budget=6, recent=2, dtype=torch.float32)
    k = torch.randn(B, H, 10, D)
    c.update(0, k, k.clone())
    attn = torch.zeros(B, H, 10)
    attn[:, 0, [0, 1, 2, 3]] = 1.0
    attn[:, 1, [4, 5, 6, 7]] = 1.0
    c.observe(0, attn)
    assert torch.equal(c.k[0][:, 0], k[:, 0, [0, 1, 2, 3, 8, 9]])
    assert torch.equal(c.k[0][:, 1], k[:, 1, [4, 5, 6, 7, 8, 9]])


def test_scores_accumulate_across_steps():
    c = H2OCache(L, B, H, D, budget=4, recent=1, dtype=torch.float32)
    k1, k2 = torch.randn(B, H, 4, D), torch.randn(B, H, 1, D)
    c.update(0, k1, k1.clone())
    c.observe(0, torch.tensor([5.0, 0.0, 0.0, 0.5]).expand(B, H, 4))
    c.update(0, k2, k2.clone())
    c.observe(0, torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]).expand(B, H, 5))
    allk = torch.cat([k1, k2], 2)
    assert torch.equal(c.k[0], allk[:, :, [0, 2, 3, 4]])
    assert torch.allclose(c.score[0][0, 0], torch.tensor([5.0, 2.0, 0.5, 0.0]))


def test_bad_args_and_shapes_raise():
    with pytest.raises(ValueError):
        H2OCache(L, B, H, D, budget=4, recent=5)
    c = H2OCache(L, B, H, D, budget=4, recent=1)
    c.update(0, torch.randn(B, H, 3, D), torch.randn(B, H, 3, D))
    with pytest.raises(ValueError):
        c.observe(0, torch.zeros(B, H, 2))


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def test_large_budget_matches_static_logits_and_tokens():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(8))
    h2o = make_cache({"type": "h2o", "budget": 64, "recent": 8}, model.config, 2, 64, torch.float32)
    ours = runner.forward(ids, h2o)
    ref = runner.forward(ids, runner.new_cache(2, 12))
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()
    h2o = make_cache({"type": "h2o", "budget": 64, "recent": 8}, model.config, 2, 64, torch.float32)
    assert torch.equal(runner.greedy_generate(ids, 30, cache=h2o), runner.greedy_generate(ids, 30))


def test_long_decode_stays_within_budget():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(9))
    cache = make_cache({"type": "h2o", "budget": 12, "recent": 4}, model.config, 2, 12, torch.float32)
    out = runner.greedy_generate(ids, 40, cache=cache)
    assert out.shape == (2, 50)
    assert all(cache.seq_len(i) == 12 for i in range(2))
    assert all(cache.tokens_seen(i) == 49 for i in range(2))
