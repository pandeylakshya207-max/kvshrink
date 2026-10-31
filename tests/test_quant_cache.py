import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvshrink.cache import StaticKVCache
from kvshrink.model import QwenRunner
from kvshrink.quant_cache import QuantizedKVCache, int8_per_token, kivi

L, B, H, D, MAXLEN = 2, 2, 2, 16, 64


def feed(cache, n_prefill: int, n_decode: int, seed: int = 0, layer: int = 0):
    """Prefill then single-token appends. Returns (ref K, ref V, stored K, stored V)."""
    g = torch.Generator().manual_seed(seed)
    ks, vs = [], []
    for t in [n_prefill] + [1] * n_decode:
        if t == 0:
            continue
        k, v = torch.randn(B, H, t, D, generator=g), torch.randn(B, H, t, D, generator=g)
        ks.append(k)
        vs.append(v)
        cache.update(layer, k, v)
    return torch.cat(ks, 2), torch.cat(vs, 2), cache.k[layer].dense(), cache.v[layer].dense()


def test_int8_per_token_error_bound():
    cache = int8_per_token(L, B, H, D, MAXLEN, dtype=torch.float32)
    k, v, K, V = feed(cache, 10, 5)
    for ref, got in ((k, K), (v, V)):
        step = (ref.amax(-1, keepdim=True) - ref.amin(-1, keepdim=True)) / 255
        assert got.shape == ref.shape
        assert ((got - ref).abs() <= step / 2 + 1e-6).all()
    assert cache.seq_len(0) == 15 and cache.seq_len(1) == 0


@pytest.mark.parametrize("n_prefill,n_decode", [(0, 21), (30, 3)])
def test_kivi_window_exact_and_quantized_within_bounds(n_prefill, n_decode):
    R, G = 8, 4
    cache = kivi(L, B, H, D, MAXLEN, bits=2, group_size=G, residual=R, dtype=torch.float32)
    k, v, K, V = feed(cache, n_prefill, n_decode)
    ks, vs = cache.k[0], cache.v[0]
    assert cache.seq_len(0) == n_prefill + n_decode == K.shape[2] == V.shape[2]
    assert R <= ks.window.shape[2] < R + G and ks.n_quant % G == 0
    assert vs.window.shape[2] == R
    w = ks.window.shape[2]
    assert torch.equal(K[:, :, -w:], k[:, :, -w:])
    assert torch.equal(V[:, :, -R:], v[:, :, -R:])
    nk = ks.n_quant
    kb = k[:, :, :nk].reshape(B, H, nk // G, G, D)
    kstep = (kb.amax(3, keepdim=True) - kb.amin(3, keepdim=True)) / 3
    assert ((K[:, :, :nk].reshape(B, H, nk // G, G, D) - kb).abs() <= kstep / 2 + 1e-6).all()
    nv = vs.n_quant
    vb = v[:, :, :nv].reshape(B, H, nv, D // G, G)
    vstep = (vb.amax(-1, keepdim=True) - vb.amin(-1, keepdim=True)) / 3
    assert ((V[:, :, :nv].reshape(B, H, nv, D // G, G) - vb).abs() <= vstep / 2 + 1e-6).all()


def test_residual_longer_than_sequence_is_lossless():
    cache = kivi(L, B, H, D, MAXLEN, bits=2, group_size=4, residual=MAXLEN, dtype=torch.float32)
    k, v, K, V = feed(cache, 20, 10)
    assert torch.equal(K, k) and torch.equal(V, v)


def test_update_returns_new_tokens_in_full_precision():
    cache = int8_per_token(L, B, H, D, MAXLEN, dtype=torch.float32)
    feed(cache, 10, 0)
    k, v = torch.randn(B, H, 3, D), torch.randn(B, H, 3, D)
    K, V = cache.update(0, k, v)
    assert torch.equal(K[:, :, -3:], k) and torch.equal(V[:, :, -3:], v)


def test_int8_used_bytes_analytic():
    cache = int8_per_token(L, B, H, D, MAXLEN, dtype=torch.float16)
    for layer in range(L):
        feed(cache, 12, 0, layer=layer)
    # per token per head: D one-byte codes + fp16 scale + fp16 zero
    assert cache.used_bytes() == L * 2 * B * H * 12 * (D + 4)


def test_used_bytes_ordering():
    def used(c):
        for layer in range(L):
            c.update(layer, torch.randn(B, H, 32, D), torch.randn(B, H, 32, D))
        return c.used_bytes()

    sizes = [
        used(StaticKVCache(L, B, H, D, MAXLEN, dtype=torch.float16)),
        used(int8_per_token(L, B, H, D, MAXLEN)),
        used(kivi(L, B, H, D, MAXLEN, bits=4, group_size=8, residual=0)),
        used(kivi(L, B, H, D, MAXLEN, bits=2, group_size=8, residual=0)),
    ]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == 4, sizes


def test_overflow_raises():
    cache = int8_per_token(L, B, H, D, 8, dtype=torch.float32)
    cache.update(0, torch.randn(B, H, 8, D), torch.randn(B, H, 8, D))
    with pytest.raises(ValueError):
        cache.update(0, torch.randn(B, H, 1, D), torch.randn(B, H, 1, D))


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def test_lossless_quant_cache_matches_static_greedy():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(4))
    cache = QuantizedKVCache.from_config(
        model.config, 2, 40, dtype=torch.float32, key_bits=2, value_bits=2,
        key_mode="channel", value_mode="token", key_group=4, value_group=4, residual=64,
    )
    assert torch.equal(runner.greedy_generate(ids, 30, cache=cache), runner.greedy_generate(ids, 30))


def test_int8_cache_decode_logits_close():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 28), generator=torch.Generator().manual_seed(5))
    ref_cache = runner.new_cache(2, 28)
    q_cache = QuantizedKVCache.from_config(model.config, 2, 28, dtype=torch.float32, key_bits=8, value_bits=8)
    runner.forward(ids[:, :20], ref_cache)
    runner.forward(ids[:, :20], q_cache)
    for t in range(20, 28):
        ref = runner.forward(ids[:, t:t + 1], ref_cache)
        got = runner.forward(ids[:, t:t + 1], q_cache)
        rel = (got - ref).norm() / ref.norm()
        assert rel < 0.05, (t, rel.item())


def test_layer_overrides_per_layer_bits():
    cache = QuantizedKVCache(L, B, H, D, MAXLEN, key_bits=4, value_bits=4, key_mode="channel",
                             value_mode="token", key_group=8, value_group=8, residual=0,
                             layer_overrides={1: {"key_bits": 2, "value_bits": 2}})
    assert (cache.k[0].bits, cache.v[0].bits) == (4, 4)
    assert (cache.k[1].bits, cache.v[1].bits) == (2, 2)
    for layer in range(L):
        cache.update(layer, torch.randn(B, H, 16, D), torch.randn(B, H, 16, D))
    full4 = QuantizedKVCache(L, B, H, D, MAXLEN, key_bits=4, value_bits=4, key_mode="channel",
                             value_mode="token", key_group=8, value_group=8, residual=0)
    for layer in range(L):
        full4.update(layer, torch.randn(B, H, 16, D), torch.randn(B, H, 16, D))
    assert cache.used_bytes() < full4.used_bytes()


def test_layer_overrides_validation():
    with pytest.raises(ValueError):
        QuantizedKVCache(L, B, H, D, MAXLEN, key_bits=4, value_bits=4, layer_overrides={L: {"key_bits": 2}})
    with pytest.raises(ValueError):
        QuantizedKVCache(L, B, H, D, MAXLEN, key_bits=4, value_bits=4, layer_overrides={0: {"bitz": 2}})


def test_factory_kivi_layer_overrides():
    from kvshrink.factory import make_cache
    model = tiny_model()
    spec = {"type": "kivi", "bits": 4, "group_size": 4, "residual": 8,
            "layer_overrides": {0: {"key_bits": 2, "value_bits": 2}}}
    c = make_cache(spec, model.config, 1, 32, torch.float32)
    assert (c.k[0].bits, c.k[1].bits) == (2, 4)
