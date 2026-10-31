import pytest
import torch

from kvshrink.cache import StaticKVCache

L, B, H, D, MAXLEN = 3, 2, 2, 8, 16


def make_cache(dtype=torch.float32):
    return StaticKVCache(L, B, H, D, MAXLEN, dtype=dtype)


def rand(t):
    return torch.randn(B, H, t, D)


def test_update_matches_concat():
    torch.manual_seed(0)
    cache = make_cache()
    chunks_k, chunks_v = [rand(5), rand(1), rand(1)], [rand(5), rand(1), rand(1)]
    for k, v in zip(chunks_k, chunks_v):
        K, V = cache.update(0, k, v)
    assert torch.equal(K, torch.cat(chunks_k, dim=2))
    assert torch.equal(V, torch.cat(chunks_v, dim=2))
    assert cache.seq_len(0) == 7


def test_layers_are_independent():
    cache = make_cache()
    cache.update(0, rand(4), rand(4))
    cache.update(1, rand(2), rand(2))
    assert [cache.seq_len(i) for i in range(L)] == [4, 2, 0]


def test_overflow_raises():
    cache = make_cache()
    cache.update(0, rand(MAXLEN), rand(MAXLEN))
    with pytest.raises(ValueError):
        cache.update(0, rand(1), rand(1))


def test_bad_shape_raises():
    cache = make_cache()
    with pytest.raises(ValueError):
        cache.update(0, torch.randn(B, H + 1, 1, D), torch.randn(B, H + 1, 1, D))


def test_memory_accounting_fp16():
    cache = make_cache(torch.float16)
    assert cache.memory_bytes() == 2 * L * B * H * MAXLEN * D * 2
    cache.update(0, rand(3), rand(3))
    cache.update(1, rand(3), rand(3))
    assert cache.used_bytes() == 2 * B * H * D * 2 * 6


def test_stores_in_cache_dtype():
    cache = make_cache(torch.float16)
    K, _ = cache.update(0, rand(2), rand(2))
    assert K.dtype == torch.float16
