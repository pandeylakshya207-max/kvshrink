import pytest
import torch

from kvshrink.quant import dequantize, pack, quantize, unpack

CASES = [(-1, 64), (-1, 32), (-2, 32), (-2, 16)]


def half_step_bound(x: torch.Tensor, bits: int, group: int, dim: int) -> torch.Tensor:
    """Independent reference: half the quantization step of each element's group."""
    xt = x.movedim(dim, -1)
    g = xt.reshape(*xt.shape[:-1], xt.shape[-1] // group, group)
    step = (g.amax(-1, keepdim=True) - g.amin(-1, keepdim=True)) / (2**bits - 1)
    return (step / 2).expand_as(g).reshape(xt.shape).movedim(-1, dim)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dim,group", CASES)
def test_error_within_half_step(bits, dim, group):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    err = (dequantize(quantize(x, bits, group, dim)) - x).abs()
    assert (err <= half_step_bound(x, bits, group, dim) + 1e-6).all()


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_unpack_roundtrip(bits):
    torch.manual_seed(0)
    q = torch.randint(0, 2**bits, (3, 5, 32), dtype=torch.uint8)
    p = pack(q, bits)
    assert p.shape[-1] == 32 * bits // 8
    assert torch.equal(unpack(p, bits), q)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_constant_group_is_exact(bits):
    x = torch.full((2, 2, 8, 64), 3.25)
    assert torch.equal(dequantize(quantize(x, bits, 32)), x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("dim,group", CASES)
def test_shape_and_dtype_preserved(dtype, dim, group):
    x = torch.randn(2, 3, 64, 64).to(dtype)
    y = dequantize(quantize(x, 4, group, dim))
    assert y.shape == x.shape and y.dtype == dtype


def test_fp16_8bit_close():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 128, 64).half()
    err = (dequantize(quantize(x, 8, 64)) - x).float().abs()
    rng = (x.float().amax(-1, keepdim=True) - x.float().amin(-1, keepdim=True))
    assert (err <= rng / 255 * 0.5 + rng * 2e-3).all()


def test_error_decreases_with_bits():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 64, 64)
    errs = [(dequantize(quantize(x, b, 32)) - x).abs().mean().item() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_memory_bytes_fp16_4bit():
    x = torch.randn(2, 2, 64, 64).half()
    qt = quantize(x, 4, 32)
    codes = 2 * 2 * 64 * 64 // 2  # 4-bit -> half a byte each
    meta = 2 * (2 * 2 * 64 * (64 // 32)) * 2  # scale + zero, one fp16 each per group
    assert qt.memory_bytes() == codes + meta


def test_invalid_args_raise():
    x = torch.randn(2, 2, 8, 64)
    with pytest.raises(ValueError):
        quantize(x, 3, 32)
    with pytest.raises(ValueError):
        quantize(x, 4, 48)
