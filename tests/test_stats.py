import pytest

from kvshrink.stats import paired_bootstrap


def test_identical_is_zero():
    assert paired_bootstrap({"t": [(0.3, 0.3)] * 20}) == (0.0, 0.0, 0.0)


def test_constant_shift():
    p, lo, hi = paired_bootstrap({"t": [(x, x + 1.0) for x in range(20)]})
    assert abs(p - 1.0) < 1e-12 and abs(lo - 1.0) < 1e-12 and abs(hi - 1.0) < 1e-12


def test_strata_weighted_equally():
    p, _, _ = paired_bootstrap({"x": [(0.0, 1.0)] * 10, "y": [(0.0, 0.0)] * 30})
    assert p == 0.5


def test_noisy_ci_contains_point_and_is_deterministic():
    pairs = [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (1.0, 1.0)] * 10
    p, lo, hi = paired_bootstrap({"t": pairs})
    assert p == 0.0 and lo < 0.0 < hi
    assert paired_bootstrap({"t": pairs}) == (p, lo, hi)


def test_empty_raises():
    with pytest.raises(ValueError):
        paired_bootstrap({"t": []})
