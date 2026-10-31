import math

import pytest

from kvshrink.needle import build_ids, extract_number, insert_at_depth, mean_ci95, t95


@pytest.mark.parametrize("depth,pos", [(0.0, 0), (0.5, 5), (1.0, 10), (0.26, 3)])
def test_insert_at_depth_position(depth, pos):
    hay = list(range(10))
    out = insert_at_depth(hay, [-1, -2], depth)
    assert out[pos:pos + 2] == [-1, -2]
    assert [t for t in out if t >= 0] == hay


def test_insert_rejects_bad_depth():
    with pytest.raises(ValueError):
        insert_at_depth([1, 2], [9], 1.5)


def test_build_ids_exact_length_and_parts():
    pre, post, needle = [100, 101], [200], [-7, -7]
    source = list(range(1000))
    ids = build_ids(pre, source, needle, post, length=40, depth=0.5, offset=50)
    assert len(ids) == 40
    assert ids[:2] == pre and ids[-1:] == post
    assert ids.count(-7) == 2
    hay = [t for t in ids[2:-1] if t != -7]
    assert hay == list(range(50, 50 + 35))


def test_build_ids_errors():
    with pytest.raises(ValueError):
        build_ids([1] * 10, list(range(100)), [9], [2] * 10, length=20, depth=0.5, offset=0)
    with pytest.raises(ValueError):
        build_ids([1], list(range(20)), [9], [2], length=30, depth=0.5, offset=0)


def test_extract_number():
    assert extract_number("The passkey is 48213.") == "48213"
    assert extract_number(" 07 then 9") == "07"
    assert extract_number("no digits") is None


def test_mean_ci95():
    assert mean_ci95([1.0, 1.0, 1.0]) == (1.0, 0.0)
    m, h = mean_ci95([0.0, 1.0])
    assert m == 0.5 and math.isclose(h, 12.706 * math.sqrt(0.5) / math.sqrt(2))
    assert t95(2) == 4.303 and t95(12) == 2.228 and t95(100) == 1.96
    assert math.isnan(mean_ci95([0.3])[1])
