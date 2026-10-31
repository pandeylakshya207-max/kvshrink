"""Needle-in-a-haystack helpers: token-level prompt construction, answer parsing, CIs."""
from __future__ import annotations

import math
import re
import statistics

# Two-sided 95% Student-t critical values by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df: int) -> float:
    """95% two-sided t critical value (nearest tabulated df at or below; 1.96 for large df)."""
    if df < 1:
        raise ValueError("need at least 2 samples for a confidence interval")
    if df > 30:
        return 1.96
    return _T95[max(k for k in _T95 if k <= df)]


def mean_ci95(xs: list[float]) -> tuple[float, float]:
    """Mean and half-width of the 95% t confidence interval."""
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, t95(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))


def insert_at_depth(haystack: list[int], needle: list[int], depth: float) -> list[int]:
    """Insert `needle` after round(depth * len(haystack)) haystack tokens."""
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    pos = round(len(haystack) * depth)
    return haystack[:pos] + needle + haystack[pos:]


def build_ids(
    pre: list[int], source: list[int], needle: list[int], post: list[int],
    length: int, depth: float, offset: int,
) -> list[int]:
    """Prompt of exactly `length` tokens: pre + haystack(with needle) + post.

    The haystack is source[offset : offset + n], with n chosen so the total is `length`.
    """
    n = length - len(pre) - len(post) - len(needle)
    if n <= 0:
        raise ValueError(f"length {length} too short for pre/post/needle")
    if offset < 0 or offset + n > len(source):
        raise ValueError("haystack source too short for this offset/length")
    ids = pre + insert_at_depth(source[offset: offset + n], needle, depth) + post
    assert len(ids) == length
    return ids


def extract_number(text: str) -> str | None:
    """First run of digits in `text`, or None."""
    m = re.search(r"\d+", text)
    return m.group(0) if m else None
