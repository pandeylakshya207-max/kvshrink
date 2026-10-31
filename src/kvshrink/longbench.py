"""LongBench-style scoring and helpers.

Metrics follow LongBench (Bai et al., 2023): answers are normalized (lowercase,
no punctuation, no articles, collapsed whitespace) before token-level F1; passage
retrieval scores the fraction of numbers in the prediction that equal the
ground-truth paragraph id.
"""
from __future__ import annotations

import random
import re
import statistics
import string
from collections import Counter

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: list[str], gold: list[str]) -> float:
    same = sum((Counter(pred) & Counter(gold)).values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(gold)
    return 2 * p * r / (p + r)


def qa_f1(prediction: str, answers: list[str]) -> float:
    """Max token F1 of the normalized prediction against any normalized answer."""
    pred = normalize_answer(prediction).split()
    return max(_f1(pred, normalize_answer(a).split()) for a in answers)


def retrieval_score(prediction: str, answers: list[str]) -> float:
    """Fraction of numbers in the prediction equal to the gold 'Paragraph N' id."""
    m = re.search(r"Paragraph (\d+)", answers[0])
    if m is None:
        raise ValueError(f"no 'Paragraph N' in ground truth: {answers[0]!r}")
    nums = re.findall(r"\d+", prediction)
    if not nums:
        return 0.0
    return sum(n == m.group(1) for n in nums) / len(nums)


METRICS = {"qa_f1": qa_f1, "retrieval": retrieval_score}


def middle_truncate(ids: list[int], max_len: int) -> list[int]:
    """Keep the first and last halves of `ids` if longer than `max_len` (LongBench style)."""
    if len(ids) <= max_len:
        return ids
    half = max_len // 2
    return ids[:half] + ids[len(ids) - (max_len - half):]


def bootstrap_ci(xs: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval (lo, hi) of the mean."""
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(statistics.fmean(rng.choices(xs, k=k)) for _ in range(n_boot))
    return statistics.fmean(xs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
