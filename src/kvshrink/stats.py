"""Paired bootstrap statistics for comparing methods evaluated on identical samples."""
from __future__ import annotations

import random
import statistics


def paired_bootstrap(
    groups: dict[str, list[tuple[float, float]]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean over groups of mean(b - a), with a stratified 95% percentile-bootstrap CI.

    `groups` maps a stratum (e.g. task or context length) to paired observations
    (a, b) measured on the same sample. Resampling happens within each stratum,
    so every stratum keeps equal weight. Returns (point, lo, hi).
    """
    diffs = {g: [b - a for a, b in pairs] for g, pairs in groups.items() if pairs}
    if not diffs:
        raise ValueError("no paired observations")
    point = statistics.fmean(statistics.fmean(d) for d in diffs.values())
    rng = random.Random(seed)
    boots = sorted(
        statistics.fmean(statistics.fmean(rng.choices(d, k=len(d))) for d in diffs.values())
        for _ in range(n_boot)
    )
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
