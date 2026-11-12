"""
Summary statistics used in every result table (PLAN §3: spread next to every
mean, Wilson intervals for rates).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95 %).
    Better behaved than the normal approximation near 0 and 1 and for small n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def target_verdict(successes: int, n: int, target: float) -> str:
    """'met' if the whole 95 % interval is >= target, 'not met' if it is
    entirely below, otherwise 'inconclusive (CI spans target)'."""
    lo, hi = wilson_interval(successes, n)
    if lo >= target:
        return "met"
    if hi < target:
        return "not met"
    return "inconclusive (CI spans target)"


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample std (ddof=1) over seeds, ignoring NaN."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0)
