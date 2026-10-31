"""Quantile helpers with an explicit linear-interpolation definition."""

from __future__ import annotations

import math
from typing import Sequence


def linear_quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Quantile at position ``(N-1)*q`` with linear interpolation.

    ``sorted_values`` must be sorted ascending.  Returns ``None`` for an empty
    input.  No epsilon substitution is performed anywhere.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    position = (n - 1) * q
    lo = int(math.floor(position))
    hi = min(lo + 1, n - 1)
    frac = position - lo
    return float(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))


def quantile_set(values: Sequence[float]) -> dict:
    ordered = sorted(values)
    return {
        "p50": linear_quantile(ordered, 0.50),
        "p90": linear_quantile(ordered, 0.90),
        "p99": linear_quantile(ordered, 0.99),
    }
