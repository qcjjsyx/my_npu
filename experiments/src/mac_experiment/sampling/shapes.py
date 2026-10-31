"""Continuous target shapes used to define coding PMFs.

All shapes are standardised to unit variance.  Interval masses are computed
with stable survival-function forms in the upper tail to avoid catastrophic
cancellation of ``cdf(u) - cdf(l)``.
"""

from __future__ import annotations

import math

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
LAPLACE_SCALE = 1.0 / math.sqrt(2.0)
OUTLIER_SCALE = math.sqrt(0.99 + 0.01 * 64.0)
OUTLIER_COMPONENT_SIGMA = 8.0


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT2)


def normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / SQRT2)


def laplace_cdf(x: float) -> float:
    if x < 0.0:
        return 0.5 * math.exp(x / LAPLACE_SCALE)
    return 1.0 - 0.5 * math.exp(-x / LAPLACE_SCALE)


def laplace_sf(x: float) -> float:
    if x >= 0.0:
        return 0.5 * math.exp(-x / LAPLACE_SCALE)
    return 1.0 - 0.5 * math.exp(x / LAPLACE_SCALE)


def _t3_cdf_t(t: float) -> float:
    return 0.5 + (math.atan(t) + t / (1.0 + t * t)) / math.pi


def _t3_sf_t(t: float) -> float:
    if t <= 0.0:
        return 1.0 - _t3_cdf_t(t)
    s = SQRT3 / t
    if s > 0.2:
        return (math.atan(s) - s / (1.0 + s * s)) / math.pi
    s2 = s * s
    term = s * s2
    total = 0.0
    k = 1
    sign = 1.0
    while True:
        add = sign * (2.0 * k / (2.0 * k + 1.0)) * term
        total += add
        if abs(add) <= 1e-17 * abs(total) or k >= 60:
            break
        term *= s2
        sign = -sign
        k += 1
    return total / math.pi


def student_t3_cdf(x: float) -> float:
    return _t3_cdf_t(x * SQRT3)


def student_t3_sf(x: float) -> float:
    return _t3_sf_t(x * SQRT3)


def outlier_cdf(x: float) -> float:
    z = x * OUTLIER_SCALE
    return 0.99 * normal_cdf(z) + 0.01 * normal_cdf(z / OUTLIER_COMPONENT_SIGMA)


def outlier_sf(x: float) -> float:
    z = x * OUTLIER_SCALE
    return 0.99 * normal_sf(z) + 0.01 * normal_sf(z / OUTLIER_COMPONENT_SIGMA)


SHAPES = {
    "laplace": (laplace_cdf, laplace_sf),
    "student_t3": (student_t3_cdf, student_t3_sf),
    "outlier": (outlier_cdf, outlier_sf),
}

SHAPE_TARGETS = {
    "laplace": "unit-variance Laplace (scale 1/sqrt(2))",
    "student_t3": "t(3)/sqrt(3) (unit variance)",
    "outlier": "0.99*N(0,1)+0.01*N(0,64), scaled to unit variance",
}


def interval_mass(cdf, sf, lo: float, hi: float) -> float:
    """Probability mass of ``(lo, hi]`` using the numerically stable branch."""
    if lo >= 0.0:
        mass = sf(lo) - sf(hi)
    elif hi <= 0.0:
        mass = cdf(hi) - cdf(lo)
    else:
        mass = cdf(hi) - cdf(lo)
    return mass if mass > 0.0 else 0.0
