"""Discrete magnitude PMFs over legal positive normal codes.

Two construction rules are used:

* ``exponent_uniform`` / ``exponent_gaussian``: the raw exponent follows a
  (truncated) discrete distribution and the fraction is conditionally uniform
  over the legal fractions of that exponent.
* ``code_interval``: a continuous unit-variance target shape is integrated over
  intervals around each ordered positive code value.  Boundaries are arithmetic
  midpoints between neighbouring values; the lowest boundary extends half a
  spacing downward (clamped at zero) and the highest half a spacing upward.
  The retained mass is renormalised over the finite code support; the excluded
  tail mass is recorded, never clamped onto the end codes.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from ..arithmetic.dyadic import Dyadic
from ..formats.base import Format
from .shapes import SHAPES, SHAPE_TARGETS, interval_mass


@dataclass
class DiscretePMF:
    codes: list[int]
    values: list[float]
    probs: list[float]
    cumsum: list[float]
    kind: str
    shape: str | None = None
    metadata: dict = field(default_factory=dict)

    def sample(self, rng) -> int:
        u = rng.random()
        idx = bisect.bisect_left(self.cumsum, u)
        if idx >= len(self.codes):
            idx = len(self.codes) - 1
        return self.codes[idx]

    def rms(self) -> float:
        return math.sqrt(math.fsum(p * v * v for p, v in zip(self.probs, self.values)))

    def quantile(self, q: float) -> float:
        idx = bisect.bisect_left(self.cumsum, q)
        if idx >= len(self.values):
            idx = len(self.values) - 1
        return self.values[idx]

    def value_quantiles(self) -> dict:
        return {
            "p50": self.quantile(0.5),
            "p90": self.quantile(0.9),
            "p99": self.quantile(0.99),
        }


def _normalise(weights: list[float]) -> tuple[list[float], float]:
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("PMF weights sum to zero")
    return [w / total for w in weights], total


def build_magnitude_pmf(
    fmt: Format,
    kind: str,
    shape: str | None = None,
    exp_min: int | None = None,
    exp_max: int | None = None,
) -> DiscretePMF:
    lo = fmt.sampling_exp_min if exp_min is None else exp_min
    hi = fmt.sampling_exp_max if exp_max is None else exp_max
    if kind in ("exponent_uniform", "exponent_gaussian"):
        return _build_exponent_pmf(fmt, kind, lo, hi)
    if kind == "code_interval":
        if shape is None:
            raise ValueError("code_interval PMF requires a shape")
        return _build_code_interval_pmf(fmt, shape, lo, hi)
    raise ValueError(f"unknown magnitude kind {kind!r}")


def _build_exponent_pmf(fmt: Format, kind: str, exp_min: int, exp_max: int) -> DiscretePMF:
    exps = list(range(exp_min, exp_max + 1))
    if kind == "exponent_uniform":
        weights = [1.0] * len(exps)
        excluded = 0.0
        params = {"rule": "raw exponent uniform; fraction conditionally uniform"}
    else:
        mu = fmt.gaussian_exp_center
        sigma = fmt.gaussian_exp_sigma
        weights = [math.exp(-0.5 * ((e - mu) / sigma) ** 2) for e in exps]
        all_exps = list(range(1, 255))
        all_weights = [math.exp(-0.5 * ((e - mu) / sigma) ** 2) for e in all_exps]
        kept = math.fsum(weights)
        excluded = 1.0 - kept / math.fsum(all_weights)
        params = {
            "rule": "raw exponent from truncated discrete Gaussian; fraction conditionally uniform",
            "center": mu,
            "sigma": sigma,
        }
    probs_exp, _ = _normalise(weights)
    codes: list[int] = []
    values: list[float] = []
    probs: list[float] = []
    for e, pe in zip(exps, probs_exp):
        fractions = fmt.legal_fractions(e)
        for frac in fractions:
            code = fmt.encode(0, e, frac)
            codes.append(code)
            values.append(fmt.operand_value(fmt.decode(code)).to_float())
            probs.append(pe / len(fractions))
    order = sorted(range(len(codes)), key=lambda i: values[i])
    codes = [codes[i] for i in order]
    values = [values[i] for i in order]
    probs = [probs[i] for i in order]
    cumsum = _cumsum(probs)
    pmf = DiscretePMF(codes=codes, values=values, probs=probs, cumsum=cumsum, kind=kind)
    pmf.metadata = {
        "kind": kind,
        "params": params,
        "support": {"exp_min": exp_min, "exp_max": exp_max, "code_count": len(codes)},
        "retained_probability": 1.0 - excluded,
        "excluded_probability": excluded,
        "discrete_rms": pmf.rms(),
        "magnitude_quantiles": pmf.value_quantiles(),
    }
    return pmf


def _build_code_interval_pmf(fmt: Format, shape: str, exp_min: int, exp_max: int) -> DiscretePMF:
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}")
    cdf, sf = SHAPES[shape]
    code_values = fmt.positive_normal_codes(exp_min, exp_max)
    codes = [code for code, _ in code_values]
    dyadic_values = [value for _, value in code_values]
    half = Dyadic(1, -1)
    boundaries: list[Dyadic] = []
    first_spacing = dyadic_values[1] - dyadic_values[0]
    lower0 = dyadic_values[0] - first_spacing * half
    boundaries.append(lower0 if lower0.m > 0 else Dyadic.zero())
    for j in range(1, len(dyadic_values)):
        boundaries.append((dyadic_values[j - 1] + dyadic_values[j]) * half)
    last_spacing = dyadic_values[-1] - dyadic_values[-2]
    boundaries.append(dyadic_values[-1] + last_spacing * half)
    masses = [
        interval_mass(cdf, sf, boundaries[j].to_float(), boundaries[j + 1].to_float())
        for j in range(len(dyadic_values))
    ]
    retained = math.fsum(masses)
    probs = [m / retained for m in masses]
    values = [value.to_float() for value in dyadic_values]
    cumsum = _cumsum(probs)
    pmf = DiscretePMF(codes=codes, values=values, probs=probs, cumsum=cumsum, kind="code_interval", shape=shape)
    pmf.metadata = {
        "kind": "code_interval",
        "shape": shape,
        "target": SHAPE_TARGETS[shape],
        "boundary_rule": (
            "interior boundaries arithmetic midpoints; lowest extends half a spacing down clamped at 0; "
            "highest extends half a spacing up"
        ),
        "support": {"exp_min": exp_min, "exp_max": exp_max, "code_count": len(codes)},
        "retained_probability": retained,
        "excluded_probability": 1.0 - retained,
        "discrete_rms": pmf.rms(),
        "magnitude_quantiles": pmf.value_quantiles(),
    }
    return pmf


def _cumsum(probs: list[float]) -> list[float]:
    out = []
    acc = 0.0
    for p in probs:
        acc += p
        out.append(acc)
    out[-1] = 1.0
    return out
