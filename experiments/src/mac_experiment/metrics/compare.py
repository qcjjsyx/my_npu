"""Streaming error metrics for one candidate/reference comparison.

All errors are exact dyadics until the metric value itself is produced; the
conversion to binary64 happens only after the exact error exists.  Undefined
metrics are reported as ``value: null`` with an explicit status; a zero
denominator is never replaced by 1.

Whole-batch policy
------------------
A comparison has a batch status:

* ``ok``                  - every sample was valid; metrics cover the batch.
* ``partial_nonfinite``   - some samples were invalid (non-finite candidate or
                            reference); whole-batch metrics are ``null`` and a
                            clearly labelled ``finite_subset`` payload is
                            provided as a diagnostic only.
* ``no_valid_samples``    - every sample was invalid; no metric is defined.
* ``no_samples``          - no snapshot record reached this cell at all.

Invalid samples are never treated as finite errors, are never dropped silently
and never get a replaced denominator, a padded zero or a synthetic large error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from ..arithmetic.dyadic import Dyadic
from .quantiles import quantile_set

STATUS_OK = "ok"
STATUS_NO_SAMPLES = "no_samples"
STATUS_NO_VALID = "no_valid_samples"
STATUS_PARTIAL = "partial_nonfinite"
STATUS_NONFINITE = "nonfinite"
STATUS_ZERO_L1 = "undefined_zero_l1"
STATUS_ZERO_REFERENCE = "undefined_zero_reference"

SUBSET_NOTE = "computed over the finite subset only; not a whole-batch metric"

ALL_METRICS = (
    "p99_abs_error",
    "p99_eta_l1",
    "relative_l2",
    "rmse_abs_error",
    "max_abs_error",
    "mean_signed_error",
    "match_rate",
    "p50_abs_error",
    "p90_abs_error",
    "p50_eta_l1",
    "p90_eta_l1",
)


@dataclass
class ComparisonAccumulator:
    name: str
    n_total: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    n_nonfinite_errors: int = 0
    n_zero_l1: int = 0
    n_zero_reference: int = 0
    n_match: int = 0
    abs_errors: list[float] = field(default_factory=list)
    etas: list[float] = field(default_factory=list)
    sum_err2: Dyadic = field(default_factory=Dyadic.zero)
    sum_ref2: Dyadic = field(default_factory=Dyadic.zero)
    sum_signed: Dyadic = field(default_factory=Dyadic.zero)
    max_abs: Dyadic | None = None

    def add(self, error: Dyadic, reference: Dyadic, l1: Dyadic) -> None:
        if error is None:
            raise ValueError("add() requires an exact error; use add_invalid() for invalid samples")
        self.n_total += 1
        self.n_valid += 1
        abs_error = abs(error)
        if error.is_zero():
            self.n_match += 1
        self.sum_err2 = self.sum_err2 + error * error
        self.sum_ref2 = self.sum_ref2 + reference * reference
        self.sum_signed = self.sum_signed + error
        if self.max_abs is None or self.max_abs < abs_error:
            self.max_abs = abs_error
        abs_float = abs_error.to_float()
        if not math.isfinite(abs_float):
            self.n_nonfinite_errors += 1
        self.abs_errors.append(abs_float)
        if l1.is_zero():
            if error.is_zero():
                self.etas.append(0.0)
            else:
                self.n_zero_l1 += 1
        else:
            self.etas.append(_ratio_to_float(abs_error, l1))

    def add_invalid(self) -> None:
        self.n_total += 1
        self.n_invalid += 1

    def merge_from(self, other: "ComparisonAccumulator") -> None:
        """Merge another accumulator of the same comparison (exact sums)."""
        if other.name != self.name:
            raise ValueError(f"cannot merge {other.name!r} into {self.name!r}")
        self.n_total += other.n_total
        self.n_valid += other.n_valid
        self.n_invalid += other.n_invalid
        self.n_nonfinite_errors += other.n_nonfinite_errors
        self.n_zero_l1 += other.n_zero_l1
        self.n_zero_reference += other.n_zero_reference
        self.n_match += other.n_match
        self.abs_errors.extend(other.abs_errors)
        self.etas.extend(other.etas)
        self.sum_err2 = self.sum_err2 + other.sum_err2
        self.sum_ref2 = self.sum_ref2 + other.sum_ref2
        self.sum_signed = self.sum_signed + other.sum_signed
        if other.max_abs is not None and (self.max_abs is None or self.max_abs < other.max_abs):
            self.max_abs = other.max_abs

    def finalize(self) -> dict:
        coverage = self._coverage()
        if self.n_total == 0:
            return self._payload(STATUS_NO_SAMPLES, coverage)
        if self.n_valid == 0:
            return self._payload(STATUS_NO_VALID, coverage)
        subset = self._compute()
        if self.n_invalid > 0:
            payload = self._payload(STATUS_PARTIAL, coverage)
            payload["finite_subset"] = {
                "note": SUBSET_NOTE,
                "coverage": self._valid_coverage(),
                "metrics": subset["metrics"],
                "reference_rms": subset["reference_rms"],
            }
            return payload
        payload = self._payload(STATUS_OK, coverage)
        payload["metrics"] = subset["metrics"]
        payload["reference_rms"] = subset["reference_rms"]
        return payload

    def _payload(self, status: str, coverage: dict) -> dict:
        return {
            "status": status,
            "coverage": coverage,
            "metrics": {key: _metric(None, status) for key in ALL_METRICS},
            "reference_rms": _metric(None, status),
            "finite_subset": None,
            "counts": self._counts(),
        }

    def _compute(self) -> dict:
        abs_q = quantile_set(self.abs_errors)
        eta_override = None
        if self.etas:
            eta_q = quantile_set(self.etas)
        elif self.n_zero_l1 > 0:
            eta_q = {"p50": None, "p90": None, "p99": None}
            eta_override = STATUS_ZERO_L1
        else:
            eta_q = {"p50": None, "p90": None, "p99": None}

        p99_abs, p99_abs_status = _float_metric(abs_q["p99"])
        p99_eta, p99_eta_status = _float_metric(eta_q["p99"])
        if eta_override is not None and p99_eta is None:
            p99_eta_status = eta_override
        rel_l2, rel_l2_status = self._relative_l2()
        rmse, rmse_status = self._rmse()
        max_abs, max_abs_status = _float_metric(self.max_abs.to_float() if self.max_abs is not None else None)
        mean_signed, mean_signed_status = _float_metric(
            (self.sum_signed.to_fraction() / self.n_valid) if self.n_valid else None
        )
        match_rate = self.n_match / self.n_valid if self.n_valid else None
        match_status = STATUS_OK if self.n_valid else STATUS_NO_VALID

        metrics = {
            "p99_abs_error": _metric(p99_abs, p99_abs_status),
            "p99_eta_l1": _metric(p99_eta, p99_eta_status),
            "relative_l2": _metric(rel_l2, rel_l2_status),
            "rmse_abs_error": _metric(rmse, rmse_status),
            "max_abs_error": _metric(max_abs, max_abs_status),
            "mean_signed_error": _metric(mean_signed, mean_signed_status),
            "match_rate": _metric(match_rate, match_status),
            "p50_abs_error": _metric(*_float_metric(abs_q["p50"])),
            "p90_abs_error": _metric(*_float_metric(abs_q["p90"])),
            "p50_eta_l1": _metric(*_float_metric(eta_q["p50"])),
            "p90_eta_l1": _metric(*_float_metric(eta_q["p90"])),
        }
        reference_rms = None
        reference_rms_status = STATUS_NO_SAMPLES
        if self.n_valid:
            reference_rms, reference_rms_status = _float_metric(
                _sqrt_fraction(self.sum_ref2.to_fraction(), self.n_valid)
            )
        return {"metrics": metrics, "reference_rms": _metric(reference_rms, reference_rms_status)}

    def _coverage(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "invalid_rate": (self.n_invalid / self.n_total) if self.n_total else None,
        }

    def _valid_coverage(self) -> dict:
        return {
            "n_total": self.n_valid,
            "n_valid": self.n_valid,
            "n_invalid": 0,
            "invalid_rate": 0.0,
        }

    def _counts(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "n_defined": self.n_valid,
            "n_undefined": self.n_invalid,
            "n_nonfinite_errors": self.n_nonfinite_errors,
            "n_zero_l1": self.n_zero_l1,
            "n_zero_reference": self.n_zero_reference,
            "n_match": self.n_match,
        }

    def _relative_l2(self) -> tuple[float | None, str]:
        if self.sum_ref2.is_zero():
            if self.sum_err2.is_zero():
                return 0.0, STATUS_OK
            self.n_zero_reference += 1
            return None, STATUS_ZERO_REFERENCE
        if self.sum_err2.is_zero():
            return 0.0, STATUS_OK
        value = _sqrt_fraction(self.sum_err2.to_fraction(), self.sum_ref2.to_fraction())
        return _float_metric(value)

    def _rmse(self) -> tuple[float | None, str]:
        if not self.n_valid:
            return None, STATUS_NO_VALID
        value = _sqrt_fraction(self.sum_err2.to_fraction(), self.n_valid)
        return _float_metric(value)


@dataclass
class StageAccumulator:
    """Stage diagnostics with the same whole-batch / finite-subset policy."""

    name: str
    n_total: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    total: Dyadic = field(default_factory=Dyadic.zero)
    total_sq: Dyadic = field(default_factory=Dyadic.zero)

    def add(self, value: Dyadic) -> None:
        self.n_total += 1
        self.n_valid += 1
        self.total = self.total + value
        self.total_sq = self.total_sq + value * value

    def add_invalid(self) -> None:
        self.n_total += 1
        self.n_invalid += 1

    def merge_from(self, other: "StageAccumulator") -> None:
        """Merge another stage accumulator with the same name (exact sums)."""
        if other.name != self.name:
            raise ValueError(f"cannot merge {other.name!r} into {self.name!r}")
        self.n_total += other.n_total
        self.n_valid += other.n_valid
        self.n_invalid += other.n_invalid
        self.total = self.total + other.total
        self.total_sq = self.total_sq + other.total_sq

    def finalize(self) -> dict:
        coverage = {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "invalid_rate": (self.n_invalid / self.n_total) if self.n_total else None,
        }
        if self.n_total == 0:
            return self._payload(STATUS_NO_SAMPLES, coverage)
        if self.n_valid == 0:
            return self._payload(STATUS_NO_VALID, coverage)
        subset = self._compute()
        if self.n_invalid > 0:
            payload = self._payload(STATUS_PARTIAL, coverage)
            payload["finite_subset"] = {"note": SUBSET_NOTE, "coverage": coverage, **subset}
            return payload
        payload = self._payload(STATUS_OK, coverage)
        payload.update(subset)
        return payload

    def _payload(self, status: str, coverage: dict) -> dict:
        return {
            "status": status,
            "coverage": coverage,
            "mean": None,
            "rms": None,
            "n": self.n_valid,
            "finite_subset": None,
        }

    def _compute(self) -> dict:
        mean = self.total.to_fraction() / self.n_valid
        rms = _sqrt_fraction(self.total_sq.to_fraction(), self.n_valid)
        mean_value, mean_status = _float_metric(mean)
        rms_value, rms_status = _float_metric(rms)
        status = STATUS_OK if mean_status == STATUS_OK and rms_status == STATUS_OK else STATUS_NONFINITE
        return {"mean": mean_value, "rms": rms_value, "status": status, "n": self.n_valid}


def _metric(value, status: str) -> dict:
    return {"value": value, "status": status}


def _float_metric(value) -> tuple[float | None, str]:
    if value is None:
        return None, STATUS_NO_SAMPLES
    try:
        as_float = float(value)
    except OverflowError:
        return None, STATUS_NONFINITE
    if not math.isfinite(as_float):
        return None, STATUS_NONFINITE
    return as_float, STATUS_OK


def _ratio_to_float(numerator: Dyadic, denominator: Dyadic) -> float:
    try:
        return float(numerator.to_fraction() / denominator.to_fraction())
    except OverflowError:
        return math.inf


def _sqrt_fraction(numerator: Fraction, denominator: Fraction) -> float | None:
    if denominator == 0:
        return None
    try:
        ratio = float(numerator / denominator)
    except OverflowError:
        return math.inf
    if ratio < 0.0:
        return None
    return math.sqrt(ratio)
