"""Bit-match / ULP / auxiliary error statistics for one comparison cell.

Rules (mirrors the experiment documentation):

* bit match compares the complete FP32 encoding and therefore distinguishes
  ``+0`` from ``-0``;
* ULP distance is computed on the ordered finite FP32 encodings with ``+0/-0``
  merged and only over pairs where both sides are finite;
* the conditional ULP distribution covers finite pairs whose bit patterns
  differ; with no such samples the quantiles are ``null`` and the status is
  ``no_mismatch_samples`` (never 0);
* non-finite results are counted separately and are never silently removed from
  the primary denominator;
* auxiliary absolute error / eta_L1 / relative L2 are exact dyadic sums; the
  order statistic lists (floats) are only used for the reported quantiles.

Cell metrics are derived from per-matrix-pair accumulators via exact-sum
``merge_from`` so a cell is not an average of per-pair ratios.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..arithmetic.ulp import fp32_ordered_index

BUCKET_LABELS = ("0", "1", "2", "3-4", ">=5")

EVENT_KEYS = (
    "both_finite",
    "reference_finite_candidate_nonfinite",
    "reference_nonfinite_candidate_finite",
    "both_nonfinite_same_sign",
    "both_nonfinite_opposite_sign",
    "any_nan",
)


def bucket_index(distance: int) -> int:
    if distance <= 0:
        return 0
    if distance == 1:
        return 1
    if distance == 2:
        return 2
    if distance <= 4:
        return 3
    return 4


def nearest_rank(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    rank = max(1, min(len(sorted_values), math.ceil(q * len(sorted_values))))
    return float(sorted_values[rank - 1])


def _quantiles(values: list[float], status_ok: str, status_empty: str, note: str) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {
            "status": status_empty,
            "note": note,
            "n": 0,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "quantile_method": "nearest_rank",
        }
    return {
        "status": status_ok,
        "note": note,
        "n": len(ordered),
        "p50": nearest_rank(ordered, 0.50),
        "p90": nearest_rank(ordered, 0.90),
        "p99": nearest_rank(ordered, 0.99),
        "max": ordered[-1],
        "quantile_method": "nearest_rank",
    }


@dataclass
class OutputComparison:
    n_total: int = 0
    n_bit_match: int = 0
    n_bit_mismatch: int = 0
    n_finite_pairs: int = 0
    n_finite_mismatch: int = 0
    signed_zero_mismatch: int = 0
    buckets: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    conditional_distances: list[int] = field(default_factory=list)
    events: Counter = field(default_factory=Counter)

    n_finite_candidate: int = 0
    n_nonfinite_candidate: int = 0
    abs_errors: list[float] = field(default_factory=list)
    eta_values: list[float] = field(default_factory=list)
    n_eta_zero_l1: int = 0
    sum_abs_error: Dyadic = field(default_factory=Dyadic.zero)
    sum_err2: Dyadic = field(default_factory=Dyadic.zero)
    sum_ref2: Dyadic = field(default_factory=Dyadic.zero)
    n_ref2_nonzero: int = 0

    def add(
        self,
        reference_bits: int,
        candidate_bits: int,
        reference_value: Dyadic,
        l1: Dyadic,
    ) -> None:
        reference = Fp32.from_bits(int(reference_bits))
        candidate = Fp32.from_bits(int(candidate_bits))
        self.n_total += 1
        reference_finite = reference.is_finite()
        candidate_finite = candidate.is_finite()
        if reference.is_nan() or candidate.is_nan():
            self.events["any_nan"] += 1
        elif reference_finite and candidate_finite:
            self.events["both_finite"] += 1
        elif reference_finite and not candidate_finite:
            self.events["reference_finite_candidate_nonfinite"] += 1
        elif not reference_finite and candidate_finite:
            self.events["reference_nonfinite_candidate_finite"] += 1
        elif reference.sign == candidate.sign:
            self.events["both_nonfinite_same_sign"] += 1
        else:
            self.events["both_nonfinite_opposite_sign"] += 1

        if reference.bits == candidate.bits:
            self.n_bit_match += 1
        else:
            self.n_bit_mismatch += 1

        if reference_finite and candidate_finite:
            self.n_finite_pairs += 1
            distance = abs(fp32_ordered_index(reference.bits) - fp32_ordered_index(candidate.bits))
            self.buckets[bucket_index(distance)] += 1
            if reference.bits != candidate.bits:
                self.n_finite_mismatch += 1
                self.conditional_distances.append(distance)
                if distance == 0:
                    self.signed_zero_mismatch += 1

        if not candidate_finite:
            self.n_nonfinite_candidate += 1
            return

        self.n_finite_candidate += 1
        candidate_value = candidate.exact_dyadic()
        error = candidate_value - reference_value
        magnitude = abs(error)
        self.sum_abs_error = self.sum_abs_error + magnitude
        self.sum_err2 = self.sum_err2 + error * error
        squared_reference = reference_value * reference_value
        if not squared_reference.is_zero():
            self.n_ref2_nonzero += 1
        self.sum_ref2 = self.sum_ref2 + squared_reference
        self.abs_errors.append(magnitude.to_float())
        if l1.is_zero():
            self.n_eta_zero_l1 += 1
        else:
            self.eta_values.append(float(magnitude.to_fraction() / l1.to_fraction()))

    def merge_from(self, other: "OutputComparison") -> None:
        self.n_total += other.n_total
        self.n_bit_match += other.n_bit_match
        self.n_bit_mismatch += other.n_bit_mismatch
        self.n_finite_pairs += other.n_finite_pairs
        self.n_finite_mismatch += other.n_finite_mismatch
        self.signed_zero_mismatch += other.signed_zero_mismatch
        for index in range(len(BUCKET_LABELS)):
            self.buckets[index] += other.buckets[index]
        self.conditional_distances.extend(other.conditional_distances)
        self.events.update(other.events)
        self.n_finite_candidate += other.n_finite_candidate
        self.n_nonfinite_candidate += other.n_nonfinite_candidate
        self.abs_errors.extend(other.abs_errors)
        self.eta_values.extend(other.eta_values)
        self.n_eta_zero_l1 += other.n_eta_zero_l1
        self.sum_abs_error = self.sum_abs_error + other.sum_abs_error
        self.sum_err2 = self.sum_err2 + other.sum_err2
        self.sum_ref2 = self.sum_ref2 + other.sum_ref2
        self.n_ref2_nonzero += other.n_ref2_nonzero

    def finalize(self) -> dict:
        conditional = _quantiles(
            [float(value) for value in sorted(self.conditional_distances)],
            "ok",
            "no_mismatch_samples",
            "finite pairs with bit mismatch only",
        )
        if conditional["status"] == "ok":
            conditional["max"] = int(conditional["max"])
            for key in ("p50", "p90", "p99"):
                conditional[key] = int(conditional[key])
        buckets = {}
        for label, count in zip(BUCKET_LABELS, self.buckets):
            buckets[label] = {
                "count": count,
                "rate": (count / self.n_finite_pairs) if self.n_finite_pairs else None,
            }
        abs_error = _quantiles(
            self.abs_errors,
            "ok",
            "no_finite_candidate",
            "finite candidate samples; |decode(candidate) - S|",
        )
        if self.abs_errors:
            mean_abs = float(self.sum_abs_error.to_fraction() / self.n_finite_candidate)
        else:
            mean_abs = None
        abs_error["mean"] = mean_abs
        eta = _quantiles(
            self.eta_values,
            "ok",
            "no_eta_samples",
            "finite candidate samples with L1 != 0",
        )
        eta["n_zero_l1"] = self.n_eta_zero_l1
        if self.sum_ref2.is_zero():
            relative_l2 = {
                "status": "undefined_zero_reference",
                "note": "sum of squared exact reference values is zero over the finite-candidate set",
                "value": None,
                "n": self.n_finite_candidate,
                "n_reference_nonzero": self.n_ref2_nonzero,
            }
        elif self.n_finite_candidate == 0:
            relative_l2 = {
                "status": "no_finite_candidate",
                "note": "no finite candidate sample",
                "value": None,
                "n": 0,
                "n_reference_nonzero": self.n_ref2_nonzero,
            }
        else:
            ratio = self.sum_err2.to_fraction() / self.sum_ref2.to_fraction()
            relative_l2 = {
                "status": "ok",
                "note": "sqrt(sum(err^2) / sum(S^2)) over finite candidate samples",
                "value": math.sqrt(float(ratio)),
                "n": self.n_finite_candidate,
                "n_reference_nonzero": self.n_ref2_nonzero,
            }
        return {
            "counts": {
                "n_total": self.n_total,
                "n_bit_match": self.n_bit_match,
                "n_bit_mismatch": self.n_bit_mismatch,
                "n_finite_pairs": self.n_finite_pairs,
                "n_finite_mismatch": self.n_finite_mismatch,
                "signed_zero_mismatch": self.signed_zero_mismatch,
                "n_finite_candidate": self.n_finite_candidate,
                "n_nonfinite_candidate": self.n_nonfinite_candidate,
            },
            "bit_match_rate": (self.n_bit_match / self.n_total) if self.n_total else None,
            "bit_mismatch_rate": (self.n_bit_mismatch / self.n_total) if self.n_total else None,
            "conditional_ulp": conditional,
            "distance_buckets": {
                "denominator": "n_finite_pairs",
                "n_finite_pairs": self.n_finite_pairs,
                "buckets": buckets,
            },
            "events": {key: int(self.events.get(key, 0)) for key in EVENT_KEYS},
            "aux": {
                "abs_error": abs_error,
                "eta_l1": eta,
                "relative_l2": relative_l2,
                "nonfinite_note": (
                    "non-finite candidates are excluded from the aux set but counted in n_total "
                    "and in the event counters"
                ),
            },
        }


class MetricsCollector:
    """Per-cell and per-matrix-pair accumulators with exact-sum cell merging."""

    def __init__(self) -> None:
        self.cells: dict[tuple, OutputComparison] = {}
        self.pairs: dict[tuple, OutputComparison] = {}
        self._pair_cells: dict[tuple, tuple] = {}

    def add(
        self,
        cell_key: tuple,
        pair_key: tuple,
        reference_bits: list[int],
        candidate_bits: list[int],
        reference_values: list[Dyadic],
        l1_values: list[Dyadic],
        seed: int,
    ) -> None:
        if not (
            len(reference_bits)
            == len(candidate_bits)
            == len(reference_values)
            == len(l1_values)
        ):
            raise ValueError("comparison array length mismatch")
        accumulator = self.pairs.get(pair_key)
        if accumulator is None:
            accumulator = OutputComparison()
            self.pairs[pair_key] = accumulator
            self._pair_cells[pair_key] = cell_key
        for index in range(len(reference_bits)):
            accumulator.add(
                reference_bits[index],
                candidate_bits[index],
                reference_values[index],
                l1_values[index],
            )

    def finalize(self) -> dict:
        for pair_key, accumulator in self.pairs.items():
            cell_key = self._pair_cells[pair_key]
            cell = self.cells.get(cell_key)
            if cell is None:
                cell = OutputComparison()
                self.cells[cell_key] = cell
            cell.merge_from(accumulator)
        return {
            "cells": {key: value.finalize() for key, value in self.cells.items()},
            "by_pair": {key: value.finalize() for key, value in self.pairs.items()},
        }


@dataclass
class ScalarErrorAccumulator:
    """Exact-dyadic error statistics for a scalar error series (e.g. T - S)."""

    errors: list[float] = field(default_factory=list)
    eta_values: list[float] = field(default_factory=list)
    n: int = 0
    n_zero_l1: int = 0
    sum_abs_error: Dyadic = field(default_factory=Dyadic.zero)
    sum_err2: Dyadic = field(default_factory=Dyadic.zero)
    sum_ref2: Dyadic = field(default_factory=Dyadic.zero)
    n_ref2_nonzero: int = 0

    def add(self, error: Dyadic, reference: Dyadic, l1: Dyadic) -> None:
        self.n += 1
        magnitude = abs(error)
        self.sum_abs_error = self.sum_abs_error + magnitude
        self.sum_err2 = self.sum_err2 + error * error
        squared_reference = reference * reference
        if not squared_reference.is_zero():
            self.n_ref2_nonzero += 1
        self.sum_ref2 = self.sum_ref2 + squared_reference
        self.errors.append(magnitude.to_float())
        if l1.is_zero():
            self.n_zero_l1 += 1
        else:
            self.eta_values.append(float(magnitude.to_fraction() / l1.to_fraction()))

    def merge_from(self, other: "ScalarErrorAccumulator") -> None:
        self.n += other.n
        self.n_zero_l1 += other.n_zero_l1
        self.errors.extend(other.errors)
        self.eta_values.extend(other.eta_values)
        self.sum_abs_error = self.sum_abs_error + other.sum_abs_error
        self.sum_err2 = self.sum_err2 + other.sum_err2
        self.sum_ref2 = self.sum_ref2 + other.sum_ref2
        self.n_ref2_nonzero += other.n_ref2_nonzero

    def finalize(self) -> dict:
        abs_error = _quantiles(
            self.errors, "ok", "no_samples", "exact error series; |error|"
        )
        abs_error["mean"] = (
            float(self.sum_abs_error.to_fraction() / self.n) if self.n else None
        )
        eta = _quantiles(
            self.eta_values, "ok", "no_eta_samples", "exact error series with L1 != 0"
        )
        eta["n_zero_l1"] = self.n_zero_l1
        if self.n == 0:
            relative_l2 = {
                "status": "no_samples",
                "value": None,
                "n": 0,
                "note": "sqrt(sum(err^2)/sum(S^2))",
            }
        elif self.sum_ref2.is_zero():
            relative_l2 = {
                "status": "undefined_zero_reference",
                "value": None,
                "n": self.n,
                "note": "sqrt(sum(err^2)/sum(S^2))",
            }
        else:
            ratio = self.sum_err2.to_fraction() / self.sum_ref2.to_fraction()
            relative_l2 = {
                "status": "ok",
                "value": math.sqrt(float(ratio)),
                "n": self.n,
                "note": "sqrt(sum(err^2)/sum(S^2))",
            }
        return {"abs_error": abs_error, "eta_l1": eta, "relative_l2": relative_l2}


class ScalarErrorCollector:
    """Per-cell and per-pair scalar error accumulation with exact-sum merging."""

    def __init__(self) -> None:
        self.pairs: dict[tuple, ScalarErrorAccumulator] = {}
        self._pair_cells: dict[tuple, tuple] = {}

    def add(self, cell_key: tuple, pair_key: tuple, errors, references, l1_values) -> None:
        if not (len(errors) == len(references) == len(l1_values)):
            raise ValueError("scalar error array length mismatch")
        accumulator = self.pairs.get(pair_key)
        if accumulator is None:
            accumulator = ScalarErrorAccumulator()
            self.pairs[pair_key] = accumulator
            self._pair_cells[pair_key] = cell_key
        for index in range(len(errors)):
            accumulator.add(errors[index], references[index], l1_values[index])

    def finalize(self) -> dict:
        cells: dict[tuple, ScalarErrorAccumulator] = {}
        for pair_key, accumulator in self.pairs.items():
            cell_key = self._pair_cells[pair_key]
            cell = cells.get(cell_key)
            if cell is None:
                cell = ScalarErrorAccumulator()
                cells[cell_key] = cell
            cell.merge_from(accumulator)
        return {
            "cells": {key: value.finalize() for key, value in cells.items()},
            "by_pair": {key: value.finalize() for key, value in self.pairs.items()},
        }
