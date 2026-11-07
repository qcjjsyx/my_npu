"""Output-quantisation comparison accumulators for the MXFP8 matrix experiment.

Two comparisons share one implementation:

* ``main_independent_scale``: ``Q_S`` and ``Q_R`` each select their own output
  group scale (the primary comparison);
* ``aux_shared_reference_scale``: both are quantised with the scale selected by
  ``Q_S`` (the auxiliary comparison that isolates element coding on a fixed
  grid).

Group validity is explicit: a group whose input contains a non-finite value or
a reserved scale code is counted invalid with its reason and never enters the
finite metrics.  Whole-batch metrics are ``null`` when any group is invalid; a
clearly labelled finite-subset payload is provided instead.

Three distinct mismatch notions are reported separately and never mixed:

* ``representation``: the pair ``(scale_code, element_code)``; a group whose two
  quantisations select different scales therefore has 32/32 representation
  mismatches even when some decoded values are equal or both zero.  The primary
  per-group histogram and the whole-group bit-match count use this definition
  (whole-group bit match == representation mismatch count 0);
* ``code_only``: element codes ignoring scale; every code-only field is
  explicitly named ``*_code_only_*`` and must not be used as the primary
  mismatch definition;
* ``decoded_value``: exact decoded dyadic values, independent of representation
  (``+0`` and ``-0`` decode to the same value).

``by_seed`` counts use the same three definitions plus a separately named
``ulp_mismatch_elements`` for the scale-equal ULP condition; the ULP-condition
count is never presented as the total mismatch count.

ULP distance uses the ordered finite E4M3 code sequence and is only computed
for elements whose group scales are equal (main comparison) or, for the shared
comparison, for all valid groups.  ``+0``/``-0`` are merged in numeric distance
but distinguished by code match.  Conditional ULP quantiles use nearest-rank
over mismatching elements; with no mismatch samples they are ``null`` with
status ``no_mismatch_samples`` (never 0).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.ulp import e4m3_ordered_index
from .bitmatch import nearest_rank
from .quantiles import linear_quantile

GROUP_MISMATCH_BINS = 33  # 0..32 mismatching elements per group

STATUS_OK = "ok"
STATUS_NO_SAMPLES = "no_samples"
STATUS_NO_VALID = "no_valid_groups"
STATUS_PARTIAL = "partial_invalid_groups"
STATUS_NO_MISMATCH = "no_mismatch_samples"
STATUS_ZERO_REFERENCE = "undefined_zero_reference"

ULP_CONDITION_INDEPENDENT = "finite elements in valid groups whose two group scales are equal"
ULP_CONDITION_SHARED = "finite elements in valid groups (shared scale, scales equal by construction)"

SUBSET_NOTE = "computed over valid groups only; not a whole-batch metric"


def _mean_of_histogram(histogram: list[int]) -> float | None:
    total = sum(histogram)
    if total == 0:
        return None
    weighted = sum(index * count for index, count in enumerate(histogram))
    return weighted / total


def _histogram_max(histogram: list[int]) -> int | None:
    return max((index for index, count in enumerate(histogram) if count > 0), default=None)


def _relative_l2(sum_err2: Dyadic, sum_ref2: Dyadic) -> tuple[float | None, str]:
    if sum_ref2.is_zero():
        if sum_err2.is_zero():
            return 0.0, STATUS_OK
        return None, STATUS_ZERO_REFERENCE
    if sum_err2.is_zero():
        return 0.0, STATUS_OK
    ratio = sum_err2.to_fraction() / sum_ref2.to_fraction()
    return math.sqrt(float(ratio)), STATUS_OK


def _metric(value, status: str) -> dict:
    return {"value": value, "status": status}


@dataclass
class MxComparisonAccumulator:
    name: str
    ulp_condition: str
    n_groups: int = 0
    n_valid_groups: int = 0
    n_invalid_groups: int = 0
    n_elements: int = 0
    n_valid_elements: int = 0
    n_invalid_elements: int = 0
    pair_ids: set = field(default_factory=set)
    n_scale_match: int = 0
    n_scale_mismatch: int = 0
    scale_diffs: list[int] = field(default_factory=list)
    n_group_bit_match: int = 0
    n_group_code_only_match: int = 0
    n_element_code_only_match: int = 0
    n_element_representation_match: int = 0
    n_decoded_match: int = 0
    n_signed_zero_mismatch: int = 0
    group_representation_mismatch_histogram: list[int] = field(
        default_factory=lambda: [0] * GROUP_MISMATCH_BINS
    )
    group_code_only_mismatch_histogram: list[int] = field(
        default_factory=lambda: [0] * GROUP_MISMATCH_BINS
    )
    abs_errors: list[float] = field(default_factory=list)
    sum_err2: Dyadic = field(default_factory=Dyadic.zero)
    sum_ref2: Dyadic = field(default_factory=Dyadic.zero)
    n_ulp_condition: int = 0
    n_ulp_mismatch: int = 0
    ulp_mismatch_distances: list[int] = field(default_factory=list)
    invalid_reasons: Counter = field(default_factory=Counter)
    by_seed: dict = field(default_factory=dict)

    @staticmethod
    def _seed_stats_entry() -> dict:
        return {
            "groups": 0,
            "invalid_groups": 0,
            "elements": 0,
            "scale_mismatch_groups": 0,
            "representation_mismatch_elements": 0,
            "code_only_mismatch_elements": 0,
            "decoded_mismatch_elements": 0,
            "ulp_mismatch_elements": 0,
        }

    def merge_from(self, other: "MxComparisonAccumulator") -> None:
        """Merge another accumulator of the same comparison."""
        if other.name != self.name:
            raise ValueError(f"cannot merge {other.name!r} into {self.name!r}")
        for attr in (
            "n_groups",
            "n_valid_groups",
            "n_invalid_groups",
            "n_elements",
            "n_valid_elements",
            "n_invalid_elements",
            "n_scale_match",
            "n_scale_mismatch",
            "n_group_bit_match",
            "n_group_code_only_match",
            "n_element_code_only_match",
            "n_element_representation_match",
            "n_decoded_match",
            "n_signed_zero_mismatch",
            "n_ulp_condition",
            "n_ulp_mismatch",
        ):
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        self.pair_ids = self.pair_ids | other.pair_ids
        self.scale_diffs.extend(other.scale_diffs)
        self.abs_errors.extend(other.abs_errors)
        self.ulp_mismatch_distances.extend(other.ulp_mismatch_distances)
        self.sum_err2 = self.sum_err2 + other.sum_err2
        self.sum_ref2 = self.sum_ref2 + other.sum_ref2
        for index in range(GROUP_MISMATCH_BINS):
            self.group_representation_mismatch_histogram[index] += (
                other.group_representation_mismatch_histogram[index]
            )
            self.group_code_only_mismatch_histogram[index] += other.group_code_only_mismatch_histogram[index]
        self.invalid_reasons.update(other.invalid_reasons)
        for seed, stats in other.by_seed.items():
            target = self.by_seed.setdefault(seed, {})
            for key, value in stats.items():
                target[key] = target.get(key, 0) + value

    def add_invalid_group(self, sample_id: str, seed: int, group_length: int, reason: str) -> None:
        self.n_groups += 1
        self.n_invalid_groups += 1
        self.n_elements += group_length
        self.n_invalid_elements += group_length
        self.pair_ids.add(sample_id)
        self.invalid_reasons[reason] += 1
        seed_stats = self.by_seed.setdefault(seed, self._seed_stats_entry())
        seed_stats["groups"] += 1
        seed_stats["invalid_groups"] += 1
        seed_stats["elements"] += group_length

    def add_group(
        self,
        sample_id: str,
        seed: int,
        scale_code_s: int,
        scale_code_r: int,
        codes_s: list[int],
        codes_r: list[int],
        decoded_s: list[Dyadic],
        decoded_r: list[Dyadic],
    ) -> None:
        if not (len(codes_s) == len(codes_r) == len(decoded_s) == len(decoded_r)):
            raise ValueError("group code/decoded length mismatch")
        group_length = len(codes_s)
        self.n_groups += 1
        self.n_valid_groups += 1
        self.n_elements += group_length
        self.n_valid_elements += group_length
        self.pair_ids.add(sample_id)
        seed_stats = self.by_seed.setdefault(seed, self._seed_stats_entry())
        seed_stats["groups"] += 1
        seed_stats["elements"] += group_length

        scale_match = scale_code_s == scale_code_r
        if scale_match:
            self.n_scale_match += 1
        else:
            self.n_scale_mismatch += 1
            seed_stats["scale_mismatch_groups"] += 1
        self.scale_diffs.append(scale_code_r - scale_code_s)

        representation_mismatch = 0
        code_only_mismatch = 0
        decoded_mismatch = 0
        for index in range(group_length):
            code_s = codes_s[index]
            code_r = codes_r[index]
            code_match = code_s == code_r
            # representation = (scale_code, element_code); different scales make
            # every element of the group a representation mismatch
            representation_match = scale_match and code_match
            if code_match:
                self.n_element_code_only_match += 1
            else:
                code_only_mismatch += 1
            if representation_match:
                self.n_element_representation_match += 1
            else:
                representation_mismatch += 1
            decoded_match = decoded_s[index] == decoded_r[index]
            if decoded_match:
                self.n_decoded_match += 1
            else:
                decoded_mismatch += 1

            error = decoded_r[index] - decoded_s[index]
            self.abs_errors.append(abs(error).to_float())
            self.sum_err2 = self.sum_err2 + error * error
            self.sum_ref2 = self.sum_ref2 + decoded_s[index] * decoded_s[index]

            if scale_match:
                self.n_ulp_condition += 1
                distance = abs(
                    e4m3_ordered_index(code_s) - e4m3_ordered_index(code_r)
                )
                if not code_match:
                    self.n_ulp_mismatch += 1
                    seed_stats["ulp_mismatch_elements"] += 1
                    self.ulp_mismatch_distances.append(distance)
                    if distance == 0:
                        self.n_signed_zero_mismatch += 1

        seed_stats["representation_mismatch_elements"] += representation_mismatch
        seed_stats["code_only_mismatch_elements"] += code_only_mismatch
        seed_stats["decoded_mismatch_elements"] += decoded_mismatch
        self.group_representation_mismatch_histogram[representation_mismatch] += 1
        self.group_code_only_mismatch_histogram[code_only_mismatch] += 1
        # bit match of the whole group == representation mismatch count is 0
        if representation_mismatch == 0:
            self.n_group_bit_match += 1
        if code_only_mismatch == 0:
            self.n_group_code_only_match += 1

    def finalize(self) -> dict:
        coverage = {
            "matrix_pairs": len(self.pair_ids),
            "groups": self.n_groups,
            "valid_groups": self.n_valid_groups,
            "invalid_groups": self.n_invalid_groups,
            "elements": self.n_elements,
            "valid_elements": self.n_valid_elements,
            "invalid_elements": self.n_invalid_elements,
        }
        if self.n_groups == 0:
            return self._payload(STATUS_NO_SAMPLES, coverage)
        if self.n_valid_groups == 0:
            return self._payload(STATUS_NO_VALID, coverage)
        subset = self._valid_metrics()
        if self.n_invalid_groups > 0:
            payload = self._payload(STATUS_PARTIAL, coverage)
            payload["finite_subset"] = {"note": SUBSET_NOTE, "coverage": coverage, **subset}
            return payload
        payload = self._payload(STATUS_OK, coverage)
        payload.update(subset)
        return payload

    def _payload(self, status: str, coverage: dict) -> dict:
        return {
            "comparison": self.name,
            "status": status,
            "coverage": coverage,
            "counts": self._counts(),
            "metrics": {key: _metric(None, status) for key in self._metric_keys()},
            "distributions": {
                "scale_exponent_diff": None,
                "group_representation_mismatch_elements": None,
                "group_code_only_mismatch_elements": None,
                "ulp": None,
            },
            "events": {"invalid_group_reasons": dict(self.invalid_reasons)},
            "by_seed": self._by_seed(),
        }

    def _metric_keys(self) -> tuple[str, ...]:
        return (
            "scale_mismatch_rate",
            "group_representation_match_rate",
            "group_representation_mismatch_rate",
            "group_code_only_mismatch_rate",
            "element_code_only_match_rate",
            "element_representation_match_rate",
            "decoded_match_rate",
            "decoded_abs_error_p50",
            "decoded_abs_error_p90",
            "decoded_abs_error_p99",
            "decoded_abs_error_max",
            "decoded_relative_l2",
            "scale_exponent_diff_p50",
            "scale_exponent_diff_p90",
            "scale_exponent_diff_p99",
            "ulp_conditional_p50",
            "ulp_conditional_p90",
            "ulp_conditional_p99",
            "ulp_conditional_max",
        )

    def _valid_metrics(self) -> dict:
        groups = self.n_valid_groups
        elements = self.n_valid_elements
        scale_rate = self.n_scale_mismatch / groups if groups else None
        group_representation_rate = self.n_group_bit_match / groups if groups else None
        group_code_only_rate = self.n_group_code_only_match / groups if groups else None
        abs_errors = sorted(self.abs_errors)
        max_abs = max((value for value in abs_errors if math.isfinite(value)), default=None)
        p50 = linear_quantile(abs_errors, 0.50)
        p90 = linear_quantile(abs_errors, 0.90)
        p99 = linear_quantile(abs_errors, 0.99)
        rel_l2, rel_l2_status = _relative_l2(self.sum_err2, self.sum_ref2)
        diffs = sorted(self.scale_diffs)
        ulp = sorted(self.ulp_mismatch_distances)
        if ulp:
            ulp_stats = {
                "condition": self.ulp_condition,
                "status": STATUS_OK,
                "n_condition_elements": self.n_ulp_condition,
                "n_mismatch": len(ulp),
                "n_signed_zero_mismatch": self.n_signed_zero_mismatch,
                "p50": nearest_rank(ulp, 0.50),
                "p90": nearest_rank(ulp, 0.90),
                "p99": nearest_rank(ulp, 0.99),
                "max": ulp[-1],
                "quantile_method": "nearest_rank",
            }
        else:
            ulp_stats = {
                "condition": self.ulp_condition,
                "status": STATUS_NO_MISMATCH,
                "n_condition_elements": self.n_ulp_condition,
                "n_mismatch": 0,
                "n_signed_zero_mismatch": 0,
                "p50": None,
                "p90": None,
                "p99": None,
                "max": None,
                "quantile_method": "nearest_rank",
            }
        metrics = {
            "scale_mismatch_rate": _metric(scale_rate, STATUS_OK if groups else STATUS_NO_SAMPLES),
            "group_representation_match_rate": _metric(
                group_representation_rate, STATUS_OK if groups else STATUS_NO_SAMPLES
            ),
            "group_representation_mismatch_rate": _metric(
                (1.0 - group_representation_rate) if group_representation_rate is not None else None,
                STATUS_OK if groups else STATUS_NO_SAMPLES,
            ),
            "group_code_only_mismatch_rate": _metric(
                (1.0 - group_code_only_rate) if group_code_only_rate is not None else None,
                STATUS_OK if groups else STATUS_NO_SAMPLES,
            ),
            "element_code_only_match_rate": _metric(
                (self.n_element_code_only_match / elements) if elements else None,
                STATUS_OK if elements else STATUS_NO_SAMPLES,
            ),
            "element_representation_match_rate": _metric(
                (self.n_element_representation_match / elements) if elements else None,
                STATUS_OK if elements else STATUS_NO_SAMPLES,
            ),
            "decoded_match_rate": _metric(
                (self.n_decoded_match / elements) if elements else None,
                STATUS_OK if elements else STATUS_NO_SAMPLES,
            ),
            "decoded_abs_error_p50": _metric(p50, STATUS_OK if p50 is not None else STATUS_NO_SAMPLES),
            "decoded_abs_error_p90": _metric(p90, STATUS_OK if p90 is not None else STATUS_NO_SAMPLES),
            "decoded_abs_error_p99": _metric(p99, STATUS_OK if p99 is not None else STATUS_NO_SAMPLES),
            "decoded_abs_error_max": _metric(max_abs, STATUS_OK if max_abs is not None else STATUS_NO_SAMPLES),
            "decoded_relative_l2": _metric(rel_l2, rel_l2_status),
            "scale_exponent_diff_p50": _metric(
                nearest_rank(diffs, 0.50) if diffs else None,
                STATUS_OK if diffs else STATUS_NO_SAMPLES,
            ),
            "scale_exponent_diff_p90": _metric(
                nearest_rank(diffs, 0.90) if diffs else None,
                STATUS_OK if diffs else STATUS_NO_SAMPLES,
            ),
            "scale_exponent_diff_p99": _metric(
                nearest_rank(diffs, 0.99) if diffs else None,
                STATUS_OK if diffs else STATUS_NO_SAMPLES,
            ),
            "ulp_conditional_p50": _metric(ulp_stats["p50"], ulp_stats["status"]),
            "ulp_conditional_p90": _metric(ulp_stats["p90"], ulp_stats["status"]),
            "ulp_conditional_p99": _metric(ulp_stats["p99"], ulp_stats["status"]),
            "ulp_conditional_max": _metric(ulp_stats["max"], ulp_stats["status"]),
        }
        diff_histogram: Counter = Counter(self.scale_diffs)
        lt = sum(1 for value in self.scale_diffs if value < 0)
        gt = sum(1 for value in self.scale_diffs if value > 0)
        eq = groups - lt - gt
        return {
            "metrics": metrics,
            "distributions": {
                "scale_exponent_diff": {
                    "histogram": {str(key): diff_histogram[key] for key in sorted(diff_histogram)},
                    "denominator": groups,
                    "negative": lt,
                    "zero": eq,
                    "positive": gt,
                    "negative_rate": (lt / groups) if groups else None,
                    "zero_rate": (eq / groups) if groups else None,
                    "positive_rate": (gt / groups) if groups else None,
                },
                "group_representation_mismatch_elements": {
                    "definition": "count of elements whose (scale_code, element_code) differs; a group with different scales is 32/32",
                    "histogram": {
                        str(index): self.group_representation_mismatch_histogram[index]
                        for index in range(GROUP_MISMATCH_BINS)
                    },
                    "denominator": groups,
                    "mean": _mean_of_histogram(self.group_representation_mismatch_histogram),
                    "max": _histogram_max(self.group_representation_mismatch_histogram),
                },
                "group_code_only_mismatch_elements": {
                    "definition": "count of elements whose element_code differs, ignoring scale (explicitly code-only)",
                    "histogram": {
                        str(index): self.group_code_only_mismatch_histogram[index]
                        for index in range(GROUP_MISMATCH_BINS)
                    },
                    "denominator": groups,
                    "mean": _mean_of_histogram(self.group_code_only_mismatch_histogram),
                    "max": _histogram_max(self.group_code_only_mismatch_histogram),
                },
                "ulp": ulp_stats,
            },
        }

    def _counts(self) -> dict:
        return {
            "n_scale_match": self.n_scale_match,
            "n_scale_mismatch": self.n_scale_mismatch,
            "n_group_bit_match": self.n_group_bit_match,
            "n_group_code_only_match": self.n_group_code_only_match,
            "n_element_code_only_match": self.n_element_code_only_match,
            "n_element_representation_match": self.n_element_representation_match,
            "n_decoded_match": self.n_decoded_match,
            "n_ulp_condition_elements": self.n_ulp_condition,
            "n_ulp_mismatch": self.n_ulp_mismatch,
            "n_signed_zero_mismatch": self.n_signed_zero_mismatch,
        }

    def _by_seed(self) -> dict:
        return {str(seed): self.by_seed[seed] for seed in sorted(self.by_seed)}


def main_accumulator() -> MxComparisonAccumulator:
    return MxComparisonAccumulator(
        name="main_independent_scale",
        ulp_condition=ULP_CONDITION_INDEPENDENT,
    )


def shared_accumulator() -> MxComparisonAccumulator:
    return MxComparisonAccumulator(
        name="aux_shared_reference_scale",
        ulp_condition=ULP_CONDITION_SHARED,
    )
