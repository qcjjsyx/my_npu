"""Offline analysis for the fixed-W_align K/B sweep.

Reads only the saved alignment records (never the samplers or the execution
engine) and produces:

* exact-reference comparisons ``T-S`` (pure alignment accumulation error),
  ``R-S`` (legacy FP32 no-window error), ``A-T`` (candidate FP32 accumulation
  error), ``A-S`` (candidate total error before output quantisation) and
  ``A-R`` (final FP32 change caused by the alignment mechanism);
* output MX quantisation comparisons for the pairs ``A-S``, ``A-R`` and
  ``R-S``, independently scaled and with the shared reference scale selected by
  ``Q_S``;
* total output errors ``decode(Q_A)-S``, ``decode(Q_R)-S`` and the ideal
  ``decode(Q_S)-S``;
* representation transition counts between the baseline (``Q_R`` vs ``Q_S``)
  and the candidate (``Q_A`` vs ``Q_S``);
* cumulative alignment diagnostics (products/block counters, ``Cmax-C_i``
  histogram, per-element accumulated/maximum alignment error magnitudes) and a
  per-pair summary so paired comparisons stay possible.

Every cumulative counter/histogram is the value at that snapshot K (covering
all blocks up to K); quantities must never be summed across K.  The
``A-S = (T-S) + (A-T)`` identity is verified exactly for every finite element
and a violation aborts the analysis instead of reporting success.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..formats.e4m3_element import is_nan_code
from ..formats.e8m0 import NAN_CODE as E8M0_NAN_CODE
from ..matrix.quantize import (
    GROUP_LENGTH,
    decoded_group_values,
    group_index_lists,
    quantize_matrix_independent,
    quantize_matrix_shared,
)
from ..matrix.tensor import MxTensor
from ..metrics.compare import ComparisonAccumulator
from ..metrics.mx_quant import (
    MxComparisonAccumulator,
    ULP_CONDITION_INDEPENDENT,
    ULP_CONDITION_SHARED,
)
from ..version import (
    MX_ALIGN_CONTAINER_EXTRA_BITS,
    MX_ALIGN_POLICY,
    MX_ALIGN_SCHEMA_VERSION,
    MX_INPUT_QUANTIZATION_POLICY,
    MX_INPUT_SAMPLING_VERSION,
    MX_INPUT_STREAM_VERSION,
    PACKAGE_VERSION,
)
from .enhance import _file_identity
from .mx_io import MX_INPUTS_NAME, MX_SAMPLES_NAME, iter_mx_inputs, iter_mx_samples
from .reader import load_manifest, require_schema, run_path
from .schema import (
    MX_ALIGN_ANALYSIS_VERSION,
    MX_ALIGN_SUMMARY_VERSION,
    source_digest as analyzer_source_digest,
)
from .writer import write_json

SOURCE_FILES = ("manifest.json", MX_INPUTS_NAME, MX_SAMPLES_NAME)

COMPARISONS = (
    "align_T_minus_S",
    "baseline_R_minus_S",
    "candidate_A_minus_T",
    "candidate_A_minus_S",
    "candidate_A_minus_R",
)

ALIGN_EVENT_KEYS = (
    "products_total",
    "products_nonzero",
    "products_changed",
    "products_zeroed",
    "dot16_total",
    "dot16_all_zero",
    "dot16_with_changed_product",
    "dot16_qb_diff",
    "r_updates",
    "r_nonfinite_updates",
    "r_overflow_updates",
    "r_absorbed_updates",
    "a_updates",
    "a_nonfinite_updates",
    "a_overflow_updates",
    "a_absorbed_updates",
)

MX_COMPARISON_NAMES = (
    "mx_A_vs_S_independent",
    "mx_A_vs_R_independent",
    "mx_R_vs_S_independent",
    "mx_A_vs_S_shared",
    "mx_A_vs_R_shared",
    "mx_R_vs_S_shared",
)

TRANSITION_GROUP_KEYS = (
    "both_match",
    "baseline_match_candidate_mismatch",
    "baseline_mismatch_candidate_match",
    "both_mismatch_candidate_differs",
    "both_mismatch_candidate_same",
)

POLICY = {
    "alignment": MX_ALIGN_POLICY,
    "alignment_formula": (
        "n_i = RNE(P_i * 2**(w_align - 6 - (Cmax - C_i))); N_b = sum n_i; "
        "q_b = N_b * 2**(Cmax - 14 - w_align + h); container = signed (w_align+7)-bit"
    ),
    "comparisons": {
        "align_T_minus_S": "error T - S, reference S (pure alignment accumulation error)",
        "baseline_R_minus_S": "error R - S, reference S (legacy no-window FP32 error)",
        "candidate_A_minus_T": "error A - T, reference T (candidate FP32 accumulation error)",
        "candidate_A_minus_S": "error A - S, reference S (total error before output quantisation)",
        "candidate_A_minus_R": "error A - R, reference R (final FP32 change from the alignment mechanism)",
        "decode_QA_minus_S": "error decode(Q_A) - S, reference S",
        "decode_QR_minus_S": "error decode(Q_R) - S, reference S",
        "decode_QS_minus_S": "error decode(Q_S) - S, reference S (ideal output quantisation)",
    },
    "denominators": {
        "etaL1": "the same output's exact MX input product L1 from the snapshot",
        "T-S/R-S/A-S": "reference S",
        "A-T": "reference T",
        "A-R": "reference R",
        "decoded MX differences": "the compared MX reference output",
    },
    "mx_quantization": {
        "independent": "Q_A/Q_R/Q_S each select their own output group scale",
        "shared": "all three use the Q_S reference scale, never reselected",
        "representation": "(scale_code, element_code); a group with different scales is 32/32 mismatched",
        "code_only": "element codes only, explicitly named *_code_only_*",
        "ulp": "ordered finite E4M3 steps; independent comparison only in scale-equal groups, shared comparison in valid groups; nearest-rank conditional quantiles, null when empty",
    },
    "identity": "A-S = (T-S) + (A-T) is verified exactly per finite element; a violation aborts the analysis",
    "independence": "the independent repeat unit is the input matrix pair; elements and K prefixes are correlated, no 3/N zero-event bound is reported",
}

NOTES = [
    "offline analysis only: the samplers and the execution engine were not imported or executed",
    "source runs are read-only; results go to an independent directory",
    "S/L1 use decoded MX inputs; T is the exact sum of aligned partial sums; A is the FP32 accumulation of those partial sums",
    "cumulative alignment counters/histograms are snapshot values covering all blocks up to K; never sum them across K",
    "input diagnostics are diagnostics, not error metrics of this experiment",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _metric(value, status: str) -> dict:
    return {"value": value, "status": status}


def _new_comparisons() -> dict[str, ComparisonAccumulator]:
    return {name: ComparisonAccumulator(name) for name in COMPARISONS}


def _new_total_errors() -> dict[str, ComparisonAccumulator]:
    return {
        name: ComparisonAccumulator(name)
        for name in ("decode_QA_minus_S", "decode_QR_minus_S", "decode_QS_minus_S")
    }


def _new_mx_comparisons() -> dict[str, MxComparisonAccumulator]:
    return {
        name: MxComparisonAccumulator(
            name=name,
            ulp_condition=ULP_CONDITION_INDEPENDENT
            if name.endswith("_independent")
            else ULP_CONDITION_SHARED,
        )
        for name in MX_COMPARISON_NAMES
    }


@dataclass
class PairSummary:
    """Per-matrix-pair accumulators inside one (distribution, weight, K) cell.

    Counters/metrics are kept per pair so paired comparisons stay possible; the
    cell payload is produced by merging these accumulators (exact sums), never
    by averaging per-pair ratios.
    """

    sample_id: str
    seed: int
    pair_index: int
    M: int
    N: int
    valid: bool = True
    invalid_reason: str | None = None
    comparisons: dict = field(default_factory=_new_comparisons)
    total_errors: dict = field(default_factory=_new_total_errors)
    mx: dict = field(default_factory=_new_mx_comparisons)
    transition_groups: Counter = field(default_factory=Counter)
    transition_elements: Counter = field(default_factory=Counter)
    transition_valid_groups: int = 0
    transition_invalid_groups: int = 0
    transition_valid_elements: int = 0
    transition_invalid_elements: int = 0
    identity_violations: int = 0
    quantization_events_by_path: dict = field(default_factory=dict)
    events_cumulative: Counter = field(default_factory=Counter)
    cmax_histogram: Counter = field(default_factory=Counter)
    align_abs_max_max: Dyadic = field(default_factory=Dyadic.zero)
    t_equals_s: bool = True
    a_equals_r: bool = True


@dataclass
class AlignedCell:
    activation_distribution: str
    weight_distribution: str
    K: int
    B: int
    M: int = 0
    N: int = 0
    w_align: int | None = None
    container_bits: int | None = None
    pairs: dict = field(default_factory=dict)
    n_invalid_input_snapshots: int = 0
    invalid_input_reasons: Counter = field(default_factory=Counter)

    def _pair(self, sample_id: str, seed: int, pair_index: int, M: int, N: int) -> PairSummary:
        pair = self.pairs.get(sample_id)
        if pair is None:
            pair = PairSummary(
                sample_id=sample_id, seed=seed, pair_index=pair_index, M=M, N=N
            )
            self.pairs[sample_id] = pair
        elif pair.seed != seed or pair.pair_index != pair_index or pair.M != M or pair.N != N:
            raise ValueError(
                f"conflicting pair identity for {sample_id} inside cell "
                f"({self.activation_distribution}/{self.weight_distribution}/K={self.K})"
            )
        return pair

    def add(self, record: dict, pair_valid: bool, pair_reason: str | None, input_meta: dict) -> None:
        M = int(record["M"])
        N = int(record["N"])
        if self.M and (self.M != M or self.N != N):
            raise ValueError(f"inconsistent M/N inside cell {self.activation_distribution}/{self.weight_distribution}/K={self.K}")
        self.M = M
        self.N = N
        self.w_align = int(record["w_align"])
        self.container_bits = int(record["container_bits"])
        sample_id = record["sample_id"]
        seed = int(record["seed"])
        pair_index = int(record["pair_index"])
        pair = self._pair(sample_id, seed, pair_index, M, N)
        pair.valid = pair.valid and pair_valid
        if not pair_valid:
            pair.invalid_reason = pair_reason or "invalid_input"
        if not pair_valid:
            self.n_invalid_input_snapshots += 1
            self.invalid_input_reasons[pair_reason or "invalid_input"] += 1
            self._count_invalid_pair(pair)
            return
        self._accumulate_valid_pair(pair, record, M, N)

    def _count_invalid_pair(self, pair: PairSummary) -> None:
        """Numerically invalid input: count invalid elements/groups in every comparison.

        T-S, R-S, A-T, A-S, A-R and the total/decode comparisons each own their
        coverage; transition coverage records valid/invalid/total separately.
        """
        elements = pair.M * pair.N
        groups = elements // GROUP_LENGTH
        for accumulator in pair.comparisons.values():
            for _ in range(elements):
                accumulator.add_invalid()
        for accumulator in pair.total_errors.values():
            for _ in range(elements):
                accumulator.add_invalid()
        for accumulator in pair.mx.values():
            for _ in range(groups):
                accumulator.add_invalid_group(
                    pair.sample_id, pair.seed, GROUP_LENGTH, pair.invalid_reason or "invalid_input"
                )
        pair.transition_invalid_groups += groups
        pair.transition_invalid_elements += elements

    def _accumulate_valid_pair(self, pair: PairSummary, record: dict, M: int, N: int) -> None:
        values_s = [Dyadic.from_dict(payload) for payload in record["S"]]
        values_t = [Dyadic.from_dict(payload) for payload in record["T"]]
        values_l1 = [Dyadic.from_dict(payload) for payload in record["L1"]]
        r_values: list[Dyadic | None] = []
        r_signs: list[int] = []
        a_values: list[Dyadic | None] = []
        a_signs: list[int] = []
        for bits in record["r_bits"]:
            state = Fp32.from_bits(int(bits))
            if state.is_finite():
                r_values.append(state.exact_dyadic())
                r_signs.append(state.sign if state.is_zero() else 0)
            else:
                r_values.append(None)
                r_signs.append(0)
        for bits in record["a_bits"]:
            state = Fp32.from_bits(int(bits))
            if state.is_finite():
                a_values.append(state.exact_dyadic())
                a_signs.append(state.sign if state.is_zero() else 0)
            else:
                a_values.append(None)
                a_signs.append(0)

        for index in range(M * N):
            s_value = values_s[index]
            t_value = values_t[index]
            l1_value = values_l1[index]
            pair.comparisons["align_T_minus_S"].add(t_value - s_value, s_value, l1_value)
            if r_values[index] is None:
                pair.comparisons["baseline_R_minus_S"].add_invalid()
            else:
                pair.comparisons["baseline_R_minus_S"].add(
                    r_values[index] - s_value, s_value, l1_value
                )
            if a_values[index] is None:
                pair.comparisons["candidate_A_minus_T"].add_invalid()
                pair.comparisons["candidate_A_minus_S"].add_invalid()
                pair.comparisons["candidate_A_minus_R"].add_invalid()
            else:
                pair.comparisons["candidate_A_minus_T"].add(
                    a_values[index] - t_value, t_value, l1_value
                )
                pair.comparisons["candidate_A_minus_S"].add(
                    a_values[index] - s_value, s_value, l1_value
                )
                if r_values[index] is None:
                    pair.comparisons["candidate_A_minus_R"].add_invalid()
                else:
                    pair.comparisons["candidate_A_minus_R"].add(
                        a_values[index] - r_values[index], r_values[index], l1_value
                    )
                lhs = a_values[index] - s_value
                rhs = (t_value - s_value) + (a_values[index] - t_value)
                if lhs != rhs:
                    pair.identity_violations += 1

        q_s = quantize_matrix_independent(values_s, None, M, N)
        q_r = quantize_matrix_independent(r_values, r_signs, M, N)
        q_a = quantize_matrix_independent(a_values, a_signs, M, N)
        reference_scales = list(q_s.scale_codes)
        aux_s = quantize_matrix_shared(values_s, None, M, N, reference_scales)
        aux_r = quantize_matrix_shared(r_values, r_signs, M, N, reference_scales)
        aux_a = quantize_matrix_shared(a_values, a_signs, M, N, reference_scales)

        positions = group_index_lists(M, N, GROUP_LENGTH, 1)
        for group_index, indices in enumerate(positions):
            group_s = q_s.groups[group_index]
            group_r = q_r.groups[group_index]
            group_a = q_a.groups[group_index]
            self._mx_pair(pair, "mx_A_vs_S_independent", group_a, group_s)
            self._mx_pair(pair, "mx_A_vs_R_independent", group_a, group_r)
            self._mx_pair(pair, "mx_R_vs_S_independent", group_r, group_s)
            self._mx_pair(pair, "mx_A_vs_S_shared", aux_a.groups[group_index], aux_s.groups[group_index])
            self._mx_pair(pair, "mx_A_vs_R_shared", aux_a.groups[group_index], aux_r.groups[group_index])
            self._mx_pair(pair, "mx_R_vs_S_shared", aux_r.groups[group_index], aux_s.groups[group_index])
            self._transitions(pair, group_s, group_r, group_a, len(indices))

        for group_index, indices in enumerate(positions):
            group_a = q_a.groups[group_index]
            group_r = q_r.groups[group_index]
            group_s = q_s.groups[group_index]
            if group_a.invalid:
                for _ in indices:
                    pair.total_errors["decode_QA_minus_S"].add_invalid()
            else:
                decoded = decoded_group_values(group_a)
                for offset, index in enumerate(indices):
                    pair.total_errors["decode_QA_minus_S"].add(
                        decoded[offset] - values_s[index], values_s[index], values_l1[index]
                    )
            if group_r.invalid:
                for _ in indices:
                    pair.total_errors["decode_QR_minus_S"].add_invalid()
            else:
                decoded = decoded_group_values(group_r)
                for offset, index in enumerate(indices):
                    pair.total_errors["decode_QR_minus_S"].add(
                        decoded[offset] - values_s[index], values_s[index], values_l1[index]
                    )
            decoded_s = decoded_group_values(group_s)
            for offset, index in enumerate(indices):
                pair.total_errors["decode_QS_minus_S"].add(
                    decoded_s[offset] - values_s[index], values_s[index], values_l1[index]
                )

        for path, quant in (
            ("Q_S_independent", q_s),
            ("Q_R_independent", q_r),
            ("Q_A_independent", q_a),
            ("Q_S_shared", aux_s),
            ("Q_R_shared", aux_r),
            ("Q_A_shared", aux_a),
        ):
            path_events = pair.quantization_events_by_path.setdefault(path, Counter())
            path_events.update(quant.events)

        events = record.get("events_cumulative") or {}
        pair.events_cumulative.update(events)
        for key, count in (record.get("cmax_minus_ci_histogram_cumulative") or {}).items():
            pair.cmax_histogram[key] += int(count)
        for payload in record.get("align_abs_max") or []:
            value = Dyadic.from_dict(payload)
            if pair.align_abs_max_max < value:
                pair.align_abs_max_max = value
        pair.t_equals_s = all(values_t[i] == values_s[i] for i in range(M * N))
        pair.a_equals_r = all(
            a_values[i] is not None and r_values[i] is not None and a_values[i] == r_values[i]
            for i in range(M * N)
        )

    def _mx_pair(self, pair: PairSummary, name, candidate, reference) -> None:
        accumulator = pair.mx[name]
        if candidate.invalid or reference.invalid:
            reason = candidate.invalid_reason or reference.invalid_reason or "invalid"
            accumulator.add_invalid_group(pair.sample_id, pair.seed, GROUP_LENGTH, reason)
            return
        accumulator.add_group(
            pair.sample_id,
            pair.seed,
            reference.scale_code,
            candidate.scale_code,
            reference.element_codes,
            candidate.element_codes,
            decoded_group_values(reference),
            decoded_group_values(candidate),
        )

    def _transitions(self, pair: PairSummary, group_s, group_r, group_a, group_length) -> None:
        if group_s.invalid or group_r.invalid or group_a.invalid:
            pair.transition_invalid_groups += 1
            pair.transition_invalid_elements += group_length
            return
        r_rep_match = group_r.scale_code == group_s.scale_code and group_r.element_codes == group_s.element_codes
        a_rep_match = group_a.scale_code == group_s.scale_code and group_a.element_codes == group_s.element_codes
        ar_same = group_a.scale_code == group_r.scale_code and group_a.element_codes == group_r.element_codes
        pair.transition_groups[_transition_key(r_rep_match, a_rep_match, ar_same)] += 1
        pair.transition_valid_groups += 1
        pair.transition_valid_elements += group_length
        for index in range(group_length):
            element_r_match = (
                group_r.scale_code == group_s.scale_code
                and group_r.element_codes[index] == group_s.element_codes[index]
            )
            element_a_match = (
                group_a.scale_code == group_s.scale_code
                and group_a.element_codes[index] == group_s.element_codes[index]
            )
            element_ar_same = (
                group_a.scale_code == group_r.scale_code
                and group_a.element_codes[index] == group_r.element_codes[index]
            )
            pair.transition_elements[
                _transition_key(element_r_match, element_a_match, element_ar_same)
            ] += 1

    def _pair_compact(self, pair: PairSummary) -> dict:
        transitions = {
            "groups": _transition_payload(
                pair.transition_groups,
                pair.transition_valid_groups,
                pair.transition_invalid_groups,
            ),
            "elements": _transition_payload(
                pair.transition_elements,
                pair.transition_valid_elements,
                pair.transition_invalid_elements,
            ),
        }

        def compact(accumulator: ComparisonAccumulator) -> dict:
            payload = accumulator.finalize()
            return {
                "status": payload["status"],
                "coverage": payload["coverage"],
                "counts": payload["counts"],
                "metrics": payload["metrics"],
                "reference_rms": payload["reference_rms"],
                "sum_err2": accumulator.sum_err2.to_dict(),
                "sum_ref2": accumulator.sum_ref2.to_dict(),
                "finite_subset": payload.get("finite_subset"),
            }

        def compact_mx(accumulator: MxComparisonAccumulator) -> dict:
            payload = accumulator.finalize()
            distributions = payload.get("distributions") or {}
            ulp = distributions.get("ulp")
            return {
                "status": payload["status"],
                "coverage": payload["coverage"],
                "counts": payload["counts"],
                "metrics": payload["metrics"],
                "sum_err2": accumulator.sum_err2.to_dict(),
                "sum_ref2": accumulator.sum_ref2.to_dict(),
                "ulp": None
                if ulp is None
                else {
                    "status": ulp["status"],
                    "n_condition_elements": ulp["n_condition_elements"],
                    "n_mismatch": ulp["n_mismatch"],
                    "n_signed_zero_mismatch": ulp["n_signed_zero_mismatch"],
                    "p50": ulp["p50"],
                    "p90": ulp["p90"],
                    "p99": ulp["p99"],
                    "max": ulp["max"],
                },
                "finite_subset": payload.get("finite_subset"),
            }

        return {
            "sample_id": pair.sample_id,
            "seed": pair.seed,
            "pair_index": pair.pair_index,
            "valid": pair.valid,
            "invalid_reason": pair.invalid_reason,
            "cells": _cell_counts(pair.M, pair.N),
            "comparisons": {name: compact(acc) for name, acc in pair.comparisons.items()},
            "total_errors": {name: compact(acc) for name, acc in pair.total_errors.items()},
            "mx_comparisons": {name: compact_mx(acc) for name, acc in pair.mx.items()},
            "transitions": transitions,
            "identity_violations": pair.identity_violations,
            "quantization_events_by_path": {
                path: {key: int(value) for key, value in sorted(events.items())}
                for path, events in sorted(pair.quantization_events_by_path.items())
            },
            "alignment_diagnostics": {
                "events_cumulative": {
                    key: int(pair.events_cumulative.get(key, 0)) for key in ALIGN_EVENT_KEYS
                },
                "cumulative_note": "counters cover all blocks up to this K; never sum across K",
            },
            "align_abs_max_max_over_elements": pair.align_abs_max_max.to_dict(),
            "t_equals_s": pair.t_equals_s,
            "a_equals_r": pair.a_equals_r,
        }

    def finalize(self) -> dict:
        ordered = sorted(self.pairs.values(), key=lambda item: (item.seed, item.pair_index))
        comparisons = _merge_comparisons([pair.comparisons for pair in ordered])
        total_errors = _merge_comparisons([pair.total_errors for pair in ordered])
        mx = _merge_mx([pair.mx for pair in ordered])
        transition_groups = Counter()
        transition_elements = Counter()
        valid_groups = invalid_groups = valid_elements = invalid_elements = 0
        events_cumulative: Counter = Counter()
        cmax_histogram: Counter = Counter()
        quantization_events: dict = {}
        identity_violations = 0
        for pair in ordered:
            transition_groups.update(pair.transition_groups)
            transition_elements.update(pair.transition_elements)
            valid_groups += pair.transition_valid_groups
            invalid_groups += pair.transition_invalid_groups
            valid_elements += pair.transition_valid_elements
            invalid_elements += pair.transition_invalid_elements
            events_cumulative.update(pair.events_cumulative)
            cmax_histogram.update(pair.cmax_histogram)
            identity_violations += pair.identity_violations
            for path, events in pair.quantization_events_by_path.items():
                quantization_events.setdefault(path, Counter()).update(events)
        return {
            "activation_distribution": self.activation_distribution,
            "weight_distribution": self.weight_distribution,
            "x": self.K,
            "K": self.K,
            "B": self.B,
            "M": self.M,
            "N": self.N,
            "w_align": self.w_align,
            "container_bits": self.container_bits,
            "comparisons": comparisons,
            "total_errors": total_errors,
            "mx_comparisons": mx,
            "transitions": {
                "definition": "representation = (scale_code, element_code)",
                "coverage": {
                    "valid_groups": valid_groups,
                    "invalid_groups": invalid_groups,
                    "total_groups": valid_groups + invalid_groups,
                    "valid_elements": valid_elements,
                    "invalid_elements": invalid_elements,
                    "total_elements": valid_elements + invalid_elements,
                },
                "groups": _transition_payload(transition_groups, valid_groups, invalid_groups),
                "elements": _transition_payload(transition_elements, valid_elements, invalid_elements),
            },
            "identity_satisfied": identity_violations == 0,
            "identity_violations": identity_violations,
            "quantization_events_by_path": {
                path: {key: int(value) for key, value in sorted(events.items())}
                for path, events in sorted(quantization_events.items())
            },
            "quantization_events_note": (
                "counts are per quantization path (Q_S/Q_R/Q_A independent and shared); they are "
                "not unique output events and must not be summed across paths"
            ),
            "alignment_diagnostics": {
                "definition": "cumulative counters/histograms at this K (covering all blocks up to K); never sum across K",
                "events_cumulative": {
                    key: int(events_cumulative.get(key, 0)) for key in ALIGN_EVENT_KEYS
                },
                "cmax_minus_ci_histogram_cumulative": {
                    key: int(cmax_histogram[key])
                    for key in sorted(cmax_histogram, key=lambda item: int(item))
                },
                "by_pair": [self._pair_compact(pair) for pair in ordered],
            },
            "counts": {
                "snapshots": len(ordered),
                "invalid_input_snapshots": self.n_invalid_input_snapshots,
            },
            "invalid_input_reasons": dict(self.invalid_input_reasons),
        }


def _merge_comparisons(per_pair: list[dict]) -> dict:
    if not per_pair:
        return {}
    names = list(per_pair[0].keys())
    merged = {}
    for name in names:
        combined = ComparisonAccumulator(name)
        for mapping in per_pair:
            combined.merge_from(mapping[name])
        merged[name] = combined.finalize()
    return merged


def _merge_mx(per_pair: list[dict]) -> dict:
    if not per_pair:
        return {}
    names = list(per_pair[0].keys())
    merged = {}
    for name in names:
        combined = MxComparisonAccumulator(name=name, ulp_condition=per_pair[0][name].ulp_condition)
        for mapping in per_pair:
            combined.merge_from(mapping[name])
        merged[name] = combined.finalize()
    return merged


def _pooled_rms(tensor: MxTensor) -> float:
    """Pooled RMS over all decoded MX elements: sqrt(sum(v^2)/elements).

    This is deliberately not an equal-weight mean of per-pair RMS values.
    """
    import math

    total = Dyadic.zero()
    count = 0
    for row in range(tensor.rows):
        for col in range(tensor.cols):
            value = tensor.decoded_value(row, col)
            total = total + value * value
            count += 1
    if count == 0:
        return 0.0
    return math.sqrt(float(total.to_fraction() / count))


def _group_scale_histogram(tensor: MxTensor) -> dict:
    """Each E8M0 scale code counted once per group (not per element)."""
    histogram: Counter = Counter()
    grid_rows, grid_cols = tensor.group_grid
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            if tensor.group_axis == 0:
                code = tensor.scale_codes[grid_row * grid_cols + grid_col]
                histogram[code - 127] += 1
            else:
                code = tensor.scale_codes[grid_row * grid_cols + grid_col]
                histogram[code - 127] += 1
    return {str(key): histogram[key] for key in sorted(histogram)}


def _transition_key(r_match: bool, a_match: bool, candidate_same: bool) -> str:
    if r_match and a_match:
        return "both_match"
    if r_match and not a_match:
        return "baseline_match_candidate_mismatch"
    if not r_match and a_match:
        return "baseline_mismatch_candidate_match"
    return "both_mismatch_candidate_same" if candidate_same else "both_mismatch_candidate_differs"


def _transition_payload(counter: Counter, valid_denominator: int, invalid_count: int = 0) -> dict:
    keys = (
        "both_match",
        "baseline_match_candidate_mismatch",
        "baseline_mismatch_candidate_match",
        "both_mismatch_candidate_differs",
        "both_mismatch_candidate_same",
    )
    payload = {key: counter[key] for key in keys}
    payload.update(
        {
            "denominator": valid_denominator,
            "valid_denominator": valid_denominator,
            "invalid": invalid_count,
            "total_denominator": valid_denominator + invalid_count,
        }
    )
    payload["rates"] = {
        key: (payload[key] / valid_denominator) if valid_denominator else None for key in keys
    }
    return payload


def _cell_counts(M: int, N: int) -> dict:
    return {"elements": M * N, "groups": M * N // GROUP_LENGTH}


class AlignedAnalysisBuilder:
    def __init__(self) -> None:
        self.cells: dict[tuple[str, str, int], AlignedCell] = {}
        self.input_meta: dict[str, dict] = {}

    def add_input(self, record: dict) -> tuple[bool, str | None, dict]:
        """Validate one input record.

        Structural damage (missing tensors/fingerprints, malformed code arrays,
        count mismatch) raises immediately.  Numeric invalidity that this
        experiment explicitly supports (E4M3 NaN element code, reserved E8M0
        code 0xFF) marks the pair invalid so every comparison records invalid
        coverage instead of silently dropping it.
        """
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("input record without a sample_id")
        meta = {
            "activation_distribution": record.get("activation_distribution"),
            "weight_distribution": record.get("weight_distribution"),
            "source": record.get("source"),
            "x_input_quantization_events": record.get("x_input_quantization_events") or {},
            "w_input_quantization_events": record.get("w_input_quantization_events") or {},
            "x_input_diagnostics": record.get("x_input_diagnostics") or {},
            "w_input_diagnostics": record.get("w_input_diagnostics") or {},
        }
        reasons: list[str] = []
        for name in ("x", "w"):
            payload = record.get(name)
            if not isinstance(payload, dict):
                raise ValueError(f"{sample_id}: missing {name} tensor record")
            if not payload.get("element_fingerprint"):
                raise ValueError(f"{sample_id}: missing {name} element fingerprint")
            codes_available = payload.get("element_codes") is not None
            if not codes_available:
                continue
            element_codes = payload.get("element_codes")
            scale_codes = payload.get("scale_codes")
            if not isinstance(element_codes, list) or not isinstance(scale_codes, list):
                raise ValueError(f"{sample_id}: {name} code arrays are missing or malformed")
            for index, raw in enumerate(element_codes):
                value = _require_code(raw, sample_id, f"{name}.element_codes", index)
                if is_nan_code(value):
                    reasons.append(f"{name}:e4m3_nan_element_code_at_{index}")
            for index, raw in enumerate(scale_codes):
                value = _require_code(raw, sample_id, f"{name}.scale_codes", index)
                if value == E8M0_NAN_CODE:
                    reasons.append(f"{name}:e8m0_reserved_scale_code_at_{index}")
            if reasons:
                continue
            try:
                tensor = MxTensor.from_dict(payload)
            except ValueError as exc:
                raise ValueError(f"{sample_id}: invalid {name} tensor structure: {exc}") from exc
            meta[f"{name}_pooled_mx_rms"] = _pooled_rms(tensor)
            meta[f"{name}_group_scale_exponent_histogram"] = _group_scale_histogram(tensor)
            meta[f"{name}_element_count"] = tensor.rows * tensor.cols
        self.input_meta[sample_id] = meta
        if reasons:
            return False, ";".join(reasons), meta
        return True, None, meta

    def add(self, record: dict, pair_valid: bool, pair_reason: str | None, input_meta: dict) -> None:
        act = record["activation_distribution"]
        weight = record["weight_distribution"]
        K = int(record["K"])
        key = (act, weight, K)
        cell = self.cells.get(key)
        if cell is None:
            cell = AlignedCell(
                activation_distribution=act,
                weight_distribution=weight,
                K=K,
                B=int(record["B"]),
            )
            self.cells[key] = cell
        cell.add(record, pair_valid, pair_reason, input_meta)

    def finalize(self, base: dict) -> dict:
        summary = dict(base)
        summary["x_name"] = "K"
        summary["x_values"] = sorted({key[2] for key in self.cells})
        summary["cells"] = [self.cells[key].finalize() for key in sorted(self.cells)]
        return summary


def _require_code(raw, sample_id: str, field: str, index: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{sample_id}: {field}[{index}] is not an integer code: {raw!r}")
    if not 0 <= raw <= 0xFF:
        raise ValueError(f"{sample_id}: {field}[{index}] out of byte range: {raw!r}")
    return raw


def _expected_identity(manifest: dict) -> dict:
    """Expected identity sets derived from the run manifest/config, not from counts."""
    if manifest.get("status") != "completed":
        raise ValueError(f"alignment run status is {manifest.get('status')!r}, not 'completed'")
    if manifest.get("run_kind") != "mx_alignment_sweep":
        raise ValueError(f"unexpected run_kind {manifest.get('run_kind')!r}")
    config = manifest.get("config") or {}
    pairs = manifest.get("distribution_pairs") or config.get("distribution_pairs") or []
    if not pairs:
        raise ValueError("alignment manifest has no distribution_pairs")
    seeds = [int(seed) for seed in (config.get("seeds") or [])]
    if not seeds:
        raise ValueError("alignment manifest has no seeds")
    pairs_per_seed = int(config.get("pairs_per_seed", 1))
    K_list = sorted({int(k) for k in (manifest.get("K_list") or [])})
    if not K_list:
        raise ValueError("alignment manifest has no K_list")
    fixed = manifest.get("fixed") or {}
    M = int(fixed.get("M"))
    N = int(fixed.get("N"))
    group_length = int(manifest.get("group_length", 32))
    mac_parallelism = int(manifest.get("mac_parallelism", 16))
    w_align = int(manifest.get("w_align"))
    container_bits = int(manifest.get("container_bits", w_align + 7))
    expected_samples: dict[str, dict] = {}
    for pair in pairs:
        act = pair["activation_distribution"]
        weight = pair["weight_distribution"]
        for seed in seeds:
            for pair_index in range(pairs_per_seed):
                sample_id = f"mx|{act}|{weight}|seed{seed}|{pair_index:05d}"
                expected_samples[sample_id] = {
                    "activation_distribution": act,
                    "weight_distribution": weight,
                    "seed": seed,
                    "pair_index": pair_index,
                }
    expected_snapshots = {(sid, k) for sid in expected_samples for k in K_list}
    counts = manifest.get("counts") or {}
    if int(counts.get("matrix_pairs", -1)) != len(expected_samples):
        raise ValueError(
            "manifest counts.matrix_pairs does not match the expected identity set: "
            f"{counts.get('matrix_pairs')} != {len(expected_samples)}"
        )
    if int(counts.get("snapshot_records", -1)) != len(expected_snapshots):
        raise ValueError(
            "manifest counts.snapshot_records does not match the expected identity set: "
            f"{counts.get('snapshot_records')} != {len(expected_snapshots)}"
        )
    expected = manifest.get("expected") or {}
    if expected.get("matrix_pairs") not in (None, len(expected_samples)):
        raise ValueError("manifest expected.matrix_pairs disagrees with the identity set")
    if expected.get("snapshot_records") not in (None, len(expected_snapshots)):
        raise ValueError("manifest expected.snapshot_records disagrees with the identity set")
    return {
        "M": M,
        "N": N,
        "group_length": group_length,
        "mac_parallelism": mac_parallelism,
        "w_align": w_align,
        "container_bits": container_bits,
        "K_list": K_list,
        "expected_samples": expected_samples,
        "expected_snapshots": expected_snapshots,
    }


def analyze_mx_alignment_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    write_plots: bool = True,
) -> Path:
    source = run_path(run_dir)
    manifest = load_manifest(source)
    require_schema(manifest, (MX_ALIGN_SCHEMA_VERSION,))
    if manifest.get("align_policy") != MX_ALIGN_POLICY:
        raise ValueError(f"unsupported align_policy {manifest.get('align_policy')!r}")
    expected = _expected_identity(manifest)

    # Read and validate everything before creating any output directory, so a
    # failed integrity check cannot leave a half-finished "successful" run.
    builder = AlignedAnalysisBuilder()
    pair_status: dict[str, tuple[bool, str | None, dict]] = {}
    n_input_records = 0
    for record in iter_mx_inputs(source):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("alignment input record without a sample_id")
        if sample_id in pair_status:
            raise ValueError(f"duplicate input record for sample_id {sample_id}")
        if sample_id not in expected["expected_samples"]:
            raise ValueError(f"unknown sample_id in alignment inputs: {sample_id}")
        identity = expected["expected_samples"][sample_id]
        for field, value in (
            ("activation_distribution", identity["activation_distribution"]),
            ("weight_distribution", identity["weight_distribution"]),
        ):
            if record.get(field) != value:
                raise ValueError(
                    f"{sample_id}: input {field}={record.get(field)!r} != expected {value!r}"
                )
        if int(record.get("seed", -1)) != identity["seed"] or int(
            record.get("pair_index", -1)
        ) != identity["pair_index"]:
            raise ValueError(f"{sample_id}: input seed/pair_index mismatch")
        if int(record.get("M", -1)) != expected["M"] or int(record.get("N", -1)) != expected["N"]:
            raise ValueError(f"{sample_id}: input M/N mismatch")
        if sorted({int(k) for k in record.get("K_list") or []}) != expected["K_list"]:
            raise ValueError(f"{sample_id}: input K_list does not match the manifest K_list")
        if int(record.get("w_align", -1)) != expected["w_align"]:
            raise ValueError(f"{sample_id}: input w_align mismatch")
        if int(record.get("group_length", -1)) != expected["group_length"]:
            raise ValueError(f"{sample_id}: input group_length mismatch")
        if int(record.get("mac_parallelism", -1)) != expected["mac_parallelism"]:
            raise ValueError(f"{sample_id}: input mac_parallelism mismatch")
        verdict = builder.add_input(record)
        pair_status[sample_id] = verdict
        n_input_records += 1
    missing_inputs = sorted(set(expected["expected_samples"]) - set(pair_status))
    if missing_inputs:
        raise ValueError(
            f"missing input records for {len(missing_inputs)} expected matrix pairs, first: "
            f"{missing_inputs[:3]}"
        )

    n_records = 0
    n_snapshots = 0
    seen_snapshots: set = set()
    for record in iter_mx_samples(source):
        n_records += 1
        if record.get("record_type") != "align_snapshot":
            raise ValueError(f"unexpected record type in alignment run: {record.get('record_type')!r}")
        sample_id = record.get("sample_id")
        if sample_id not in expected["expected_samples"]:
            raise ValueError(f"unknown sample_id in alignment snapshots: {sample_id!r}")
        K = int(record.get("K", -1))
        if K not in expected["K_list"]:
            raise ValueError(f"{sample_id}: unexpected snapshot K={K}")
        key = (sample_id, K)
        if key in seen_snapshots:
            raise ValueError(f"duplicate alignment snapshot for sample_id/K {key}")
        seen_snapshots.add(key)
        identity = expected["expected_samples"][sample_id]
        for field, value in (
            ("activation_distribution", identity["activation_distribution"]),
            ("weight_distribution", identity["weight_distribution"]),
        ):
            if record.get(field) != value:
                raise ValueError(f"{sample_id} K={K}: snapshot {field} mismatch")
        if int(record.get("seed", -1)) != identity["seed"] or int(
            record.get("pair_index", -1)
        ) != identity["pair_index"]:
            raise ValueError(f"{sample_id} K={K}: snapshot seed/pair_index mismatch")
        if int(record.get("M", -1)) != expected["M"] or int(record.get("N", -1)) != expected["N"]:
            raise ValueError(f"{sample_id} K={K}: snapshot M/N mismatch")
        if int(record.get("B", -1)) != K // expected["mac_parallelism"]:
            raise ValueError(f"{sample_id} K={K}: snapshot B mismatch")
        if int(record.get("w_align", -1)) != expected["w_align"]:
            raise ValueError(f"{sample_id} K={K}: snapshot w_align mismatch")
        if int(record.get("container_bits", -1)) != expected["container_bits"]:
            raise ValueError(f"{sample_id} K={K}: snapshot container_bits mismatch")
        if int(record.get("group_length", -1)) != expected["group_length"]:
            raise ValueError(f"{sample_id} K={K}: snapshot group_length mismatch")
        if int(record.get("mac_parallelism", -1)) != expected["mac_parallelism"]:
            raise ValueError(f"{sample_id} K={K}: snapshot mac_parallelism mismatch")
        for field in ("S", "L1", "T", "r_bits", "a_bits"):
            values = record.get(field)
            if not isinstance(values, list) or len(values) != expected["M"] * expected["N"]:
                raise ValueError(
                    f"{sample_id} K={K}: snapshot {field} length "
                    f"{None if not isinstance(values, list) else len(values)} != "
                    f"{expected['M'] * expected['N']}"
                )
        for field in ("align_abs_sum", "align_abs_max"):
            values = record.get(field)
            if values is not None and (
                not isinstance(values, list) or len(values) != expected["M"] * expected["N"]
            ):
                raise ValueError(f"{sample_id} K={K}: snapshot {field} length mismatch")
        n_snapshots += 1
        valid, reason, meta = pair_status[sample_id]
        builder.add(record, valid, reason, meta)
    missing_snapshots = sorted(expected["expected_snapshots"] - seen_snapshots)
    if missing_snapshots:
        raise ValueError(
            f"missing alignment snapshots for {len(missing_snapshots)} expected (sample_id, K) "
            f"keys, first: {missing_snapshots[:3]}"
        )

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    analyzer_digest = analyzer_source_digest()
    base = {
        "schema_version": MX_ALIGN_SCHEMA_VERSION,
        "summary_version": MX_ALIGN_SUMMARY_VERSION,
        "analysis_version": MX_ALIGN_ANALYSIS_VERSION,
        "run_id": f"{manifest['run_id']}+align_analysis",
        "source_run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "w_align": manifest.get("w_align"),
        "container_bits": manifest.get("container_bits"),
        "align_policy": manifest.get("align_policy"),
        "fixed": manifest["fixed"],
        "K_list": manifest.get("K_list"),
        "config": manifest.get("config"),
        "config_hash": manifest.get("config_hash"),
        "created_utc": created,
        "source_created_utc": manifest.get("created_utc"),
        "source_digest": manifest.get("source_digest"),
        "analyzer_source_digest": analyzer_digest,
        "input_sampling_version": manifest.get("input_sampling_version"),
        "input_stream_version": manifest.get("input_stream_version"),
        "input_quantization_policy": manifest.get("input_quantization_policy"),
        "source": manifest.get("source"),
        "n_input_records_read": n_input_records,
        "n_records_read": n_records,
        "n_snapshots_analyzed": n_snapshots,
        "input_diagnostics": _aggregate_input_diagnostics(pair_status),
        "policy": POLICY,
        "notes": NOTES,
    }
    summary = builder.finalize(base)
    for cell in summary["cells"]:
        if not cell["identity_satisfied"]:
            raise ValueError(
                "A-S = (T-S) + (A-T) identity violated in "
                f"{cell['activation_distribution']}/{cell['weight_distribution']}/K={cell['K']}; "
                "refusing to report success"
            )

    if out_dir is None:
        out = Path("results/numerical/mx_alignment/analysis") / f"{manifest['run_id']}_analysis_{_timestamp()}"
    else:
        out = Path(out_dir)
    source_resolved = source.resolve()
    out_resolved = out.resolve()
    if out_resolved == source_resolved or source_resolved in out_resolved.parents:
        raise ValueError(f"output directory must not be inside the source run: {out}")
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True)

    write_json(out / "summary.json", summary)
    write_alignment_csv(out / "summary.csv", summary)

    source_files = {}
    for name in SOURCE_FILES:
        path = source / name
        if path.exists():
            source_files[name] = _file_identity(path)
    analysis_manifest = {
        "analysis_version": PACKAGE_VERSION,
        "analysis_kind": "mx_alignment_k_sweep",
        "alignment_analysis_version": MX_ALIGN_ANALYSIS_VERSION,
        "summary_version": MX_ALIGN_SUMMARY_VERSION,
        "schema_version": MX_ALIGN_SCHEMA_VERSION,
        "align_policy": manifest.get("align_policy"),
        "w_align": manifest.get("w_align"),
        "container_bits": manifest.get("container_bits"),
        "created_utc": created,
        "analyzer_source_digest": analyzer_digest,
        "numeric_run_source_digest": manifest.get("source_digest"),
        "source": {
            "run_dir": str(source_resolved),
            "run_id": manifest["run_id"],
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
            "config_hash": manifest.get("config_hash"),
            "input_sampling_version": manifest.get("input_sampling_version"),
            "input_stream_version": manifest.get("input_stream_version"),
            "input_quantization_policy": manifest.get("input_quantization_policy"),
            "mx_source": manifest.get("source"),
            "files": source_files,
        },
        "input_records_read": n_input_records,
        "records_read": n_records,
        "snapshots_analyzed": n_snapshots,
        "policy": POLICY,
        "notes": NOTES,
    }
    write_json(out / "analysis_manifest.json", analysis_manifest)

    if write_plots:
        from ..plotting.mx_alignment_plots import plot_mx_alignment_summary

        plot_mx_alignment_summary(summary, out / "plots")
    return out


def _aggregate_input_diagnostics(pair_status: dict) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for _, _, meta in pair_status.values():
        key = (meta.get("activation_distribution"), meta.get("weight_distribution"))
        entry = grouped.setdefault(
            key,
            {
                "activation_distribution": key[0],
                "weight_distribution": key[1],
                "matrix_pairs": 0,
                "x_input_quantization_events": Counter(),
                "w_input_quantization_events": Counter(),
                "x_pooled_mx_rms": [],
                "w_pooled_mx_rms": [],
                "scale_exponent_histogram_group_weighted": Counter(),
                "scale_exponent_histogram_element_weighted": Counter(),
                "pooled_rms_available": True,
            },
        )
        entry["matrix_pairs"] += 1
        for path in ("x", "w"):
            entry[f"{path}_input_quantization_events"].update(
                meta.get(f"{path}_input_quantization_events") or {}
            )
            pooled = meta.get(f"{path}_pooled_mx_rms")
            if pooled is None:
                entry["pooled_rms_available"] = False
            else:
                entry[f"{path}_pooled_mx_rms"].append(pooled)
            group_hist = meta.get(f"{path}_group_scale_exponent_histogram")
            if group_hist:
                for exponent, count in group_hist.items():
                    entry["scale_exponent_histogram_group_weighted"][exponent] += int(count)
            element_hist = (meta.get(f"{path}_input_diagnostics") or {}).get(
                "scale_exponent_histogram"
            ) or {}
            for exponent, count in element_hist.items():
                entry["scale_exponent_histogram_element_weighted"][exponent] += int(count)

    def mean(values):
        return sum(values) / len(values) if values else None

    result = []
    for key in sorted(grouped):
        entry = grouped[key]
        result.append(
            {
                "activation_distribution": entry["activation_distribution"],
                "weight_distribution": entry["weight_distribution"],
                "matrix_pairs": entry["matrix_pairs"],
                "definition": (
                    "input diagnostics only, not error metrics; RMS is pooled over elements when "
                    "MX codes are stored (otherwise null); scale histograms are explicitly labelled"
                ),
                "input_quantization_events": {
                    "x": {k: int(v) for k, v in sorted(entry["x_input_quantization_events"].items())},
                    "w": {k: int(v) for k, v in sorted(entry["w_input_quantization_events"].items())},
                },
                "x_pooled_mx_rms_per_pair_mean": mean(entry["x_pooled_mx_rms"]),
                "w_pooled_mx_rms_per_pair_mean": mean(entry["w_pooled_mx_rms"]),
                "scale_exponent_histogram_group_weighted": {
                    str(key): int(entry["scale_exponent_histogram_group_weighted"][key])
                    for key in sorted(entry["scale_exponent_histogram_group_weighted"], key=lambda x: int(x))
                },
                "scale_exponent_histogram_element_weighted": {
                    str(key): int(entry["scale_exponent_histogram_element_weighted"][key])
                    for key in sorted(
                        entry["scale_exponent_histogram_element_weighted"], key=lambda x: int(x)
                    )
                },
            }
        )
    return result


def write_alignment_csv(output_path: str | Path, summary: dict) -> None:
    import csv

    fieldnames = [
        "activation_distribution",
        "weight_distribution",
        "x_name",
        "K",
        "B",
        "section",
        "metric",
        "value",
        "status",
        "denominator",
        "note",
    ]
    rows: list[dict] = []

    def emit(base, section, metric, value, status, denominator="", note=""):
        if value is None:
            text = ""
        elif isinstance(value, float):
            text = repr(value)
        else:
            text = str(value)
        rows.append(
            {
                **base,
                "section": section,
                "metric": metric,
                "value": text,
                "status": status,
                "denominator": denominator,
                "note": note,
            }
        )

    for cell in summary["cells"]:
        base = {
            "activation_distribution": cell["activation_distribution"],
            "weight_distribution": cell["weight_distribution"],
            "x_name": summary["x_name"],
            "K": cell["K"],
            "B": cell["B"],
        }
        emit(base, "cell", "w_align", cell["w_align"], "ok")
        def emit_payload(name, payload):
            emit(base, name, "status", payload.get("status"), "ok", _coverage_text(payload.get("coverage")))
            for metric, entry in (payload.get("metrics") or {}).items():
                emit(base, name, metric, entry["value"], entry["status"], "scope:whole_batch")
            subset = payload.get("finite_subset")
            if isinstance(subset, dict):
                emit(
                    base,
                    name,
                    "finite_subset_status",
                    subset.get("status", payload.get("status")),
                    "ok",
                    _coverage_text(subset.get("coverage")),
                    subset.get("note", ""),
                )
                for metric, entry in (subset.get("metrics") or {}).items():
                    emit(
                        base,
                        name,
                        metric,
                        entry["value"],
                        entry["status"],
                        "scope:finite_subset",
                        subset.get("note", ""),
                    )

        for name, payload in cell["comparisons"].items():
            emit_payload(name, payload)
        for name, payload in cell["total_errors"].items():
            emit_payload(name, payload)
        for name, payload in cell["mx_comparisons"].items():
            emit_payload(name, payload)
            distributions = payload.get("distributions") or {}
            ulp = distributions.get("ulp")
            if ulp is not None:
                for key in ("p50", "p90", "p99", "max"):
                    emit(base, name, f"ulp_{key}", ulp[key], ulp["status"], f"n_mismatch={ulp['n_mismatch']}")
        transitions = cell["transitions"]["groups"]
        for key in (
            "both_match",
            "baseline_match_candidate_mismatch",
            "baseline_mismatch_candidate_match",
            "both_mismatch_candidate_differs",
            "both_mismatch_candidate_same",
        ):
            emit(base, "transitions_groups", key, transitions[key], "ok", f"groups={transitions['denominator']}")
        diagnostics = cell["alignment_diagnostics"]
        for key, value in diagnostics["events_cumulative"].items():
            emit(base, "alignment_events_cumulative", key, value, "ok", note="cumulative up to K")
        for key, value in diagnostics["cmax_minus_ci_histogram_cumulative"].items():
            emit(base, "cmax_minus_ci", key, value, "ok", note="cumulative up to K")
        for quant_path, events in cell["quantization_events_by_path"].items():
            for key, value in events.items():
                emit(base, f"quantization:{quant_path}", key, value, "ok", note="per-path call counts")
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coverage_text(coverage) -> str:
    if not isinstance(coverage, dict):
        return ""
    return " ".join(f"{key}={value}" for key, value in sorted(coverage.items()))
