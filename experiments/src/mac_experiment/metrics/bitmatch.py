"""Bit-level match and ULP distance statistics for output rounding.

Two primary comparisons per record (both on the window-loss-free path only,
never on the candidate ``a_B``):

* FP32: ``o32 = RN32(S_B)`` vs ``r_B`` (stored FP32 accumulation state);
* BF16: ``o16 = RN_BF16(S_B)`` (directly from the exact dyadic, no FP32
  intermediate) vs ``r16 = RN_BF16(r_B)``.

Two auxiliary double-rounding diagnostics are also accumulated (they never
replace the primary comparisons and their rates/distances must not be added
together):

* ``aux_double_rounding``: ``d16 = RN_BF16(o32)`` vs ``o16``;
* ``aux_accum_bf16``: ``d16`` vs ``r16``.

Bit match compares the complete encoding and therefore distinguishes +0 from
-0.  ULP distance is computed only when both values are finite, on the ordered
sequence of representable values of the target format with +0/-0 merged.
Conditional distance statistics only cover finite pairs whose bit patterns
differ; if there are none, quantiles are ``null`` with status
``no_mismatch_samples`` (never 0).  Distance buckets use all finite pairs as
denominator.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..arithmetic.bf16 import Bf16, round_dyadic_to_bf16
from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..arithmetic.rounding import round_dyadic_to_fp32
from ..arithmetic.ulp import bf16_ordered_index, fp32_ordered_index

EVENT_TYPES = (
    "reference_finite_result_nonfinite",
    "reference_nonfinite_result_finite",
    "both_inf_same_sign",
    "both_inf_opposite_sign",
    "any_nan",
)

PAIR_SPECS = {
    "fp32": ("o32 = RN32(S_B)", "r_B (FP32 accumulation)"),
    "bf16": ("o16 = RN_BF16(S_B) direct", "r16 = RN_BF16(r_B)"),
    "aux_double_rounding": ("o16 = RN_BF16(S_B) direct", "d16 = RN_BF16(o32)"),
    "aux_accum_bf16": ("r16 = RN_BF16(r_B)", "d16 = RN_BF16(o32)"),
}

PRIMARY_PAIRS = ("fp32", "bf16")
AUX_PAIRS = ("aux_double_rounding", "aux_accum_bf16")

BUCKET_LABELS = ("0", "1", "2", "3-4", ">=5")

QUANTILE_METHOD = "nearest_rank"


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


def nearest_rank(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    rank = max(1, min(len(sorted_values), math.ceil(q * len(sorted_values))))
    return sorted_values[rank - 1]


@dataclass
class PairStats:
    name: str
    reference_label: str
    result_label: str
    index_fn: object
    n_total: int = 0
    bit_match: int = 0
    bit_mismatch: int = 0
    signed_zero_mismatch: int = 0
    n_finite_pairs: int = 0
    n_finite_mismatch: int = 0
    conditional_distances: list[int] = field(default_factory=list)
    buckets: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    events: Counter = field(default_factory=Counter)
    by_seed: dict = field(default_factory=dict)

    def add(self, reference, result, seed: int) -> None:
        self.n_total += 1
        seed_stats = self.by_seed.setdefault(
            seed, {"n_total": 0, "bit_mismatch": 0, "finite_mismatch": 0}
        )
        seed_stats["n_total"] += 1

        reference_finite = reference.is_finite()
        result_finite = result.is_finite()
        if reference.is_nan() or result.is_nan():
            self.events["any_nan"] += 1
        elif not reference_finite and not result_finite:
            if reference.sign == result.sign:
                self.events["both_inf_same_sign"] += 1
            else:
                self.events["both_inf_opposite_sign"] += 1
        elif reference_finite and not result_finite:
            self.events["reference_finite_result_nonfinite"] += 1
        elif not reference_finite and result_finite:
            self.events["reference_nonfinite_result_finite"] += 1

        if reference.bits == result.bits:
            self.bit_match += 1
        else:
            self.bit_mismatch += 1
            seed_stats["bit_mismatch"] += 1

        if reference_finite and result_finite:
            self.n_finite_pairs += 1
            distance = abs(self.index_fn(reference.bits) - self.index_fn(result.bits))
            self.buckets[bucket_index(distance)] += 1
            if reference.bits != result.bits:
                self.n_finite_mismatch += 1
                seed_stats["finite_mismatch"] += 1
                self.conditional_distances.append(distance)
                if distance == 0:
                    self.signed_zero_mismatch += 1

    def finalize(self) -> dict:
        distances = sorted(self.conditional_distances)
        if distances:
            conditional = {
                "status": "ok",
                "note": "only finite pairs with bit mismatch",
                "n": len(distances),
                "p50": nearest_rank(distances, 0.50),
                "p90": nearest_rank(distances, 0.90),
                "p99": nearest_rank(distances, 0.99),
                "max": distances[-1],
                "quantile_method": QUANTILE_METHOD,
            }
        else:
            conditional = {
                "status": "no_mismatch_samples",
                "note": "only finite pairs with bit mismatch",
                "n": 0,
                "p50": None,
                "p90": None,
                "p99": None,
                "max": None,
                "quantile_method": QUANTILE_METHOD,
            }
        buckets = {}
        for label, count in zip(BUCKET_LABELS, self.buckets):
            buckets[label] = {
                "count": count,
                "rate": (count / self.n_finite_pairs) if self.n_finite_pairs else None,
            }
        return {
            "comparison": self.name,
            "reference": self.reference_label,
            "result": self.result_label,
            "counts": {
                "n_total": self.n_total,
                "bit_match": self.bit_match,
                "bit_mismatch": self.bit_mismatch,
                "signed_zero_mismatch": self.signed_zero_mismatch,
                "n_finite_pairs": self.n_finite_pairs,
                "n_finite_mismatch": self.n_finite_mismatch,
            },
            "bit_match_rate": (self.bit_match / self.n_total) if self.n_total else None,
            "bit_mismatch_rate": (self.bit_mismatch / self.n_total) if self.n_total else None,
            "conditional_ulp": conditional,
            "distance_buckets": {
                "denominator": "n_finite_pairs",
                "n_finite_pairs": self.n_finite_pairs,
                "buckets": buckets,
            },
            "events": {key: self.events[key] for key in EVENT_TYPES},
            "by_seed": {str(seed): self.by_seed[seed] for seed in sorted(self.by_seed)},
        }


@dataclass
class BitMatchCell:
    format: str
    distribution: str
    x: int
    pairs: dict = field(default_factory=dict)
    n_records: int = 0

    def __post_init__(self) -> None:
        if not self.pairs:
            self.pairs = {
                name: PairStats(
                    name=name,
                    reference_label=PAIR_SPECS[name][0],
                    result_label=PAIR_SPECS[name][1],
                    index_fn=fp32_ordered_index if name == "fp32" else bf16_ordered_index,
                )
                for name in PAIR_SPECS
            }

    def add_record(self, record: dict) -> None:
        s_total = Dyadic.from_dict(record["S"])
        r_state = Fp32.from_bits(record["r_bits"])
        seed = int(record["seed"])
        self.n_records += 1

        o32, o32_exact = round_dyadic_to_fp32(s_total)
        o16, _ = round_dyadic_to_bf16(s_total)
        if r_state.is_nan():
            r16 = Bf16.nan()
        elif r_state.is_inf():
            r16 = Bf16.inf(r_state.sign)
        elif r_state.is_zero():
            r16 = Bf16.zero(r_state.sign)
        else:
            r16, _ = round_dyadic_to_bf16(r_state.exact_dyadic())
        if o32.is_nan():
            d16 = Bf16.nan()
        elif o32.is_inf():
            d16 = Bf16.inf(o32.sign)
        elif o32.is_zero():
            d16 = Bf16.zero(o32.sign)
        else:
            d16, _ = round_dyadic_to_bf16(o32_exact)

        self.pairs["fp32"].add(o32, r_state, seed)
        self.pairs["bf16"].add(o16, r16, seed)
        self.pairs["aux_double_rounding"].add(o16, d16, seed)
        self.pairs["aux_accum_bf16"].add(r16, d16, seed)

    def finalize(self) -> dict:
        return {
            "format": self.format,
            "distribution": self.distribution,
            "x": self.x,
            "n_records": self.n_records,
            "pairs": {name: stats.finalize() for name, stats in self.pairs.items()},
        }


class BitMatchBuilder:
    def __init__(self, mode: str, x_name: str):
        if x_name not in ("W", "B"):
            raise ValueError("x_name must be 'W' or 'B'")
        self.mode = mode
        self.x_name = x_name
        self.cells: dict[tuple[str, str, int], BitMatchCell] = {}

    def add(self, record: dict) -> None:
        if record.get("mode") != self.mode:
            raise ValueError(f"record mode {record.get('mode')!r} does not match builder mode {self.mode!r}")
        if record.get("record_type") != "snapshot":
            return
        x = int(record[self.x_name])
        key = (record["format"], record["distribution"], x)
        cell = self.cells.get(key)
        if cell is None:
            cell = BitMatchCell(format=record["format"], distribution=record["distribution"], x=x)
            self.cells[key] = cell
        cell.add_record(record)

    def finalize(self, base: dict) -> dict:
        summary = dict(base)
        summary["x_name"] = self.x_name
        summary["x_values"] = sorted({key[2] for key in self.cells})
        summary["cells"] = [self.cells[key].finalize() for key in sorted(self.cells)]
        return summary
