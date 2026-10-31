"""Cell aggregation: per (format, distribution, W or B) metric accumulators.

The aggregator consumes the same serialised snapshot records that are written
to disk, so an offline re-analysis reproduces the original summary exactly.

Validity is decided per comparison:

* ``total``           needs a finite candidate ``a_B`` (``S_B`` is always exact).
* ``window_effect``   needs finite ``a_B`` and ``r_B``.
* ``fp32_baseline``   needs a finite reference ``r_B``.
* ``alignment_only``  needs a complete exact ``E_align``; it is independent of
  the FP32 candidate/reference states, because ``Q_B = S_B + E_align`` is an
  exact dyadic accumulation and is never rounded to FP32.

A non-finite candidate therefore does not discard a still-defined
``fp32_baseline`` or ``alignment_only``; a non-finite reference does not
discard a still-defined ``total``.  The cell keeps an explicit failure status
either way.

Stage diagnostics are independent: ``E_align`` stays defined as long as the
window reduction itself ran, while ``E_acc`` is undefined once the candidate
accumulator has produced a non-finite state (an update whose rounding delta
does not exist).  Record schema v1 lacks the explicit ``E_acc_complete`` flag;
for those records completeness is derived from the recorded candidate state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..events import EVENT_KEYS
from .compare import STATUS_OK, ComparisonAccumulator, StageAccumulator

COMPARISONS = ("total", "window_effect", "fp32_baseline", "alignment_only")


@dataclass
class Cell:
    format: str
    distribution: str
    x: int
    comparisons: dict = field(default_factory=dict)
    stage: dict = field(default_factory=dict)
    events: Counter = field(default_factory=Counter)
    n_snapshots: int = 0
    n_aborted: int = 0
    n_nonfinite: int = 0
    n_other_status: int = 0

    def __post_init__(self) -> None:
        if not self.comparisons:
            self.comparisons = {name: ComparisonAccumulator(name) for name in COMPARISONS}
        if not self.stage:
            self.stage = {"E_align": StageAccumulator("E_align"), "E_acc": StageAccumulator("E_acc")}

    def add_record(self, record: dict) -> None:
        if record["record_type"] == "abort":
            self.n_aborted += 1
            self.events["window_overflow"] += 1
            return
        a = Fp32.from_bits(record["a_bits"])
        r = Fp32.from_bits(record["r_bits"])
        a_finite = a.is_finite()
        r_finite = r.is_finite()
        s_total = Dyadic.from_dict(record["S"])
        l1_total = Dyadic.from_dict(record["L1"])
        status = record.get("status", "ok")
        self.n_snapshots += 1
        if status == "nonfinite_fp32":
            self.n_nonfinite += 1
        elif status != "ok":
            self.n_other_status += 1

        if a_finite:
            self.comparisons["total"].add(a.exact_dyadic() - s_total, s_total, l1_total)
        else:
            self.comparisons["total"].add_invalid()
        if a_finite and r_finite:
            a_exact = a.exact_dyadic()
            r_exact = r.exact_dyadic()
            self.comparisons["window_effect"].add(a_exact - r_exact, r_exact, l1_total)
        else:
            self.comparisons["window_effect"].add_invalid()
        if r_finite:
            self.comparisons["fp32_baseline"].add(r.exact_dyadic() - s_total, s_total, l1_total)
        else:
            self.comparisons["fp32_baseline"].add_invalid()

        e_align_value = _validated_e_align(record)
        if e_align_value is not None:
            self.comparisons["alignment_only"].add(e_align_value, s_total, l1_total)
            self.stage["E_align"].add(e_align_value)
        else:
            self.comparisons["alignment_only"].add_invalid()
            self.stage["E_align"].add_invalid()
        e_acc_complete = bool(record.get("E_acc_complete", a_finite))
        e_acc_payload = record.get("E_acc")
        if e_acc_complete and e_acc_payload is not None:
            self.stage["E_acc"].add(Dyadic.from_dict(e_acc_payload))
        else:
            self.stage["E_acc"].add_invalid()

        for key in EVENT_KEYS:
            self.events[key] += record.get("events_endpoint", {}).get(key, 0)

    def finalize(self) -> dict:
        comparisons = {name: accumulator.finalize() for name, accumulator in self.comparisons.items()}
        statuses = {name: payload["status"] for name, payload in comparisons.items()}
        if all(status == STATUS_OK for status in statuses.values()):
            cell_status = STATUS_OK
        elif any(status == STATUS_OK for status in statuses.values()):
            cell_status = "partial_comparisons_invalid"
        else:
            cell_status = "all_comparisons_invalid"
        return {
            "format": self.format,
            "distribution": self.distribution,
            "x": self.x,
            "status": cell_status,
            "comparison_statuses": statuses,
            "comparisons": comparisons,
            "stage": {name: accumulator.finalize() for name, accumulator in self.stage.items()},
            "events": {key: self.events[key] for key in EVENT_KEYS},
            "counts": {
                "snapshots": self.n_snapshots,
                "aborted": self.n_aborted,
                "nonfinite": self.n_nonfinite,
                "other_status": self.n_other_status,
            },
        }


def _validated_e_align(record: dict) -> Dyadic | None:
    """Exact E_align of a record, or ``None`` when it cannot be trusted.

    Snapshot records carry the complete prefix ``E_align`` by construction
    (aborted samples are written as separate abort records and never reach
    this function), so v1/v2 records are accepted when the field is a
    well-formed dyadic.  A record that explicitly marks ``E_align_complete``
    false, or whose field is missing, null or malformed, is invalid -- it is
    never defaulted to a zero error.
    """
    if record.get("E_align_complete") is False:
        return None
    payload = record.get("E_align")
    if not isinstance(payload, dict) or "m" not in payload or "e" not in payload:
        return None
    try:
        return Dyadic.from_dict(payload)
    except (TypeError, ValueError):
        return None


class SummaryBuilder:
    def __init__(self, mode: str, x_name: str):
        if x_name not in ("W", "B"):
            raise ValueError("x_name must be 'W' or 'B'")
        self.mode = mode
        self.x_name = x_name
        self.cells: dict[tuple[str, str, int], Cell] = {}

    def add(self, record: dict) -> None:
        if record.get("mode") != self.mode:
            raise ValueError(f"record mode {record.get('mode')!r} does not match builder mode {self.mode!r}")
        x = int(record[self.x_name])
        key = (record["format"], record["distribution"], x)
        cell = self.cells.get(key)
        if cell is None:
            cell = Cell(format=record["format"], distribution=record["distribution"], x=x)
            self.cells[key] = cell
        cell.add_record(record)

    def finalize(self, base: dict) -> dict:
        summary = dict(base)
        summary["x_name"] = self.x_name
        summary["x_values"] = sorted({key[2] for key in self.cells})
        summary["cells"] = [self.cells[key].finalize() for key in sorted(self.cells)]
        return summary
