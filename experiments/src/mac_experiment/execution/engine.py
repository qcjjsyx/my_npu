"""Streaming execution of one sample sequence with prefix snapshots.

For a fixed sample the engine walks ``Bmax`` blocks.  Each block contributes
16 exact products; the exact block sum ``s_b`` feeds the window-loss-free
reference path and the candidate accumulators (one per W in the run).  The
accumulator state is FP32; the partial sum is exact and is added exactly before
the single FP32 RNE step.  Snapshot records are emitted at the requested B
prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..accumulation.accumulator import Fp32Accumulator
from ..arithmetic.dyadic import Dyadic
from ..events import EVENT_KEYS, zero_events
from ..formats.base import Format
from ..reduction.reducer import BlockReducer, ProductTerm, WindowOverflowError
from ..reference.exact import ReferencePath, exact_sum, l1_sum
from ..sampling.sampler import BlockDraw, Sampler
from ..sampling.streams import CodeFingerprint
from ..version import SCHEMA_VERSION


@dataclass
class SampleOutcome:
    records: list[dict]
    status: str
    blocks_completed: int


class SampleEngine:
    def __init__(
        self,
        fmt: Format,
        sampler: Sampler,
        windows: Sequence[int],
        snapshots: Sequence[int],
        mode: str,
        record_block_rho: bool = False,
    ):
        self.fmt = fmt
        self.sampler = sampler
        self.windows = list(windows)
        self.snapshots = sorted(set(snapshots))
        if not self.snapshots:
            raise ValueError("at least one snapshot is required")
        self.mode = mode
        self.record_block_rho = record_block_rho
        self.bmax = max(self.snapshots)
        self.reducer = BlockReducer(window_offset=fmt.window_offset)

    def run(self, master_seed: int, sample_index: int) -> SampleOutcome:
        fmt = self.fmt
        sample_id = f"{fmt.name}|{self.sampler.spec.name}|seed{master_seed}|{sample_index:05d}"
        records: list[dict] = []
        accumulators = {w: Fp32Accumulator() for w in self.windows}
        e_align = {w: Dyadic.zero() for w in self.windows}
        e_acc = {w: Dyadic.zero() for w in self.windows}
        e_acc_complete = {w: True for w in self.windows}
        e_acc_last_block = {w: None for w in self.windows}
        events_cum = {w: zero_events() for w in self.windows}
        endpoint_events = {w: zero_events() for w in self.windows}
        aborted_at: dict[int, int] = {}
        reference = ReferencePath()
        total_sum = Dyadic.zero()
        l1_total = Dyadic.zero()
        fingerprint = CodeFingerprint()
        rho_values: list[float] = []
        blocks_completed = 0
        snapshot_set = set(self.snapshots)

        for block_index in range(self.bmax):
            draw = self.sampler.sample_block(master_seed, sample_index, block_index)
            terms = self._build_terms(draw)
            for code_a, code_b in draw.pairs:
                fingerprint.update(code_a)
                fingerprint.update(code_b)
            s_b = exact_sum(term.dyadic for term in terms)
            l1_b = l1_sum(term.dyadic for term in terms)
            total_sum = total_sum + s_b
            l1_total = l1_total + l1_b
            reference.update(s_b)
            r_state = reference.state
            if draw.rho is not None:
                rho_values.append(draw.rho)

            for w in self.windows:
                endpoint_events[w] = zero_events()
                if w in aborted_at:
                    continue
                try:
                    reduction = self.reducer.reduce(terms, w)
                except WindowOverflowError:
                    aborted_at[w] = block_index
                    events_cum[w]["window_overflow"] += 1
                    endpoint_events[w]["window_overflow"] += 1
                    records.append(
                        self._abort_record(
                            sample_id, master_seed, sample_index, w, block_index, fingerprint
                        )
                    )
                    continue
                outcome = accumulators[w].update(reduction.aligned_sum)
                e_align[w] = e_align[w] + (reduction.aligned_sum - s_b)
                if outcome.delta is not None:
                    e_acc[w] = e_acc[w] + outcome.delta
                    e_acc_last_block[w] = block_index
                else:
                    e_acc_complete[w] = False

                events = endpoint_events[w]
                events["alignment_zeroed"] += reduction.zeroed_terms
                if reduction.all_zero:
                    events["all_zero_block"] += 1
                if outcome.non_finite or not r_state.is_finite():
                    events["non_finite_fp32"] += 1
                if outcome.fp32.is_zero() and not r_state.is_zero():
                    events["unexpected_zero"] += 1
                if outcome.absorbed:
                    events["update_absorbed"] += 1
                for key in EVENT_KEYS:
                    events_cum[w][key] += events[key]

            blocks_completed = block_index + 1
            if blocks_completed in snapshot_set:
                for w in self.windows:
                    if w in aborted_at:
                        continue
                    records.append(
                        self._snapshot_record(
                            sample_id=sample_id,
                            master_seed=master_seed,
                            sample_index=sample_index,
                            w=w,
                            b=blocks_completed,
                            accumulators=accumulators,
                            e_align=e_align,
                            e_acc=e_acc,
                            e_acc_complete=e_acc_complete,
                            e_acc_last_block=e_acc_last_block,
                            total_sum=total_sum,
                            l1_total=l1_total,
                            reference=reference,
                            events_cum=events_cum,
                            endpoint_events=endpoint_events,
                            fingerprint=fingerprint,
                            rho_values=rho_values,
                        )
                    )

        status = "ok"
        if aborted_at:
            status = "aborted_window_overflow"
        elif not reference.state.is_finite() or any(
            not accumulators[w].state.is_finite() for w in self.windows
        ):
            status = "nonfinite_fp32"
        return SampleOutcome(records=records, status=status, blocks_completed=blocks_completed)

    def _build_terms(self, draw: BlockDraw) -> list[ProductTerm]:
        terms = []
        for code_a, code_b in draw.pairs:
            product = self.fmt.product(self.fmt.decode(code_a), self.fmt.decode(code_b))
            terms.append(ProductTerm(product.dyadic, product.c))
        return terms

    def _base_record(
        self,
        record_type: str,
        sample_id: str,
        master_seed: int,
        sample_index: int,
        fingerprint: CodeFingerprint,
    ) -> dict:
        return {
            "record_type": record_type,
            "schema": SCHEMA_VERSION,
            "mode": self.mode,
            "sample_id": sample_id,
            "seed": master_seed,
            "sample_index": sample_index,
            "format": self.fmt.name,
            "distribution": self.sampler.spec.name,
            "input_fingerprint": fingerprint.hexdigest(),
        }

    def _abort_record(
        self,
        sample_id: str,
        master_seed: int,
        sample_index: int,
        w: int,
        block_index: int,
        fingerprint: CodeFingerprint,
    ) -> dict:
        record = self._base_record("abort", sample_id, master_seed, sample_index, fingerprint)
        record.update(
            {
                "W": w,
                "B": block_index + 1,
                "K": 16 * (block_index + 1),
                "status": "aborted_window_overflow",
                "abort_reason": "window_integer_overflow",
            }
        )
        return record

    def _snapshot_record(
        self,
        sample_id: str,
        master_seed: int,
        sample_index: int,
        w: int,
        b: int,
        accumulators: dict,
        e_align: dict,
        e_acc: dict,
        e_acc_complete: dict,
        e_acc_last_block: dict,
        total_sum: Dyadic,
        l1_total: Dyadic,
        reference: ReferencePath,
        events_cum: dict,
        endpoint_events: dict,
        fingerprint: CodeFingerprint,
        rho_values: list[float],
    ) -> dict:
        a_state = accumulators[w].state
        r_state = reference.state
        a_finite = a_state.is_finite()
        r_finite = r_state.is_finite()
        acc_complete = e_acc_complete[w]
        if a_finite and r_finite and acc_complete:
            lhs = a_state.exact_dyadic() - total_sum
            rhs = e_align[w] + e_acc[w]
            if lhs != rhs:
                raise AssertionError(
                    f"error decomposition identity violated for {sample_id} W={w} B={b}"
                )
        record = self._base_record("snapshot", sample_id, master_seed, sample_index, fingerprint)
        record.update(
            {
                "W": w,
                "B": b,
                "K": 16 * b,
                "a_bits": a_state.bits,
                "r_bits": r_state.bits,
                "a_finite": a_finite,
                "r_finite": r_finite,
                "S": total_sum.to_dict(),
                "L1": l1_total.to_dict(),
                "E_align": e_align[w].to_dict(),
                "E_acc": e_acc[w].to_dict() if acc_complete else None,
                "E_acc_complete": acc_complete,
                "E_acc_prefix": None if acc_complete else e_acc[w].to_dict(),
                "E_acc_prefix_until_block": None if acc_complete else e_acc_last_block[w],
                "events_cumulative": dict(events_cum[w]),
                "events_endpoint": dict(endpoint_events[w]),
                "rho": _rho_summary(rho_values),
                "rho_per_block": list(rho_values) if self.record_block_rho else None,
                "status": "ok" if (a_finite and r_finite) else "nonfinite_fp32",
            }
        )
        return record


def _rho_summary(values: Iterable[float]) -> dict | None:
    values = list(values)
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "blocks": len(values),
    }
