"""Aligned candidate evaluation: fixed-W_align dot16 path on MXFP8 inputs.

For each output element two trajectories are maintained in addition to the
exact reference ``S`` and the no-window FP32 baseline ``R``:

* ``T``: exact sum of the aligned partial sums ``q_b`` (never converted to FP32);
* ``A``: FP32 accumulation of the same ``q_b`` with one RNE per dot16:
  ``a_{b+1} = RN32(exact(a_b) + q_b)``.

``S``, ``L1`` and ``R`` use exactly the same definitions as
``engine.evaluate_mx_pair`` (which is untouched); the aligned path only adds the
per-dot16 integer alignment from ``matrix/alignment.py``.  One G32 group holds
two dot16 blocks; each block selects its own ``Cmax``, and the two blocks update
the FP32 accumulators separately (never merged into a dot32).

Recorded per snapshot: exact ``S``/``L1``/``T`` (dyadic), ``r_bits``, ``a_bits``,
cumulative alignment counters/histograms, and per-element cumulative and
maximum alignment-error magnitudes ``q_b - s_b``.  Event counters are
cumulative over k up to the snapshot K; offline analysis must use the snapshot
at K (or differences between snapshots) and never sum across K.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..accumulation.accumulator import Fp32Accumulator
from ..arithmetic.dyadic import Dyadic
from ..formats.e8m0 import code_to_exponent
from ..version import MX_ALIGN_DEFAULT
from .alignment import aligned_container_bounds, align_dot16, decode_magnitude, summarize_trace
from .engine import MAC_PARALLELISM
from .tensor import GROUP_LENGTH, MxTensor


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


@dataclass
class AlignedPairResult:
    snapshots: list[dict]
    status: str
    blocks_completed: int


def _validate(x: MxTensor, w: MxTensor, k_values: list[int], mac_parallelism: int, group_length: int, w_align: int):
    if x.group_axis != 1 or w.group_axis != 0:
        raise ValueError("MX inputs must group X along K (axis 1) and W along K (axis 0)")
    if x.cols != w.rows:
        raise ValueError(f"contraction mismatch: X is {x.shape}, W is {w.shape}")
    if x.cols % group_length or w.rows % group_length:
        raise ValueError("contraction dimension must be a multiple of the group length")
    snapshots = sorted({int(k) for k in k_values})
    if not snapshots:
        raise ValueError("at least one K value is required")
    kmax = max(snapshots)
    if x.cols != kmax:
        raise ValueError(f"X has K={x.cols} but the maximum requested K is {kmax}")
    if any(k % group_length for k in snapshots):
        raise ValueError("all K values must be multiples of the group length")
    if any(k % mac_parallelism for k in snapshots):
        raise ValueError("all K values must be multiples of the MAC parallelism")
    if group_length % mac_parallelism:
        raise ValueError("group length must be a multiple of the MAC parallelism")
    if not isinstance(w_align, int) or w_align < 1 or w_align > 60:
        raise ValueError(f"w_align must be an integer in [1, 60], got {w_align!r}")
    aligned_container_bounds(w_align)  # validation
    return snapshots, kmax


def evaluate_mx_pair_aligned(
    x: MxTensor,
    w: MxTensor,
    k_values: list[int],
    w_align: int = MX_ALIGN_DEFAULT,
    mac_parallelism: int = MAC_PARALLELISM,
    group_length: int = GROUP_LENGTH,
    record_alignment_arrays: bool = True,
) -> AlignedPairResult:
    snapshots_list, kmax = _validate(x, w, k_values, mac_parallelism, group_length, w_align)
    M, N = x.rows, w.cols
    grid = M * N
    kvalues_set = set(snapshots_list)
    dot16 = mac_parallelism
    group = group_length

    total_s = [Dyadic.zero() for _ in range(grid)]
    total_l1 = [Dyadic.zero() for _ in range(grid)]
    total_t = [Dyadic.zero() for _ in range(grid)]
    align_abs_sum = [Dyadic.zero() for _ in range(grid)]
    align_abs_max = [Dyadic.zero() for _ in range(grid)]
    r_acc = [Fp32Accumulator() for _ in range(grid)]
    a_acc = [Fp32Accumulator() for _ in range(grid)]

    counters = Counter()
    cmax_histogram: Counter = Counter()

    records: list[dict] = []
    blocks_done = 0
    for block in range(kmax // dot16):
        g = (block * dot16) // group
        xt_signed: list[list[int]] = []
        xt_e: list[list[int]] = []
        wt_signed: list[list[int]] = []
        wt_e: list[list[int]] = []
        for i in range(dot16):
            k = block * dot16 + i
            x_row: list[int] = []
            x_exp: list[int] = []
            for m in range(M):
                sign, t, e = decode_magnitude(x.element_codes[m * x.cols + k])
                if t == 0:
                    x_row.append(0)
                    x_exp.append(0)
                else:
                    x_row.append(-t if sign else t)
                    x_exp.append(e)
            w_row: list[int] = []
            w_exp: list[int] = []
            for n in range(N):
                sign, t, e = decode_magnitude(w.element_codes[k * N + n])
                if t == 0:
                    w_row.append(0)
                    w_exp.append(0)
                else:
                    w_row.append(-t if sign else t)
                    w_exp.append(e)
            xt_signed.append(x_row)
            xt_e.append(x_exp)
            wt_signed.append(w_row)
            wt_e.append(w_exp)

        for m in range(M):
            base = m * N
            alpha = code_to_exponent(x.scale_codes[m * (x.cols // group) + g])
            for n in range(N):
                products: list[tuple[int, int]] = []
                for i in range(dot16):
                    a = xt_signed[i][m]
                    b = wt_signed[i][n]
                    if a and b:
                        products.append((a * b, xt_e[i][m] + wt_e[i][n]))
                beta = code_to_exponent(w.scale_codes[g * N + n])
                aligned = align_dot16(products, alpha + beta, w_align)
                index = base + n
                total_s[index] = total_s[index] + aligned.s_b
                total_l1[index] = total_l1[index] + aligned.l1_b
                total_t[index] = total_t[index] + aligned.q_b

                counters["dot16_total"] += 1
                counters["products_total"] += dot16
                counters["products_nonzero"] += len(products)
                if aligned.all_zero:
                    counters["dot16_all_zero"] += 1
                else:
                    changed = aligned.changed_count
                    zeroed = aligned.zeroed_count
                    counters["products_changed"] += changed
                    counters["products_zeroed"] += zeroed
                    if changed:
                        counters["dot16_with_changed_product"] += 1
                    for diff in aligned.cmax_minus_ci:
                        cmax_histogram[str(diff)] += 1

                if aligned.q_b != aligned.s_b:
                    counters["dot16_qb_diff"] += 1
                    error = aligned.q_b - aligned.s_b
                    magnitude = abs(error)
                    align_abs_sum[index] = align_abs_sum[index] + magnitude
                    if align_abs_max[index] < magnitude:
                        align_abs_max[index] = magnitude

                r_outcome = r_acc[index].update(aligned.s_b)
                counters["r_updates"] += 1
                if r_outcome.non_finite:
                    counters["r_nonfinite_updates"] += 1
                if r_outcome.overflowed:
                    counters["r_overflow_updates"] += 1
                if r_outcome.absorbed:
                    counters["r_absorbed_updates"] += 1
                a_outcome = a_acc[index].update(aligned.q_b)
                counters["a_updates"] += 1
                if a_outcome.non_finite:
                    counters["a_nonfinite_updates"] += 1
                if a_outcome.overflowed:
                    counters["a_overflow_updates"] += 1
                if a_outcome.absorbed:
                    counters["a_absorbed_updates"] += 1

        blocks_done += 1
        K_completed = blocks_done * dot16
        if K_completed in kvalues_set:
            record = _snapshot(
                M,
                N,
                K_completed,
                dot16,
                group,
                w_align,
                total_s,
                total_l1,
                total_t,
                r_acc,
                a_acc,
                counters,
                cmax_histogram,
                align_abs_sum,
                align_abs_max,
                record_alignment_arrays,
            )
            records.append(record)

    r_nonfinite = sum(1 for acc in r_acc if not acc.state.is_finite())
    a_nonfinite = sum(1 for acc in a_acc if not acc.state.is_finite())
    if r_nonfinite == 0 and a_nonfinite == 0:
        status = "ok"
    elif r_nonfinite == grid and a_nonfinite == grid:
        status = "all_nonfinite"
    else:
        status = "partial_nonfinite"
    return AlignedPairResult(snapshots=records, status=status, blocks_completed=blocks_done)


def _snapshot(
    M,
    N,
    K,
    dot16,
    group,
    w_align,
    total_s,
    total_l1,
    total_t,
    r_acc,
    a_acc,
    counters,
    cmax_histogram,
    align_abs_sum,
    align_abs_max,
    record_alignment_arrays,
):
    grid = M * N
    r_bits = [acc.state.bits for acc in r_acc]
    a_bits = [acc.state.bits for acc in a_acc]
    r_nonfinite = sum(1 for acc in r_acc if not acc.state.is_finite())
    a_nonfinite = sum(1 for acc in a_acc if not acc.state.is_finite())
    if r_nonfinite == 0 and a_nonfinite == 0:
        status = "ok"
    elif r_nonfinite == grid and a_nonfinite == grid:
        status = "all_nonfinite"
    else:
        status = "partial_nonfinite"
    payload = {
        "record_type": "align_snapshot",
        "M": M,
        "N": N,
        "K": K,
        "B": K // dot16,
        "group_length": group,
        "mac_parallelism": dot16,
        "w_align": w_align,
        "container_bits": w_align + 7,
        "S": [value.to_dict() for value in total_s],
        "L1": [value.to_dict() for value in total_l1],
        "T": [value.to_dict() for value in total_t],
        "r_bits": r_bits,
        "a_bits": a_bits,
        "r_nonfinite_count": r_nonfinite,
        "a_nonfinite_count": a_nonfinite,
        "events_cumulative": {key: int(counters.get(key, 0)) for key in ALIGN_EVENT_KEYS},
        "cmax_minus_ci_histogram_cumulative": {
            key: int(cmax_histogram[key]) for key in sorted(cmax_histogram, key=lambda item: int(item))
        },
        "status": status,
    }
    if record_alignment_arrays:
        payload["align_abs_sum"] = [value.to_dict() for value in align_abs_sum]
        payload["align_abs_max"] = [value.to_dict() for value in align_abs_max]
    return payload


def trace_dot16(
    x: MxTensor,
    w: MxTensor,
    row: int,
    col: int,
    block: int,
    w_align: int = MX_ALIGN_DEFAULT,
    mac_parallelism: int = MAC_PARALLELISM,
    group_length: int = GROUP_LENGTH,
) -> dict:
    """Reproduce one output dot16 (``sample_id`` filtering happens in the CLI)."""
    if x.group_axis != 1 or w.group_axis != 0:
        raise ValueError("MX inputs must group X along K (axis 1) and W along K (axis 0)")
    if not (0 <= row < x.rows and 0 <= col < w.cols):
        raise ValueError(f"output coordinate ({row}, {col}) outside {x.rows}x{w.cols}")
    n_blocks = x.cols // mac_parallelism
    if not 0 <= block < n_blocks:
        raise ValueError(f"block {block} outside [0, {n_blocks})")
    g = (block * mac_parallelism) // group_length
    alpha = code_to_exponent(x.scale_codes[row * (x.cols // group_length) + g])
    beta = code_to_exponent(w.scale_codes[g * w.cols + col])
    products: list[tuple[int, int]] = []
    for i in range(mac_parallelism):
        k = block * mac_parallelism + i
        sx, tx, ex = decode_magnitude(x.element_codes[row * x.cols + k])
        sw, tw, ew = decode_magnitude(w.element_codes[k * w.cols + col])
        if tx and tw:
            p = tx * tw
            if sx ^ sw:
                p = -p
            products.append((p, ex + ew))
    aligned = align_dot16(products, alpha + beta, w_align)
    payload = summarize_trace(aligned, w_align)
    payload.update(
        {
            "sample_row": row,
            "sample_col": col,
            "block": block,
            "k_begin": block * mac_parallelism,
            "k_end": (block + 1) * mac_parallelism - 1,
            "w_align": w_align,
            "h": alpha + beta,
            "alpha": alpha,
            "beta": beta,
        }
    )
    return payload
