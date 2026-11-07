"""Experiment 1 engine: finite-W_align dot16 alignment with exact accumulation.

For every output element the 16 exact products of each dot16 are aligned into a
``W_align``-bit window (format-parameterized constants, see
``docs/mac_width_selection.md`` section 2), summed as exact integers and
recovered as an exact dyadic ``q_b``.  The candidate trajectory is the *exact*
sum ``T = sum_b q_b`` (never rounded to FP32 per update); ``Y_align = RN32(T)``
is produced only at the requested K snapshots, and snapshotting never mutates
the exact accumulation state.

The E4M3 constants reproduce the historical ``matrix/alignment.py`` formula
bit-for-bit (covered by an oracle test); TF32 uses ``k_shift = 20`` with the
same container width.  Nothing here changes historical semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import rne_shift, round_dyadic_to_fp32
from ..formats.e8m0 import code_to_exponent
from ..matrix.alignment import MX_ALIGN_CONTAINER_EXTRA_BITS
from .acc_format import (
    ACC_EVENT_KEYS,
    decode_state,
    hw_add,
    output_bits,
    validate_w_acc,
    zero_state,
)
from .tensors import TargetTensor

MAC_PARALLELISM = 16
GROUP_LENGTH = 32

FORMAT_ALIGNMENT = {
    "mx_fp8": {"k_shift": 6, "window_offset": 14, "product_lsb_offset": 20},
    "tf32": {"k_shift": 20, "window_offset": 254, "product_lsb_offset": 274},
}

W_ALIGN_MIN = 1
W_ALIGN_MAX = 60


def format_constants(fmt: str) -> dict:
    if fmt not in FORMAT_ALIGNMENT:
        raise ValueError(f"unknown target format {fmt!r}")
    return FORMAT_ALIGNMENT[fmt]


def validate_w_align(w_align: int) -> None:
    if not isinstance(w_align, int) or w_align < W_ALIGN_MIN or w_align > W_ALIGN_MAX:
        raise ValueError(f"W_align must be an integer in [{W_ALIGN_MIN}, {W_ALIGN_MAX}], got {w_align!r}")


def container_bounds(w_align: int) -> tuple[int, int]:
    validate_w_align(w_align)
    bits = w_align + MX_ALIGN_CONTAINER_EXTRA_BITS
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def check_container(value: int, w_align: int, label: str) -> None:
    low, high = container_bounds(w_align)
    if not low <= value <= high:
        bits = w_align + MX_ALIGN_CONTAINER_EXTRA_BITS
        raise ArithmeticError(
            f"{label} {value} outside the signed {bits}-bit alignment container "
            f"[{low}, {high}] at W_align={w_align}; refusing to wrap/truncate/saturate"
        )


@dataclass
class BlockAlignment:
    all_zero: bool
    cmax: int | None
    integer_sum: int
    q_b: Dyadic
    s_b: Dyadic
    l1_b: Dyadic
    changed: int = 0
    zeroed: int = 0
    products: list[tuple[int, int]] = field(default_factory=list)

    @property
    def qb_differs(self) -> bool:
        return self.q_b != self.s_b


def exact_sums(products: list[tuple[int, int]], h: int, fmt: str) -> tuple[Dyadic, Dyadic]:
    """Exact ``(s_b, l1_b)`` of one dot16 over decoded products."""
    offset = format_constants(fmt)["product_lsb_offset"]
    s_b = Dyadic.zero()
    l1_b = Dyadic.zero()
    for p, c in products:
        value = Dyadic(p, c - offset + h)
        s_b = s_b + value
        l1_b = l1_b + abs(value)
    return s_b, l1_b


def _align_sum(products: list[tuple[int, int]], cmax: int, h: int, w_align: int, fmt: str) -> tuple[int, int, int, Dyadic]:
    constants = format_constants(fmt)
    k_shift = constants["k_shift"]
    window_offset = constants["window_offset"]
    integer_sum = 0
    changed = 0
    zeroed = 0
    for p, c in products:
        shift = (cmax - c) + k_shift - w_align
        value = rne_shift(p, shift)
        check_container(value, w_align, "aligned product")
        integer_sum += value
        if shift > 0 and (value << shift) != p:
            changed += 1
        if value == 0:
            zeroed += 1
    check_container(integer_sum, w_align, "aligned dot16 integer sum")
    q_b = Dyadic(integer_sum, cmax - window_offset - w_align + h)
    return integer_sum, changed, zeroed, q_b


def align_block(products: list[tuple[int, int]], h: int, w_align: int, fmt: str) -> BlockAlignment:
    """Align and sum one dot16 (exact integer add, exact ``q_b`` recovery)."""
    validate_w_align(w_align)
    s_b, l1_b = exact_sums(products, h, fmt)
    if not products:
        return BlockAlignment(True, None, 0, Dyadic.zero(), s_b, l1_b)
    cmax = max(c for _, c in products)
    integer_sum, changed, zeroed, q_b = _align_sum(products, cmax, h, w_align, fmt)
    return BlockAlignment(False, cmax, integer_sum, q_b, s_b, l1_b, changed, zeroed, products)


def align_block_trace(products: list[tuple[int, int]], h: int, w_align: int, fmt: str) -> dict:
    """Same computation with the full per-term diagnostic trace (replay tool)."""
    validate_w_align(w_align)
    constants = format_constants(fmt)
    k_shift = constants["k_shift"]
    window_offset = constants["window_offset"]
    s_b, l1_b = exact_sums(products, h, fmt)
    if not products:
        return {
            "all_zero": True,
            "cmax": None,
            "h": h,
            "k_shift": k_shift,
            "terms": [],
            "integer_sum": 0,
            "s_b": s_b.to_dict(),
            "q_b": Dyadic.zero().to_dict(),
            "l1_b": l1_b.to_dict(),
            "q_b_equals_s_b": True,
            "changed": 0,
            "zeroed": 0,
        }
    cmax = max(c for _, c in products)
    integer_sum = 0
    changed = 0
    zeroed = 0
    terms = []
    for p, c in products:
        shift = (cmax - c) + k_shift - w_align
        n = rne_shift(p, shift)
        check_container(n, w_align, "aligned product")
        term_changed = ((n << shift) != p) if shift > 0 else False
        integer_sum += n
        changed += int(term_changed)
        zeroed += int(n == 0)
        terms.append(
            {
                "P": p,
                "C": c,
                "cmax_minus_C": cmax - c,
                "shift": shift,
                "n": n,
                "changed": term_changed,
                "zeroed": n == 0,
            }
        )
    check_container(integer_sum, w_align, "aligned dot16 integer sum")
    q_b = Dyadic(integer_sum, cmax - window_offset - w_align + h)
    return {
        "all_zero": False,
        "cmax": cmax,
        "h": h,
        "k_shift": k_shift,
        "terms": terms,
        "integer_sum": integer_sum,
        "s_b": s_b.to_dict(),
        "q_b": q_b.to_dict(),
        "l1_b": l1_b.to_dict(),
        "q_b_equals_s_b": q_b == s_b,
        "changed": changed,
        "zeroed": zeroed,
    }


def decompose_operand(fmt: str, code: int) -> tuple[int, int]:
    """``(signed_magnitude, scale_free_exponent)`` of one finite operand."""
    if fmt == "mx_fp8":
        from ..matrix.alignment import decode_magnitude

        sign, t, e = decode_magnitude(code)
        if t == 0:
            return 0, 0
        return (-t if sign else t), e
    from .tf32_rne import decode_tf32_magnitude

    sign, t, e = decode_tf32_magnitude(code)
    if t == 0:
        return 0, 0
    return (-t if sign else t), e


def block_products(
    x: TargetTensor,
    w: TargetTensor,
    row: int,
    col: int,
    block: int,
    fmt: str,
) -> list[tuple[int, int]]:
    """Non-zero exact products ``(P_i, C_i)`` of one output dot16."""
    products: list[tuple[int, int]] = []
    for i in range(MAC_PARALLELISM):
        k = block * MAC_PARALLELISM + i
        a, ea = decompose_operand(fmt, x.codes[row * x.cols + k])
        b, eb = decompose_operand(fmt, w.codes[k * w.cols + col])
        if a and b:
            products.append((a * b, ea + eb))
    return products


def block_scale_exponent(x: TargetTensor, w: TargetTensor, row: int, col: int, block: int) -> int:
    if x.format == "tf32":
        return 0
    group = GROUP_LENGTH
    g = (block * MAC_PARALLELISM) // group
    alpha = code_to_exponent(x.scale_codes[row * (x.cols // group) + g])
    beta = code_to_exponent(w.scale_codes[g * w.cols + col])
    return alpha + beta


def _block_operands(
    x: TargetTensor,
    w: TargetTensor,
    block: int,
    fmt: str,
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Per-timestep decomposed X and W operands for one dot16 block."""
    x_terms: list[list[tuple[int, int]]] = []
    w_terms: list[list[tuple[int, int]]] = []
    for i in range(MAC_PARALLELISM):
        k = block * MAC_PARALLELISM + i
        x_row = [decompose_operand(fmt, x.codes[m * x.cols + k]) for m in range(x.rows)]
        w_col = [decompose_operand(fmt, w.codes[k * w.cols + n]) for n in range(w.cols)]
        x_terms.append(x_row)
        w_terms.append(w_col)
    return x_terms, w_terms


def _block_scales(x: TargetTensor, w: TargetTensor, block: int) -> tuple[list[int] | None, list[int] | None]:
    if x.format == "tf32":
        return None, None
    group = GROUP_LENGTH
    g = (block * MAC_PARALLELISM) // group
    grid_cols = x.cols // group
    alpha = [code_to_exponent(x.scale_codes[m * grid_cols + g]) for m in range(x.rows)]
    beta = [code_to_exponent(w.scale_codes[g * w.cols + n]) for n in range(w.cols)]
    return alpha, beta


@dataclass
class PairAlignResult:
    base_snapshots: list[dict]
    width_snapshots: list[dict]
    status: str
    blocks_completed: int


EVENT_KEYS = (
    "products_total",
    "products_nonzero",
    "products_changed",
    "products_zeroed",
    "dot16_total",
    "dot16_all_zero",
    "dot16_with_changed_product",
    "dot16_qb_diff",
)


def validate_pair(x: TargetTensor, w: TargetTensor, k_values: list[int]) -> list[int]:
    if x.cols != w.rows:
        raise ValueError(f"contraction mismatch: X is {x.rows}x{x.cols}, W is {w.rows}x{w.cols}")
    snapshots = sorted({int(k) for k in k_values})
    if not snapshots:
        raise ValueError("at least one K value is required")
    kmax = max(snapshots)
    if x.cols != kmax:
        raise ValueError(f"X has K={x.cols} but the maximum requested K is {kmax}")
    for k in snapshots:
        if k < MAC_PARALLELISM or k % MAC_PARALLELISM:
            raise ValueError(f"K={k} must be a positive multiple of {MAC_PARALLELISM}")
        if x.format == "mx_fp8" and k % GROUP_LENGTH:
            raise ValueError(f"K={k} must be a multiple of the MXFP8 group length {GROUP_LENGTH}")
    return snapshots


def _nonfinite_count(bits_list: list[int]) -> int:
    return sum(1 for bits in bits_list if ((bits >> 23) & 0xFF) == 0xFF)


def _status(nonfinite: int, grid: int) -> str:
    if nonfinite == 0:
        return "ok"
    if nonfinite == grid:
        return "all_nonfinite"
    return "partial_nonfinite"


def _base_record(K: int, S: list[Dyadic], L1: list[Dyadic], grid: int) -> dict:
    o_bits = []
    for value in S:
        bits, _ = round_dyadic_to_fp32(value)
        o_bits.append(bits.bits)
    o_nonfinite = _nonfinite_count(o_bits)
    return {
        "record_type": "widthselect_base_snapshot",
        "K": K,
        "B": K // MAC_PARALLELISM,
        "S": [value.to_dict() for value in S],
        "L1": [value.to_dict() for value in L1],
        "O_bits": o_bits,
        "o_nonfinite_count": o_nonfinite,
        "status": _status(o_nonfinite, grid),
    }


def evaluate_pair_align(
    x: TargetTensor,
    w: TargetTensor,
    k_values: list[int],
    w_align_list: list[int],
) -> PairAlignResult:
    """One streaming pass to Kmax with exact T per W_align and K snapshots."""
    if x.format != w.format:
        raise ValueError("X and W must use the same target format")
    fmt = x.format
    widths = list(w_align_list)
    if not widths:
        raise ValueError("at least one W_align value is required")
    for value in widths:
        validate_w_align(value)
    snapshots = validate_pair(x, w, k_values)
    kmax = max(snapshots)
    M, N = x.rows, w.cols
    grid = M * N
    kvalues_set = set(snapshots)

    S = [Dyadic.zero() for _ in range(grid)]
    L1 = [Dyadic.zero() for _ in range(grid)]
    T = {width: [Dyadic.zero() for _ in range(grid)] for width in widths}
    counters = {width: {key: 0 for key in EVENT_KEYS} for width in widths}

    base_records: list[dict] = []
    width_records: list[dict] = []
    blocks_done = 0
    last_o_nonfinite = 0

    for block in range(kmax // MAC_PARALLELISM):
        x_terms, w_terms = _block_operands(x, w, block, fmt)
        alpha, beta = _block_scales(x, w, block)
        for m in range(M):
            base = m * N
            for n in range(N):
                h = 0 if alpha is None else alpha[m] + beta[n]
                products: list[tuple[int, int]] = []
                for i in range(MAC_PARALLELISM):
                    a, ea = x_terms[i][m]
                    b, eb = w_terms[i][n]
                    if a and b:
                        products.append((a * b, ea + eb))
                index = base + n
                s_b, l1_b = exact_sums(products, h, fmt)
                S[index] = S[index] + s_b
                L1[index] = L1[index] + l1_b
                cmax = max((c for _, c in products), default=None)
                for width in widths:
                    bucket = counters[width]
                    bucket["products_total"] += MAC_PARALLELISM
                    bucket["products_nonzero"] += len(products)
                    bucket["dot16_total"] += 1
                    if cmax is None:
                        bucket["dot16_all_zero"] += 1
                        continue
                    integer_sum, changed, zeroed, q_b = _align_sum(products, cmax, h, width, fmt)
                    T[width][index] = T[width][index] + q_b
                    if changed:
                        bucket["products_changed"] += changed
                        bucket["dot16_with_changed_product"] += 1
                    if zeroed:
                        bucket["products_zeroed"] += zeroed
                    if q_b != s_b:
                        bucket["dot16_qb_diff"] += 1

        blocks_done += 1
        K_completed = blocks_done * MAC_PARALLELISM
        if K_completed not in kvalues_set:
            continue

        record = _base_record(K_completed, S, L1, grid)
        last_o_nonfinite = record["o_nonfinite_count"]
        base_records.append(record)
        for width in widths:
            y_bits = []
            for value in T[width]:
                bits, _ = round_dyadic_to_fp32(value)
                y_bits.append(bits.bits)
            y_nonfinite = _nonfinite_count(y_bits)
            width_records.append(
                {
                    "record_type": "widthselect_align_snapshot",
                    "K": K_completed,
                    "B": K_completed // MAC_PARALLELISM,
                    "w_align": width,
                    "T": [value.to_dict() for value in T[width]],
                    "Y_bits": y_bits,
                    "y_nonfinite_count": y_nonfinite,
                    "events_cumulative": dict(counters[width]),
                    "status": _status(y_nonfinite, grid),
                }
            )

    return PairAlignResult(
        base_snapshots=base_records,
        width_snapshots=width_records,
        status=_status(last_o_nonfinite, grid),
        blocks_completed=blocks_done,
    )


@dataclass
class PairAccResult:
    base_snapshots: list[dict]
    acc_snapshots: list[dict]
    status: str
    blocks_completed: int


JOINT_EVENT_KEYS = ACC_EVENT_KEYS + ("align_products_changed", "align_dot16_qb_diff")


def evaluate_pair_acc(
    x: TargetTensor,
    w: TargetTensor,
    k_values: list[int],
    w_acc_list: list[int],
) -> PairAccResult:
    """Experiment 2: exact ``s_b`` fed through ``HWAdd_{W_acc}`` per dot16."""
    if x.format != w.format:
        raise ValueError("X and W must use the same target format")
    fmt = x.format
    widths = sorted({int(value) for value in w_acc_list})
    if not widths:
        raise ValueError("at least one W_acc value is required")
    for value in widths:
        validate_w_acc(value)
    snapshots = validate_pair(x, w, k_values)
    kmax = max(snapshots)
    M, N = x.rows, w.cols
    grid = M * N
    kvalues_set = set(snapshots)

    S = [Dyadic.zero() for _ in range(grid)]
    L1 = [Dyadic.zero() for _ in range(grid)]
    states = {width: [zero_state(width) for _ in range(grid)] for width in widths}
    counters = {width: {key: 0 for key in ACC_EVENT_KEYS} for width in widths}
    base_records: list[dict] = []
    acc_records: list[dict] = []
    blocks_done = 0
    last_o_nonfinite = 0

    for block in range(kmax // MAC_PARALLELISM):
        x_terms, w_terms = _block_operands(x, w, block, fmt)
        alpha, beta = _block_scales(x, w, block)
        for m in range(M):
            base = m * N
            for n in range(N):
                h = 0 if alpha is None else alpha[m] + beta[n]
                products: list[tuple[int, int]] = []
                for i in range(MAC_PARALLELISM):
                    a, ea = x_terms[i][m]
                    b, eb = w_terms[i][n]
                    if a and b:
                        products.append((a * b, ea + eb))
                index = base + n
                s_b, l1_b = exact_sums(products, h, fmt)
                S[index] = S[index] + s_b
                L1[index] = L1[index] + l1_b
                for width in widths:
                    state, flags = hw_add(states[width][index], s_b, width)
                    states[width][index] = state
                    bucket = counters[width]
                    for key in ACC_EVENT_KEYS:
                        bucket[key] += flags[key]

        blocks_done += 1
        K_completed = blocks_done * MAC_PARALLELISM
        if K_completed not in kvalues_set:
            continue
        record = _base_record(K_completed, S, L1, grid)
        last_o_nonfinite = record["o_nonfinite_count"]
        base_records.append(record)
        for width in widths:
            state_list = states[width]
            c_state = [state.to_dict() for state in state_list]
            c_exact = [
                None if state.is_overflow_marker else decode_state(state).to_dict()
                for state in state_list
            ]
            y_bits = [output_bits(state) for state in state_list]
            y_nonfinite = _nonfinite_count(y_bits)
            acc_records.append(
                {
                    "record_type": "widthselect_acc_snapshot",
                    "K": K_completed,
                    "B": K_completed // MAC_PARALLELISM,
                    "w_acc": width,
                    "C_state": c_state,
                    "C_exact": c_exact,
                    "Y_bits": y_bits,
                    "y_nonfinite_count": y_nonfinite,
                    "events_cumulative": dict(counters[width]),
                    "status": _status(y_nonfinite, grid),
                }
            )
    return PairAccResult(
        base_snapshots=base_records,
        acc_snapshots=acc_records,
        status=_status(last_o_nonfinite, grid),
        blocks_completed=blocks_done,
    )


@dataclass
class PairJointResult:
    base_snapshots: list[dict]
    joint_snapshots: list[dict]
    status: str
    blocks_completed: int


def evaluate_pair_joint(
    x: TargetTensor,
    w: TargetTensor,
    k_values: list[int],
    joint_pairs: list[dict],
) -> PairJointResult:
    """Optional joint verification: ``C_{b+1} = HWAdd(C_b, q_b(W_align))``."""
    if x.format != w.format:
        raise ValueError("X and W must use the same target format")
    fmt = x.format
    combos = sorted({(int(pair["w_align"]), int(pair["w_acc"])) for pair in joint_pairs})
    if not combos:
        raise ValueError("at least one joint (W_align, W_acc) combination is required")
    for w_align, w_acc in combos:
        validate_w_align(w_align)
        validate_w_acc(w_acc)
    snapshots = validate_pair(x, w, k_values)
    kmax = max(snapshots)
    M, N = x.rows, w.cols
    grid = M * N
    kvalues_set = set(snapshots)

    S = [Dyadic.zero() for _ in range(grid)]
    L1 = [Dyadic.zero() for _ in range(grid)]
    states = {combo: [zero_state(combo[1]) for _ in range(grid)] for combo in combos}
    counters = {combo: {key: 0 for key in JOINT_EVENT_KEYS} for combo in combos}
    base_records: list[dict] = []
    joint_records: list[dict] = []
    blocks_done = 0
    last_o_nonfinite = 0

    for block in range(kmax // MAC_PARALLELISM):
        x_terms, w_terms = _block_operands(x, w, block, fmt)
        alpha, beta = _block_scales(x, w, block)
        for m in range(M):
            base = m * N
            for n in range(N):
                h = 0 if alpha is None else alpha[m] + beta[n]
                products: list[tuple[int, int]] = []
                for i in range(MAC_PARALLELISM):
                    a, ea = x_terms[i][m]
                    b, eb = w_terms[i][n]
                    if a and b:
                        products.append((a * b, ea + eb))
                index = base + n
                s_b, l1_b = exact_sums(products, h, fmt)
                S[index] = S[index] + s_b
                L1[index] = L1[index] + l1_b
                cmax = max((c for _, c in products), default=None)
                for combo in combos:
                    w_align, w_acc = combo
                    bucket = counters[combo]
                    if cmax is None:
                        q_b = Dyadic.zero()
                    else:
                        _, changed, _, q_b = _align_sum(products, cmax, h, w_align, fmt)
                        if changed:
                            bucket["align_products_changed"] += changed
                            if q_b != s_b:
                                bucket["align_dot16_qb_diff"] += 1
                    state, flags = hw_add(states[combo][index], q_b, w_acc)
                    states[combo][index] = state
                    for key in ACC_EVENT_KEYS:
                        bucket[key] += flags[key]

        blocks_done += 1
        K_completed = blocks_done * MAC_PARALLELISM
        if K_completed not in kvalues_set:
            continue
        record = _base_record(K_completed, S, L1, grid)
        last_o_nonfinite = record["o_nonfinite_count"]
        base_records.append(record)
        for combo in combos:
            w_align, w_acc = combo
            state_list = states[combo]
            c_state = [state.to_dict() for state in state_list]
            c_exact = [
                None if state.is_overflow_marker else decode_state(state).to_dict()
                for state in state_list
            ]
            y_bits = [output_bits(state) for state in state_list]
            y_nonfinite = _nonfinite_count(y_bits)
            joint_records.append(
                {
                    "record_type": "widthselect_joint_snapshot",
                    "K": K_completed,
                    "B": K_completed // MAC_PARALLELISM,
                    "w_align": w_align,
                    "w_acc": w_acc,
                    "C_state": c_state,
                    "C_exact": c_exact,
                    "Y_bits": y_bits,
                    "y_nonfinite_count": y_nonfinite,
                    "events_cumulative": dict(counters[combo]),
                    "status": _status(y_nonfinite, grid),
                }
            )
    return PairJointResult(
        base_snapshots=base_records,
        joint_snapshots=joint_records,
        status=_status(last_o_nonfinite, grid),
        blocks_completed=blocks_done,
    )
