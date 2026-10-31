"""Output MXFP8 quantisation with an explicit, exact scale policy.

Fixed experiment policy: for each output group (32 consecutive elements along
N) the scale is the smallest binary power covering the group maximum magnitude,

    A = max_j |z_j|;  choose the smallest integer e with A <= 448 * 2**e,

clamped to the finite E8M0 exponent range (clamping is recorded).  Element
codes are E4M3 RNE with subnormals kept and saturation to the largest finite
magnitude of the same sign.  Every max, threshold comparison, scale choice and
rounding step is computed from exact dyadics -- ``log2(float(...))`` is never
used to decide a boundary.

An all-zero group uses the canonical scale code for exponent 0 (``0x7F``, i.e.
1.0) and keeps signed zeros when the input carries a sign (FP32 ``-0``).
Non-finite input elements mark the group invalid with a recorded reason; they
are never silently ignored and never enter finite error metrics.

Shared-scale quantisation (auxiliary comparison) must use the supplied scale
code and never reselects a scale; elements outside the finite range saturate
and the event is counted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

from ..arithmetic.dyadic import Dyadic
from ..formats.e4m3_element import MAX_FINITE_MAGNITUDE, round_dyadic_to_e4m3
from ..formats.e8m0 import (
    MAX_EXPONENT,
    MIN_EXPONENT,
    code_to_exponent,
    exponent_to_code,
    is_finite_code,
)

GROUP_LENGTH = 32
CANONICAL_ZERO_EXPONENT = 0

EVENT_KEYS = (
    "fully_zero_group",
    "scale_clamped_low",
    "scale_clamped_high",
    "element_overflow",
    "element_saturated",
    "element_zeroed",
    "nonfinite_group",
    "nonfinite_element",
    "reserved_scale_code",
)

MAX_SCALE_MAGNITUDE = MAX_FINITE_MAGNITUDE


def ceil_log2_fraction(q: Fraction) -> int:
    """Smallest integer ``e`` with ``2**e >= q`` for ``q > 0`` (exact)."""
    if q <= 0:
        raise ValueError("ceil_log2_fraction requires a positive value")
    e = q.numerator.bit_length() - q.denominator.bit_length()
    while Fraction(2) ** e < q:
        e += 1
    while Fraction(2) ** (e - 1) >= q:
        e -= 1
    return e


def select_scale_exponent(max_abs: Dyadic) -> tuple[int, bool, bool]:
    """Smallest ``e`` with ``max_abs <= 448 * 2**e``, clamped to E8M0 range."""
    if max_abs.is_zero():
        return CANONICAL_ZERO_EXPONENT, False, False
    q = max_abs.to_fraction() / MAX_SCALE_MAGNITUDE.to_fraction()
    e_raw = ceil_log2_fraction(q)
    if e_raw < MIN_EXPONENT:
        return MIN_EXPONENT, True, False
    if e_raw > MAX_EXPONENT:
        return MAX_EXPONENT, False, True
    return e_raw, False, False


@dataclass
class GroupQuantization:
    group_index: int
    scale_code: int | None
    scale_exponent: int | None
    element_codes: list[int] | None
    fully_zero: bool = False
    clamp_low: bool = False
    clamp_high: bool = False
    n_overflowed: int = 0
    n_saturated: int = 0
    n_zeroed: int = 0
    invalid: bool = False
    invalid_reason: str | None = None


@dataclass
class MatrixQuantization:
    rows: int
    cols: int
    group_length: int
    group_axis: int
    groups: list[GroupQuantization]
    events: Counter = field(default_factory=Counter)

    @property
    def group_grid(self) -> tuple[int, int]:
        if self.group_axis == 0:
            return (self.rows // self.group_length, self.cols)
        return (self.rows, self.cols // self.group_length)

    @property
    def scale_codes(self) -> list[int | None]:
        return [group.scale_code for group in self.groups]

    def group_index(self, row: int, col: int) -> int:
        grid_cols = self.group_grid[1]
        if self.group_axis == 0:
            return (row // self.group_length) * grid_cols + col
        return row * grid_cols + (col // self.group_length)

    def element_code(self, row: int, col: int) -> int | None:
        group = self.groups[self.group_index(row, col)]
        if group.element_codes is None:
            return None
        offset = col % self.group_length if self.group_axis == 1 else row % self.group_length
        return group.element_codes[offset]

    def decoded_value(self, row: int, col: int) -> Dyadic | None:
        from ..formats.e4m3_element import decode_element

        code = self.element_code(row, col)
        if code is None:
            return None
        element = decode_element(code)
        if element.is_zero():
            return element
        group = self.groups[self.group_index(row, col)]
        return Dyadic(element.m, element.e + group.scale_exponent)


def group_index_lists(rows: int, cols: int, group_length: int, group_axis: int) -> list[list[int]]:
    """Row-major element indices of every logical group, in group order."""
    if group_axis == 0:
        if rows % group_length:
            raise ValueError("grouped dimension is not a multiple of the group length")
        groups = []
        for g in range(rows // group_length):
            for c in range(cols):
                groups.append([(g * group_length + i) * cols + c for i in range(group_length)])
        return groups
    if group_axis == 1:
        if cols % group_length:
            raise ValueError("grouped dimension is not a multiple of the group length")
        groups = []
        for r in range(rows):
            for g in range(cols // group_length):
                start = r * cols + g * group_length
                groups.append(list(range(start, start + group_length)))
        return groups
    raise ValueError("group_axis must be 0 or 1")


def _quantize_finite_group(
    values: Sequence[Dyadic],
    zero_signs: Sequence[int] | None,
    exponent: int,
    group_index: int,
    clamp_low: bool,
    clamp_high: bool,
) -> GroupQuantization:
    codes: list[int] = []
    n_overflowed = 0
    n_saturated = 0
    n_zeroed = 0
    fully_zero = True
    for offset, value in enumerate(values):
        zero_sign = 0 if zero_signs is None else int(zero_signs[offset])
        if not value.is_zero():
            fully_zero = False
        scaled = value if value.is_zero() else Dyadic(value.m, value.e - exponent)
        result = round_dyadic_to_e4m3(scaled, zero_sign=zero_sign)
        codes.append(result.code)
        n_overflowed += int(result.overflowed)
        n_saturated += int(result.saturated)
        n_zeroed += int(result.zeroed)
    return GroupQuantization(
        group_index=group_index,
        scale_code=exponent_to_code(exponent),
        scale_exponent=exponent,
        element_codes=codes,
        fully_zero=fully_zero,
        clamp_low=clamp_low,
        clamp_high=clamp_high,
        n_overflowed=n_overflowed,
        n_saturated=n_saturated,
        n_zeroed=n_zeroed,
    )


def _invalid_group(group_index: int, reason: str, n_elements: int) -> GroupQuantization:
    return GroupQuantization(
        group_index=group_index,
        scale_code=None,
        scale_exponent=None,
        element_codes=None,
        invalid=True,
        invalid_reason=reason,
    )


def quantize_matrix_independent(
    values: Sequence[Dyadic | None],
    zero_signs: Sequence[int] | None,
    rows: int,
    cols: int,
    group_length: int = GROUP_LENGTH,
    group_axis: int = 1,
) -> MatrixQuantization:
    groups = group_index_lists(rows, cols, group_length, group_axis)
    if len(values) != rows * cols:
        raise ValueError(f"value count {len(values)} != {rows * cols}")
    if zero_signs is not None and len(zero_signs) != rows * cols:
        raise ValueError(f"zero_sign count {len(zero_signs)} != {rows * cols}")
    out_groups: list[GroupQuantization] = []
    events: Counter = Counter()
    for group_index, indices in enumerate(groups):
        group_values = [values[i] for i in indices]
        group_signs = None if zero_signs is None else [zero_signs[i] for i in indices]
        if any(value is None for value in group_values):
            out_groups.append(_invalid_group(group_index, "nonfinite_value", len(indices)))
            events["nonfinite_group"] += 1
            events["nonfinite_element"] += sum(1 for value in group_values if value is None)
            continue
        max_abs = Dyadic.zero()
        for value in group_values:
            magnitude = abs(value)
            if max_abs < magnitude:
                max_abs = magnitude
        exponent, clamp_low, clamp_high = select_scale_exponent(max_abs)
        group = _quantize_finite_group(
            group_values, group_signs, exponent, group_index, clamp_low, clamp_high
        )
        _accumulate_group_events(events, group)
        out_groups.append(group)
    return MatrixQuantization(rows, cols, group_length, group_axis, out_groups, events)


def quantize_matrix_shared(
    values: Sequence[Dyadic | None],
    zero_signs: Sequence[int] | None,
    rows: int,
    cols: int,
    scale_codes: Sequence[int | None],
    group_length: int = GROUP_LENGTH,
    group_axis: int = 1,
) -> MatrixQuantization:
    groups = group_index_lists(rows, cols, group_length, group_axis)
    if len(values) != rows * cols:
        raise ValueError(f"value count {len(values)} != {rows * cols}")
    if len(scale_codes) != len(groups):
        raise ValueError(f"scale code count {len(scale_codes)} != number of groups {len(groups)}")
    out_groups: list[GroupQuantization] = []
    events: Counter = Counter()
    for group_index, indices in enumerate(groups):
        group_values = [values[i] for i in indices]
        if any(value is None for value in group_values):
            out_groups.append(_invalid_group(group_index, "nonfinite_value", len(indices)))
            events["nonfinite_group"] += 1
            events["nonfinite_element"] += sum(1 for value in group_values if value is None)
            continue
        code = scale_codes[group_index]
        if code is None or not is_finite_code(code):
            out_groups.append(_invalid_group(group_index, "reserved_scale_code", len(indices)))
            events["reserved_scale_code"] += 1
            continue
        exponent = code_to_exponent(code)
        group_signs = None if zero_signs is None else [zero_signs[i] for i in indices]
        group = _quantize_finite_group(
            group_values, group_signs, exponent, group_index, False, False
        )
        _accumulate_group_events(events, group)
        out_groups.append(group)
    return MatrixQuantization(rows, cols, group_length, group_axis, out_groups, events)


def _accumulate_group_events(events: Counter, group: GroupQuantization) -> None:
    if group.fully_zero:
        events["fully_zero_group"] += 1
    if group.clamp_low:
        events["scale_clamped_low"] += 1
    if group.clamp_high:
        events["scale_clamped_high"] += 1
    events["element_overflow"] += group.n_overflowed
    events["element_saturated"] += group.n_saturated
    events["element_zeroed"] += group.n_zeroed


def decoded_group_values(group: GroupQuantization) -> list[Dyadic] | None:
    """Exact decoded values of a finite quantisation group (``None`` if invalid)."""
    from ..formats.e4m3_element import decode_element

    if group.element_codes is None or group.scale_exponent is None:
        return None
    values = []
    for code in group.element_codes:
        element = decode_element(code)
        if element.is_zero():
            values.append(element)
        else:
            values.append(Dyadic(element.m, element.e + group.scale_exponent))
    return values


def metadata() -> dict:
    return {
        "policy": "minimal binary power covering the exact group maximum magnitude",
        "group_length": GROUP_LENGTH,
        "group_axis_output": "N",
        "element_format": "fp8_e4m3",
        "element_rounding": "RNE ties-to-even; subnormals kept; saturation to +/-448; no infinity",
        "scale_format": "e8m0",
        "scale_range": [MIN_EXPONENT, MAX_EXPONENT],
        "all_zero_group_scale": "exponent 0 (code 0x7F, value 1.0)",
        "zero_sign": "exact reference zero canonicalises to +0; FP32 signed zeros are preserved",
        "exactness": "no log2(float(...)) boundary decisions; all comparisons on exact dyadics",
        "shared_scale": "auxiliary comparison uses the supplied scale code and never reselects",
    }
