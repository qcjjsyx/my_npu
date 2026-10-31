"""Transaction-level, bit-exact integer oracle for IP_Design.md §§1–9.

This module deliberately does not use RTL-style G/R/S, barrel shifts, CSA,
signed fixed-width arithmetic, or Python float. It computes exact integers and
binary exponents. ALIGN uses per-term RNE; ACC preserves alignment tails
and rounds the completed sum once. ACC writeback uses FTZ/Inf; MX input
exponent boundaries are deferred. It models no cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction

DOT_N = 16
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class ContractViolation(ValueError):
    """Input or intermediate value outside the currently defined contract."""


class UnfrozenContract(ContractViolation):
    """A numerical policy is unfrozen or deferred; no hardware fallback implied."""


class SourceFormat(IntEnum):
    FP16 = 0
    BF16 = 1
    E4M3 = 2
    E5M2 = 3
    INT16 = 4
    INT8 = 5
    INT4 = 6


class ComputeMode(IntEnum):
    FP19 = 0
    FP9 = 1
    INT16 = 2
    INT8 = 3
    INT4 = 4


class AccKind(IntEnum):
    INT32 = 0
    TC_FLOAT = 1


FLOAT8 = frozenset((SourceFormat.E4M3, SourceFormat.E5M2))
BIAS = {
    SourceFormat.FP16: 15,
    SourceFormat.BF16: 127,
    SourceFormat.E4M3: 7,
    SourceFormat.E5M2: 15,
}


def signed(bits: int, width: int) -> int:
    bits &= (1 << width) - 1
    return bits - (1 << width) if bits & (1 << (width - 1)) else bits


def require_signed(value: int, width: int, name: str) -> int:
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise ContractViolation(f"{name}={value} does not fit signed{width}")
    return value


def require_unsigned(value: int, width: int, name: str) -> int:
    if not 0 <= value < (1 << width):
        raise ContractViolation(f"{name}={value} does not fit unsigned{width}")
    return value


def mode_for(fmt_a: SourceFormat, fmt_b: SourceFormat) -> ComputeMode:
    if fmt_a != fmt_b and not (fmt_a in FLOAT8 and fmt_b in FLOAT8):
        raise ContractViolation("unsupported A/B source-format pair")
    if fmt_a in (SourceFormat.FP16, SourceFormat.BF16):
        return ComputeMode.FP19
    if fmt_a in FLOAT8:
        return ComputeMode.FP9
    return {
        SourceFormat.INT16: ComputeMode.INT16,
        SourceFormat.INT8: ComputeMode.INT8,
        SourceFormat.INT4: ComputeMode.INT4,
    }[fmt_a]


def round_div_pow2(value: int, shift: int) -> tuple[int, bool]:
    """RNE(value / 2**shift), using floor quotient and nonnegative remainder."""
    if shift < 0:
        return value << -shift, False
    if shift == 0:
        return value, False
    divisor = 1 << shift
    quotient, remainder = divmod(value, divisor)
    twice = remainder << 1
    increment = twice > divisor or (twice == divisor and (quotient & 1) != 0)
    return quotient + int(increment), remainder != 0


def saturate_int32(value: int) -> int:
    require_signed(value, 36, "integer work sum")
    return min(INT32_MAX, max(INT32_MIN, value))


@dataclass(frozen=True)
class Transaction:
    raw_a: tuple[int, ...]
    raw_b: tuple[int, ...]
    fmt_a: SourceFormat
    fmt_b: SourceFormat
    group_valid: int
    acc_first: int
    tag: tuple[int, int]
    mx_enable: bool = False
    scale_a: tuple[int, int] = (0, 0)
    scale_b: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class Entry:
    mode: ComputeMode
    kind: AccKind
    payload_a: tuple[int, ...]
    payload_b: tuple[int, ...]
    bias_sum: int
    group_valid: int


@dataclass(frozen=True)
class MPhase:
    phase: int
    product0: tuple[int, ...]
    product1: tuple[int, ...]
    exp_sum0: tuple[int, ...]
    exp_sum1: tuple[int, ...]
    nz0: tuple[bool, ...]
    nz1: tuple[bool, ...]


@dataclass(frozen=True)
class Aligned:
    lane0: tuple[int, ...]
    lane1: tuple[int, ...]
    emax: tuple[int, int]
    has_nonzero: tuple[bool, bool]
    inexact: tuple[bool, bool]


@dataclass(frozen=True)
class Psum:
    """FP19: signed36 Q7.29; FP9: signed28 Q7.21; integers retain units."""

    packed56: int
    contribution0: int
    contribution1: int


@dataclass(frozen=True)
class Mx:
    contribution: tuple[int, int]
    exp: tuple[int, int]  # Mathematical biased E; MX range handling is deferred.
    kind: AccKind
    phase: int


@dataclass(frozen=True)
class FloatAcc36State:
    """Q2.34 finite state, canonical zero, or logical signed infinity.

    E=255 uses infinity_sign; sig=0 is a placeholder, not a finite coefficient.
    Physical packing of infinity's sign is deferred to the RTL interface design.
    """

    sig: int
    exp: int
    infinity_sign: int = 0

    def __post_init__(self) -> None:
        require_signed(self.sig, 36, "Q2.34 coefficient")
        if self.infinity_sign:
            if self.infinity_sign not in (-1, 1) or self.exp != 255 or self.sig != 0:
                raise ContractViolation("invalid logical infinity")
        elif self.sig == 0:
            if self.exp != 0:
                raise ContractViolation("zero state must be M=0/E=0")
        elif not 1 <= self.exp <= 254:
            raise ContractViolation("finite nonzero state requires normal E")

    @property
    def data36(self) -> int | None:
        # Never present an infinity placeholder as a finite zero snapshot.
        return None if self.infinity_sign else self.sig & ((1 << 36) - 1)


FloatState = FloatAcc36State


def _check_float_exp(exp: int) -> None:
    # MX unnormalized input boundary behavior is deliberately deferred.
    if not 1 <= exp <= 254:
        raise UnfrozenContract(f"E={exp}: MX input exponent boundary handling deferred")


@dataclass(frozen=True)
class FloatAcc36Update:
    old_operand: int
    new_operand: int
    common_exp: int | None
    full_sum: int  # signed37 integer part, before normalization
    old_tail: Fraction
    new_tail: Fraction
    state: FloatAcc36State


def trace_float_acc36_update(
    old: FloatAcc36State | None, contribution: int, exp: int, *, first: bool,
) -> FloatAcc36Update:
    """Q2.34 ACC with exact alignment tails and one final RNE.

    The oracle uses arbitrary precision, not a presumed RTL GRS width.
    floor operands are signed36; their signed37 sum plus both nonnegative
    tails is the exact value at common E. Nonzero output magnitudes normalize
    to [2**34, 2**35); rounding carry adjusts E before boundary validation.
    """
    require_signed(contribution, 36, "floating Q2.34 contribution")
    if contribution:
        _check_float_exp(exp)
    state = FloatAcc36State(0, 0) if first else old
    if state is None:
        raise ContractViolation("TC_FLOAT state requires acc_first")
    if state.infinity_sign:
        return FloatAcc36Update(0, 0, None, 0, Fraction(0), Fraction(0), state)
    nonzero_exps = ([state.exp] if state.sig else []) + ([exp] if contribution else [])
    if not nonzero_exps:
        return FloatAcc36Update(0, 0, None, 0, Fraction(0), Fraction(0), FloatAcc36State(0, 0))
    common = max(nonzero_exps)

    def align_exact(sig, e):
        if not sig:
            return 0, Fraction(0)
        divisor = 1 << (common - e)
        q, r = divmod(sig, divisor)
        return require_signed(q, 36, "aligned operand"), Fraction(r, divisor)

    old_op, old_tail = align_exact(state.sig, state.exp)
    new_op, new_tail = align_exact(contribution, exp)
    total = require_signed(old_op + new_op, 37, "full adder result")
    # Independent exact integer sum on the finest input grid drives rounding.
    fine = min(nonzero_exps)
    old_exact = (state.sig << (state.exp - fine)) if state.sig else 0
    new_exact = (contribution << (exp - fine)) if contribution else 0
    exact = old_exact + new_exact
    if exact == 0:
        result = FloatAcc36State(0, 0)
    else:
        shift = abs(exact).bit_length() - 1 - 34
        sig = round_div_pow2(exact, shift)[0]
        result_exp = fine + shift
        if abs(sig) == 1 << 35:
            sig //= 2  # Exact carry renormalization, not a second rounding.
            result_exp += 1
        if result_exp < 1:
            result = FloatAcc36State(0, 0)  # FTZ, canonical positive zero.
        elif result_exp > 254:
            result = FloatAcc36State(0, 255, -1 if sig < 0 else 1)
        else:
            result = FloatAcc36State(sig, result_exp)
    return FloatAcc36Update(old_op, new_op, common, total, old_tail, new_tail, result)


def update_float_acc36(
    old: FloatAcc36State | None, contribution: int, exp: int, *, first: bool,
) -> FloatAcc36State:
    return trace_float_acc36_update(old, contribution, exp, first=first).state


def update_float_acc36_candidate(*args, **kwargs):
    """Removed signed37/lsb_exp API: never reinterpret old vectors as Q2.34."""
    raise UnfrozenContract("legacy ACC36 candidate retired; use update_float_acc36 with biased E")


@dataclass(frozen=True)
class Snapshot:
    kind: AccKind
    group_valid: int
    data: tuple[int | None, int | None]
    exp: tuple[int | None, int | None]
    tag: tuple[int | None, int | None]
    infinity_sign: tuple[int | None, int | None] = (None, None)


@dataclass(frozen=True)
class PhaseTrace:
    m: MPhase
    aligned: Aligned
    psum: Psum
    mx: Mx


@dataclass(frozen=True)
class TransactionTrace:
    entry: Entry
    phases: tuple[PhaseTrace, ...]
    snapshot: Snapshot


def _decode_finite_float(raw: int, fmt: SourceFormat) -> tuple[int, int]:
    """Return Eeff and signed integer significand after source-format policy."""
    if fmt == SourceFormat.FP16:
        raw &= 0xFFFF
        sign, exp, frac, max_exp = raw >> 15, (raw >> 10) & 31, raw & 1023, 31
        frac_width, pad, max_finite_exp, max_finite_frac = 10, 0, 30, 1023
    elif fmt == SourceFormat.BF16:
        raw &= 0xFFFF
        sign, exp, frac, max_exp = raw >> 15, (raw >> 7) & 255, raw & 127, 255
        frac_width, pad, max_finite_exp, max_finite_frac = 7, 3, 254, 127
    elif fmt == SourceFormat.E4M3:
        raw &= 0xFF
        sign, exp, frac, max_exp = raw >> 7, (raw >> 3) & 15, raw & 7, 15
        frac_width, pad, max_finite_exp, max_finite_frac = 3, 0, 15, 6
    elif fmt == SourceFormat.E5M2:
        raw &= 0xFF
        sign, exp, frac, max_exp = raw >> 7, (raw >> 2) & 31, raw & 3, 31
        frac_width, pad, max_finite_exp, max_finite_frac = 2, 1, 30, 3
    else:
        raise ContractViolation("float decoder called for integer format")

    is_nan = (exp == max_exp and frac == (1 << frac_width) - 1) if fmt == SourceFormat.E4M3 else (exp == max_exp and frac != 0)
    is_inf = fmt != SourceFormat.E4M3 and exp == max_exp and frac == 0
    if is_nan or (exp == 0 and frac == 0):
        return 0, 0  # NaN and either signed zero become canonical +0.
    if is_inf:
        exp, frac = max_finite_exp, max_finite_frac

    hidden = int(exp != 0)
    effective_exp = exp if exp != 0 else 1
    magnitude = (hidden << frac_width | frac) << pad
    significand = -magnitude if sign else magnitude
    require_signed(significand, 12 if fmt in (SourceFormat.FP16, SourceFormat.BF16) else 5, "entry significand")
    return effective_exp, significand


def _pack_float19(raw: int, fmt: SourceFormat) -> int:
    exp, sig = _decode_finite_float(raw, fmt)
    return (exp << 16) | (sig & 0xFFF)


def _pack_float9(raw: int, fmt: SourceFormat) -> int:
    exp0, sig0 = _decode_finite_float(raw & 0xFF, fmt)
    exp1, sig1 = _decode_finite_float(raw >> 8, fmt)
    return (exp1 << 21) | (exp0 << 16) | ((sig1 & 31) << 5) | (sig0 & 31)


def _check_integer_inputs(raw: int, mode: ComputeMode, group_valid: int) -> None:
    width = {ComputeMode.INT16: 16, ComputeMode.INT8: 8, ComputeMode.INT4: 4}[mode]
    groups = (0,) if mode == ComputeMode.INT16 else (0, 1)
    for group in groups:
        if not (group_valid & (1 << group)):
            continue
        value = signed(raw >> (group * width), width)
        if value == -(1 << (width - 1)):
            raise ContractViolation(f"reserved minimum signed{width} input")


def enter(tx: Transaction, *, tag_w: int) -> Entry:
    if tag_w < 1:
        raise ContractViolation("TAG_W must be positive and explicit")
    if len(tx.raw_a) != DOT_N or len(tx.raw_b) != DOT_N:
        raise ContractViolation("each input vector must have 16 elements")
    if len(tx.tag) != 2 or len(tx.scale_a) != 2 or len(tx.scale_b) != 2:
        raise ContractViolation("group metadata must have two entries")
    for raw in tx.raw_a + tx.raw_b:
        require_unsigned(raw, 16, "raw operand")
    try:
        fmt_a, fmt_b = SourceFormat(tx.fmt_a), SourceFormat(tx.fmt_b)
    except ValueError as exc:
        raise ContractViolation("invalid source-format encoding") from exc
    mode = mode_for(fmt_a, fmt_b)
    if tx.mx_enable and mode not in (ComputeMode.FP9, ComputeMode.INT8):
        raise ContractViolation("MX is legal only for FP8 and INT8")
    require_unsigned(tx.group_valid, 2, "group_valid")
    require_unsigned(tx.acc_first, 2, "acc_first")
    if tx.group_valid == 0 or (mode in (ComputeMode.FP19, ComputeMode.INT16) and tx.group_valid != 1):
        raise ContractViolation("invalid group_valid for WIDE/DUAL mode")
    for group in (0, 1):
        require_unsigned(tx.tag[group], tag_w, f"tag[{group}]")
        require_unsigned(tx.scale_a[group], 8, f"scale_a[{group}]")
        require_unsigned(tx.scale_b[group], 8, f"scale_b[{group}]")
        if tx.mx_enable and (tx.group_valid & (1 << group)) and (tx.scale_a[group] == 255 or tx.scale_b[group] == 255):
            raise ContractViolation("E8M0 255 has no frozen valid-transaction policy")

    payload_a: list[int] = []
    payload_b: list[int] = []
    for raw_a, raw_b in zip(tx.raw_a, tx.raw_b):
        if mode == ComputeMode.FP19:
            a, b = _pack_float19(raw_a, fmt_a), _pack_float19(raw_b, fmt_b)
        elif mode == ComputeMode.FP9:
            a, b = _pack_float9(raw_a, fmt_a), _pack_float9(raw_b, fmt_b)
            if not (tx.group_valid & 1):
                a &= ~((31 << 16) | 31)
                b &= ~((31 << 16) | 31)
            if not (tx.group_valid & 2):
                a &= ~((31 << 21) | (31 << 5))
                b &= ~((31 << 21) | (31 << 5))
        else:
            if mode == ComputeMode.INT4 and ((raw_a | raw_b) & 0xFF00):
                raise ContractViolation("INT4 raw container [15:8] must be zero")
            _check_integer_inputs(raw_a, mode, tx.group_valid)
            _check_integer_inputs(raw_b, mode, tx.group_valid)
            mask = {ComputeMode.INT16: 0xFFFF, ComputeMode.INT8: 0xFFFF, ComputeMode.INT4: 0xFF}[mode]
            a, b = raw_a & mask, raw_b & mask
            if mode != ComputeMode.INT16:
                lane_mask = 0xFF if mode == ComputeMode.INT8 else 0xF
                shift = 8 if mode == ComputeMode.INT8 else 4
                if not (tx.group_valid & 1):
                    a &= ~lane_mask
                    b &= ~lane_mask
                if not (tx.group_valid & 2):
                    a &= ~(lane_mask << shift)
                    b &= ~(lane_mask << shift)
        require_unsigned(a, 26, "A payload")
        require_unsigned(b, 26, "B payload")
        payload_a.append(a)
        payload_b.append(b)

    kind = AccKind.TC_FLOAT if mode in (ComputeMode.FP19, ComputeMode.FP9) or tx.mx_enable else AccKind.INT32
    bias_sum = BIAS.get(fmt_a, 0) + BIAS.get(fmt_b, 0)
    return Entry(mode, kind, tuple(payload_a), tuple(payload_b), bias_sum, tx.group_valid)


def multiply(entry: Entry, phase: int = 0) -> MPhase:
    if phase not in (0, 1) or (entry.mode != ComputeMode.INT16 and phase != 0):
        raise ContractViolation("invalid INT16 phase")
    p0, p1, e0, e1, nz0, nz1 = [], [], [], [], [], []
    for a, b in zip(entry.payload_a, entry.payload_b):
        if entry.mode == ComputeMode.FP19:
            sa, sb = signed(a, 12), signed(b, 12)
            product0, product1 = sa * sb, 0
            exp0, exp1 = ((a >> 16) & 255) + ((b >> 16) & 255), 0
            nonzero0, nonzero1 = sa != 0 and sb != 0, False
        elif entry.mode == ComputeMode.FP9:
            sa0, sb0 = signed(a, 5), signed(b, 5)
            sa1, sb1 = signed(a >> 5, 5), signed(b >> 5, 5)
            product0, product1 = sa0 * sb0, sa1 * sb1
            exp0 = ((a >> 16) & 31) + ((b >> 16) & 31)
            exp1 = ((a >> 21) & 31) + ((b >> 21) & 31)
            nonzero0 = bool(entry.group_valid & 1) and sa0 != 0 and sb0 != 0
            nonzero1 = bool(entry.group_valid & 2) and sa1 != 0 and sb1 != 0
        elif entry.mode == ComputeMode.INT16:
            al, bl = a & 255, b & 255
            ah, bh = signed(a >> 8, 8), signed(b >> 8, 8)
            product0 = al * (bl if phase == 0 else bh)
            product1 = ah * (bl if phase == 0 else bh)
            exp0 = exp1 = 0
            nonzero0 = nonzero1 = False
        else:
            width = 8 if entry.mode == ComputeMode.INT8 else 4
            sa0, sb0 = signed(a, width), signed(b, width)
            sa1, sb1 = signed(a >> width, width), signed(b >> width, width)
            product0, product1 = sa0 * sb0, sa1 * sb1
            exp0 = exp1 = 0
            nonzero0 = nonzero1 = False
        require_signed(product0, 24, "MUL12 product")
        require_signed(product1, 18, "MUL9 product")
        require_unsigned(exp0, 9, "M exp_sum0")
        require_unsigned(exp1, 6, "M exp_sum1")
        p0.append(product0)
        p1.append(product1)
        e0.append(exp0)
        e1.append(exp1)
        nz0.append(nonzero0)
        nz1.append(nonzero1)
    return MPhase(phase, tuple(p0), tuple(p1), tuple(e0), tuple(e1), tuple(nz0), tuple(nz1))


def align(entry: Entry, m: MPhase) -> Aligned:
    if entry.mode == ComputeMode.FP19:
        max0 = max((e for e, nz in zip(m.exp_sum0, m.nz0) if nz), default=0)
        max1 = 0
    elif entry.mode == ComputeMode.FP9:
        max0 = max((e for e, nz in zip(m.exp_sum0, m.nz0) if nz), default=0)
        max1 = max((e for e, nz in zip(m.exp_sum1, m.nz1) if nz), default=0)
    else:
        max0 = max1 = 0
    lane0, lane1 = [], []
    inexact = [False, False]
    for i in range(DOT_N):
        if entry.mode == ComputeMode.FP19:
            if m.nz0[i]:
                d = max0 - m.exp_sum0[i]
                x0, lost = (0, m.product0[i] != 0) if d >= 32 else round_div_pow2(m.product0[i] << 9, d)
                inexact[0] |= lost
            else:
                x0 = 0
            x1 = 0
        elif entry.mode == ComputeMode.FP9:
            x = []
            for group, (product, exp, nz, maximum) in enumerate(((m.product0[i], m.exp_sum0[i], m.nz0[i], max0), (m.product1[i], m.exp_sum1[i], m.nz1[i], max1))):
                if nz:
                    d = maximum - exp
                    value, lost = (0, product != 0) if d >= 24 else round_div_pow2(product << 15, d)
                    inexact[group] |= lost
                else:
                    value = 0
                x.append(value)
            x0, x1 = x
        else:
            x0, x1 = m.product0[i], m.product1[i]
        require_signed(x0, 32, "ALIGN lane0")
        require_signed(x1, 24, "ALIGN lane1")
        lane0.append(x0)
        lane1.append(x1)
    return Aligned(tuple(lane0), tuple(lane1), (max0, max1), (any(m.nz0), any(m.nz1)), tuple(inexact))


def partial_sum(entry: Entry, m: MPhase, aligned: Aligned) -> Psum:
    if entry.mode == ComputeMode.FP19:
        for x in aligned.lane0:
            require_signed(x, 32, "FP19 ALIGN term")
        low = sum(x & ((1 << 23) - 1) for x in aligned.lane0)
        high = sum(x >> 23 for x in aligned.lane0)
        require_signed(low, 28, "FP19 low tree")
        require_signed(high, 28, "FP19 high tree")
        result0 = low + (high << 23)
        result1 = 0
    elif entry.mode == ComputeMode.INT16:
        low = sum(aligned.lane0)
        high = sum(aligned.lane1)
        require_signed(low, 28, "INT16 low tree")
        require_signed(high, 28, "INT16 high tree")
        term = low + (high << 8)
        require_signed(term, 28 if m.phase == 0 else 27, "INT16 phase sum")
        result0 = term if m.phase == 0 else term << 8
        result1 = 0
        require_signed(result0, 36, "weighted INT16 contribution")
    else:
        result0 = sum(aligned.lane0)
        result1 = sum(aligned.lane1)
        require_signed(result1, 28, "DUAL group1 tree")
    require_signed(result0, 36 if entry.mode in (ComputeMode.FP19, ComputeMode.INT16) else 28, "PSUM group0")
    if entry.mode in (ComputeMode.FP19, ComputeMode.INT16):
        packed = result0 & ((1 << 56) - 1)
    else:
        packed = ((result1 & ((1 << 28) - 1)) << 28) | (result0 & ((1 << 28) - 1))
    return Psum(packed, result0, result1)


def mx(entry: Entry, tx: Transaction, m: MPhase, aligned: Aligned, psum: Psum) -> Mx:
    mode = entry.mode
    if mode in (ComputeMode.FP19, ComputeMode.INT16):
        if signed(psum.packed56, 56) != psum.contribution0:
            raise ContractViolation("WIDE PSUM sign extension failure")
        c0, c1 = signed(psum.packed56, 37), 0
    else:
        c0 = signed(psum.packed56, 28)
        c1 = signed(psum.packed56 >> 28, 28)
    coefficients = [c0, c1]
    exps = [0, 0]
    for g in (0, 1):
        if not tx.group_valid & (1 << g):
            coefficients[g] = 0
            continue
        scale = 0
        if tx.mx_enable:
            for raw in (tx.scale_a[g], tx.scale_b[g]):
                require_unsigned(raw, 8, "E8M0 raw byte")
                if raw == 255:
                    raise ContractViolation("E8M0 255 policy not frozen")
            scale = tx.scale_a[g] + tx.scale_b[g] - 254
        if mode == ComputeMode.FP19:
            require_signed(coefficients[g], 36, "FP19 Q7.29 PSUM")
            exps[g] = aligned.emax[g] - entry.bias_sum + 132
        elif mode == ComputeMode.FP9:
            coefficients[g] <<= 8
            exps[g] = aligned.emax[g] - entry.bias_sum + 132 + scale
        elif mode == ComputeMode.INT8 and tx.mx_enable:
            require_signed(coefficients[g], 19, "MXINT8 dot16 integer")
            coefficients[g] <<= 17
            exps[g] = scale + 132
        if entry.kind == AccKind.TC_FLOAT:
            require_signed(coefficients[g], 36, "MX Q2.34 coefficient")
    # E remains mathematical here: no masking to 8 bits or invented overflow
    # result. ACC refuses unresolved field encodings before committing state.
    return Mx(tuple(coefficients), tuple(exps), entry.kind, m.phase)


class ReferenceAccumulator:
    """Two architectural states; phase0 INT16 uses one pending 36-bit state."""

    def __init__(self, *, tag_w: int):
        if tag_w < 1:
            raise ContractViolation("TAG_W must be positive and explicit")
        self.tag_w = tag_w
        self.reset()

    def reset(self) -> None:
        self._kind: list[AccKind | None] = [None, None]
        self._integer: list[int] = [0, 0]
        self._float: list[FloatState] = [FloatState(0, 0), FloatState(0, 0)]
        self._int16_mid: int | None = None

    def update_float(self, group: int, contribution: int, exp: int, *, first: bool) -> FloatState:
        if group not in (0, 1):
            raise ContractViolation("invalid ACC group")
        if not first and self._kind[group] != AccKind.TC_FLOAT:
            raise ContractViolation("TC_FLOAT group requires acc_first")
        new_state = update_float_acc36(self._float[group], contribution, exp, first=first)
        self._kind[group] = AccKind.TC_FLOAT
        self._float[group] = new_state
        return new_state

    def update_integer(self, group: int, contribution: int, *, first: bool) -> int:
        if group not in (0, 1):
            raise ContractViolation("invalid ACC group")
        require_signed(contribution, 36, "integer contribution")
        if not first and self._kind[group] != AccKind.INT32:
            raise ContractViolation("INT32 group requires acc_first")
        old = 0 if first else self._integer[group]
        result = saturate_int32(old + contribution)
        self._integer[group] = result
        self._kind[group] = AccKind.INT32
        return result

    def int16_phase0(self, contribution: int, *, first: bool) -> None:
        if self._int16_mid is not None:
            raise ContractViolation("INT16 phase0 already pending")
        require_signed(contribution, 36, "INT16 phase0 contribution")
        if not first and self._kind[0] != AccKind.INT32:
            raise ContractViolation("INT16 ACC0 requires acc_first")
        old = 0 if first else self._integer[0]
        self._int16_mid = require_signed(old + contribution, 36, "INT16 intermediate")

    def int16_phase1(self, contribution: int) -> int:
        if self._int16_mid is None:
            raise ContractViolation("INT16 phase1 without phase0")
        require_signed(contribution, 36, "INT16 phase1 contribution")
        result = saturate_int32(self._int16_mid + contribution)
        self._int16_mid = None
        self._integer[0] = result
        self._kind[0] = AccKind.INT32
        return result

    def _commit_phase(self, tx: Transaction, entry: Entry, result: Mx) -> Snapshot | None:
        if entry.mode == ComputeMode.INT16:
            if result.phase == 0:
                self.int16_phase0(result.contribution[0], first=bool(tx.acc_first & 1))
                return None
            self.int16_phase1(result.contribution[0])
        else:
            if self._int16_mid is not None:
                raise ContractViolation("transaction interleaved into INT16 phases")
            pending_float: dict[int, FloatAcc36State] = {}
            for group in (0, 1):
                if not (tx.group_valid & (1 << group)):
                    continue
                first = bool(tx.acc_first & (1 << group))
                if result.kind == AccKind.TC_FLOAT:
                    if not first and self._kind[group] != AccKind.TC_FLOAT:
                        raise ContractViolation("TC_FLOAT group requires acc_first")
                    pending_float[group] = update_float_acc36(
                        self._float[group], result.contribution[group],
                        result.exp[group], first=first,
                    )
                else:
                    self.update_integer(group, result.contribution[group], first=first)
            # Commit only after every active floating group has validated.
            for group, state in pending_float.items():
                self._float[group] = state
                self._kind[group] = AccKind.TC_FLOAT
        data: list[int | None] = [None, None]
        exp: list[int | None] = [None, None]
        tags: list[int | None] = [None, None]
        infinity_sign: list[int | None] = [None, None]
        for group in (0, 1):
            if not (tx.group_valid & (1 << group)):
                continue
            if result.kind == AccKind.TC_FLOAT:
                data[group] = self._float[group].data36
                exp[group] = self._float[group].exp
                infinity_sign[group] = self._float[group].infinity_sign
            else:
                data[group] = self._integer[group] & 0xFFFFFFFF
                exp[group] = 0
            tags[group] = tx.tag[group]
        return Snapshot(result.kind, tx.group_valid, tuple(data), tuple(exp), tuple(tags), tuple(infinity_sign))

    def step(self, tx: Transaction) -> TransactionTrace:
        entry = enter(tx, tag_w=self.tag_w)
        traces: list[PhaseTrace] = []
        snapshot: Snapshot | None = None
        for phase in ((0, 1) if entry.mode == ComputeMode.INT16 else (0,)):
            m = multiply(entry, phase)
            aligned = align(entry, m)
            psum = partial_sum(entry, m, aligned)
            result = mx(entry, tx, m, aligned, psum)
            snapshot = self._commit_phase(tx, entry, result)
            traces.append(PhaseTrace(m, aligned, psum, result))
        assert snapshot is not None
        return TransactionTrace(entry, tuple(traces), snapshot)
