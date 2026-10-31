"""W_acc custom accumulator: state format and ``HWAdd`` core (confirmed design).

The numeric format is the one confirmed by the user from
``docs/mac_width_selection.md`` section 4.1 (F1-F7):

* state significand ``M`` is a ``W_acc + 1`` bit **two's complement** signed
  integer, sign included, no implicit bit and no separate sign field;
* the 8-bit exponent code has bias **126** and ``E_code = 255`` is reserved as
  a sticky overflow/invalid marker; finite codes are ``0..254``;
* ``decode(M, E) = M * 2**(E - 126 - W_acc)``; the state logical width is
  exactly ``W_acc + 9`` bits and no extra precision is carried across updates;
* each HWAdd presents two ``W_acc``-bit signed operands, adds them exactly into
  a ``W_acc + 1`` bit result, then normalizes (left shift, exponent down) until
  the top two two's-complement bits differ, stopping at the exponent floor;
* operand alignment uses one common window ``w* = max(fit(state), fit(input))``
  clamped to the exponent floor; low bits are rounded with RNE ties-to-even on
  the signed value (sign-symmetric), and GRS is only a combinational view of
  the same rounding inside one HWAdd;
* overflow (a normalized exponent above the code range) sets ``E_code = 255``
  and is sticky; the final conversion maps the marker to ``+/-Inf`` by the sign
  of ``M`` and the event is counted.

Canonical zero is ``M = 0, E_code = 0``; the state has no signed zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import rne_shift
from ..version import WIDTHSELECT_ACC_FORMAT_STATUS, WIDTHSELECT_ACC_POLICY

OPERAND_BITS_RULE = "operands are W_acc signed bits including the sign"
RESULT_BITS_RULE = "full adder result is W_acc + 1 signed bits including the sign"
EXPONENT_BITS = 8
STATE_BITS_RULE = "state logical width = W_acc + 9 bits"
W_ACC_MIN = 8
W_ACC_MAX = 60

BIAS = 126
RESERVED_EXPONENT_CODE = 0xFF

ACC_POLICY = WIDTHSELECT_ACC_POLICY
ACC_EVENT_KEYS = (
    "acc_updates",
    "acc_input_zero",
    "acc_input_absorbed",
    "acc_state_bits_dropped",
    "acc_input_bits_dropped",
    "acc_carry",
    "acc_cancellation",
    "acc_overflow",
    "acc_subnormal",
    "acc_marker_propagated",
)


class DesignPendingError(RuntimeError):
    """Retained name for external callers; the format is confirmed and implemented."""


def validate_w_acc(w_acc: int) -> None:
    if not isinstance(w_acc, int) or w_acc < W_ACC_MIN or w_acc > W_ACC_MAX:
        raise ValueError(f"W_acc must be an integer in [{W_ACC_MIN}, {W_ACC_MAX}], got {w_acc!r}")


def state_bits(w_acc: int) -> int:
    validate_w_acc(w_acc)
    return w_acc + 9


def operands_bits(w_acc: int) -> int:
    validate_w_acc(w_acc)
    return w_acc


def result_bits(w_acc: int) -> int:
    validate_w_acc(w_acc)
    return w_acc + 1


def significand_bounds(w_acc: int) -> tuple[int, int]:
    validate_w_acc(w_acc)
    return -(1 << w_acc), (1 << w_acc) - 1


def operand_bounds(w_acc: int) -> tuple[int, int]:
    validate_w_acc(w_acc)
    return -(1 << (w_acc - 1)), (1 << (w_acc - 1)) - 1


def window_floor(w_acc: int) -> int:
    """Smallest allowed window LSB exponent (freezes at ``E_code = 0``)."""
    validate_w_acc(w_acc)
    return -126 - w_acc


@dataclass(frozen=True)
class HwAccState:
    E_code: int
    M: int
    w_acc: int

    def __post_init__(self) -> None:
        validate_w_acc(self.w_acc)
        if not 0 <= self.E_code <= RESERVED_EXPONENT_CODE:
            raise ValueError(f"E_code out of range: {self.E_code}")
        low, high = significand_bounds(self.w_acc)
        if not low <= self.M <= high:
            raise ValueError(
                f"state significand {self.M} outside the {self.w_acc + 1} bit two's complement "
                f"range [{low}, {high}]"
            )
        if self.M == 0 and self.E_code not in (0,):
            raise ValueError("the state has one canonical zero: M = 0 with E_code = 0")
        if self.E_code == RESERVED_EXPONENT_CODE and self.M == 0:
            raise ValueError("an overflow marker must keep a nonzero sign-carrying significand")

    @property
    def is_overflow_marker(self) -> bool:
        return self.E_code == RESERVED_EXPONENT_CODE

    @property
    def sign(self) -> int:
        return 1 if self.M < 0 else 0

    def to_dict(self) -> dict:
        return {"E_code": int(self.E_code), "M": str(int(self.M)), "w_acc": int(self.w_acc)}

    @staticmethod
    def from_dict(payload: dict) -> "HwAccState":
        return HwAccState(int(payload["E_code"]), int(payload["M"]), int(payload["w_acc"]))


def zero_state(w_acc: int) -> HwAccState:
    validate_w_acc(w_acc)
    return HwAccState(0, 0, w_acc)


def overflow_marker(w_acc: int, M: int) -> HwAccState:
    validate_w_acc(w_acc)
    sign = -1 if M < 0 else 1
    magnitude = abs(M)
    low, high = significand_bounds(w_acc)
    if not low <= M <= high or M == 0:
        raise ValueError("overflow marker keeps a finite-range nonzero significand")
    return HwAccState(RESERVED_EXPONENT_CODE, sign * magnitude, w_acc)


def decode_state(state: HwAccState) -> Dyadic:
    """Exact decoded dyadic of a finite state (overflow markers have none)."""
    if state.is_overflow_marker:
        raise ValueError("the overflow marker has no finite decoded value")
    return Dyadic(state.M, state.E_code - BIAS - state.w_acc)


def output_bits(state: HwAccState) -> int:
    """Final FP32 conversion; the overflow marker maps to +/-Inf."""
    from ..arithmetic.rounding import round_dyadic_to_fp32

    sign_bit = 0x8000_0000 if state.M < 0 else 0
    if state.is_overflow_marker:
        return 0x7F80_0000 | sign_bit
    bits, _ = round_dyadic_to_fp32(decode_state(state))
    return bits.bits


def _round_shift(m: int, d: int) -> int:
    """RNE ``m * 2**d`` for any integer ``d`` (ties-to-even on the value)."""
    if m == 0:
        return 0
    if d >= 0:
        return m << d
    return rne_shift(m, -d)


def fit_operand(m: int, e: int, w_acc: int) -> tuple[int, int]:
    """``(n, w)``: ``n = RNE(m * 2**(e-w))`` in the W_acc-bit window, maximal ``w``.

    ``w`` is the smallest window LSB exponent (finest grid) for which the
    rounded operand fits the signed ``W_acc``-bit range; ties-to-even on the
    value, negative semantics are sign-symmetric.  The two's-complement window
    has one extra representable value at the negative end (``-2**(w_acc-1)``),
    so after finding a fitting window the next finer one is checked before
    giving up; missing it would silently drop up to one extra bit.
    """
    if m == 0:
        raise ValueError("fit_operand requires a nonzero value")
    low, high = operand_bounds(w_acc)
    d = w_acc - 1 - abs(m).bit_length()
    n = _round_shift(m, d)
    while not low <= n <= high:
        d -= 1
        n = _round_shift(m, d)
    finer = _round_shift(m, d + 1)
    if low <= finer <= high:
        d += 1
        n = finer
    return n, e - d


@dataclass
class HwAddTrace:
    state_before: dict
    w_star: int | None
    n_state: int
    n_input: int
    dropped_state: bool
    dropped_input: bool
    input_absorbed: bool
    integer_sum: int
    normalization_shift: int
    e_after: int
    E_after: int
    overflow: bool
    marker_propagated: bool

    def to_dict(self) -> dict:
        return {
            "state_before": self.state_before,
            "w_star": self.w_star,
            "n_state": self.n_state,
            "n_input": self.n_input,
            "dropped_state": self.dropped_state,
            "dropped_input": self.dropped_input,
            "input_absorbed": self.input_absorbed,
            "integer_sum": self.integer_sum,
            "normalization_shift": self.normalization_shift,
            "e_after": self.e_after,
            "E_after": self.E_after,
            "overflow": self.overflow,
            "marker_propagated": self.marker_propagated,
        }


def hw_add(
    state: HwAccState | None,
    value: Dyadic,
    w_acc: int,
    trace: list | None = None,
) -> tuple[HwAccState, dict]:
    """``C' = HWAdd_{W_acc}(C, value)`` with an exact dyadic incoming value.

    Returns ``(new_state, event_flags)``; when ``trace`` is a list the full
    per-update diagnostic record is appended (used by the replay tool).
    """
    validate_w_acc(w_acc)
    if state is None:
        state = zero_state(w_acc)
    if state.w_acc != w_acc:
        raise ValueError("state width does not match the requested W_acc")
    flags = {key: 0 for key in ACC_EVENT_KEYS}
    flags["acc_updates"] = 1
    if state.is_overflow_marker:
        flags["acc_marker_propagated"] = 1
        if trace is not None:
            trace.append(
                HwAddTrace(
                    state_before=state.to_dict(),
                    w_star=None,
                    n_state=0,
                    n_input=0,
                    dropped_state=False,
                    dropped_input=False,
                    input_absorbed=False,
                    integer_sum=0,
                    normalization_shift=0,
                    e_after=state.E_code,
                    E_after=state.E_code,
                    overflow=False,
                    marker_propagated=True,
                ).to_dict()
            )
        return state, flags
    state_exact = decode_state(state)
    m_s, e_s = (state_exact.m, state_exact.e) if not state_exact.is_zero() else (0, 0)
    m_i, e_i = (value.m, value.e) if not value.is_zero() else (0, 0)
    if value.is_zero():
        flags["acc_input_zero"] = 1
    if m_s == 0 and m_i == 0:
        # canonical zero plus zero: nothing to narrow, zero stays zero
        if trace is not None:
            trace.append(
                HwAddTrace(
                    state_before=state.to_dict(),
                    w_star=None,
                    n_state=0,
                    n_input=0,
                    dropped_state=False,
                    dropped_input=False,
                    input_absorbed=False,
                    integer_sum=0,
                    normalization_shift=0,
                    e_after=0,
                    E_after=0,
                    overflow=False,
                    marker_propagated=False,
                ).to_dict()
            )
        return zero_state(w_acc), flags
    w_floor = window_floor(w_acc)

    if m_s == 0:
        _, w_star = fit_operand(m_i, e_i, w_acc)
    elif m_i == 0:
        _, w_star = fit_operand(m_s, e_s, w_acc)
    else:
        _, w_s = fit_operand(m_s, e_s, w_acc)
        _, w_i = fit_operand(m_i, e_i, w_acc)
        w_star = max(w_s, w_i)
    w_star = max(w_star, w_floor)

    if m_s == 0:
        n_state = 0
    else:
        n_state = _round_shift(m_s, e_s - w_star)
    n_input = _round_shift(m_i, e_i - w_star) if m_i != 0 else 0
    dropped_state = m_s != 0 and Dyadic(n_state, w_star) != Dyadic(m_s, e_s)
    dropped_input = m_i != 0 and Dyadic(n_input, w_star) != Dyadic(m_i, e_i)
    if dropped_state:
        flags["acc_state_bits_dropped"] = 1
    if dropped_input:
        flags["acc_input_bits_dropped"] = 1
    input_absorbed = m_i != 0 and n_input == 0
    if input_absorbed:
        flags["acc_input_absorbed"] = 1

    low, high = operand_bounds(w_acc)
    if not (low <= n_state <= high and low <= n_input <= high):
        raise ArithmeticError("aligned operand outside the W_acc-bit window")

    integer_sum = n_state + n_input
    if not (-(1 << w_acc) <= integer_sum <= (1 << w_acc) - 1):
        raise ArithmeticError("W_acc+1 bit adder result out of two's complement range")
    _, operand_high = operand_bounds(w_acc)
    if integer_sum > operand_high or integer_sum < -(1 << (w_acc - 1)):
        flags["acc_carry"] = 1

    if integer_sum == 0:
        if m_s != 0 and m_i != 0:
            flags["acc_cancellation"] = 1
        if trace is not None:
            trace.append(
                HwAddTrace(
                    state_before=state.to_dict(),
                    w_star=w_star,
                    n_state=n_state,
                    n_input=n_input,
                    dropped_state=dropped_state,
                    dropped_input=dropped_input,
                    input_absorbed=input_absorbed,
                    integer_sum=0,
                    normalization_shift=0,
                    e_after=0,
                    E_after=0,
                    overflow=False,
                    marker_propagated=False,
                ).to_dict()
            )
        return zero_state(w_acc), flags

    e = w_star
    normalization_shift = 0
    while True:
        top = (integer_sum >> (w_acc - 1)) & 0b11
        if top not in (0, 0b11):
            break
        if e - 1 < w_floor:
            break
        integer_sum <<= 1
        e -= 1
        normalization_shift += 1
        if integer_sum < -(1 << w_acc) or integer_sum > (1 << w_acc) - 1:
            raise AssertionError("normalization overflowed the state significand field")
    E_code = e + BIAS + w_acc
    if E_code > 254:
        flags["acc_overflow"] = 1
        new_state = overflow_marker(w_acc, integer_sum)
        if trace is not None:
            trace.append(
                HwAddTrace(
                    state_before=state.to_dict(),
                    w_star=w_star,
                    n_state=n_state,
                    n_input=n_input,
                    dropped_state=dropped_state,
                    dropped_input=dropped_input,
                    input_absorbed=input_absorbed,
                    integer_sum=integer_sum,
                    normalization_shift=normalization_shift,
                    e_after=e,
                    E_after=E_code,
                    overflow=True,
                    marker_propagated=False,
                ).to_dict()
            )
        return new_state, flags
    if E_code == 0 and ((integer_sum >> (w_acc - 1)) & 0b11) in (0, 0b11):
        flags["acc_subnormal"] = 1
    new_state = HwAccState(E_code, integer_sum, w_acc)
    if trace is not None:
        trace.append(
            HwAddTrace(
                state_before=state.to_dict(),
                w_star=w_star,
                n_state=n_state,
                n_input=n_input,
                dropped_state=dropped_state,
                dropped_input=dropped_input,
                input_absorbed=input_absorbed,
                integer_sum=integer_sum,
                normalization_shift=normalization_shift,
                e_after=e,
                E_after=E_code,
                overflow=False,
                marker_propagated=False,
            ).to_dict()
        )
    return new_state, flags


def metadata() -> dict:
    return {
        "status": WIDTHSELECT_ACC_FORMAT_STATUS,
        "policy": ACC_POLICY,
        "operand_bits_rule": OPERAND_BITS_RULE,
        "result_bits_rule": RESULT_BITS_RULE,
        "exponent_bits": EXPONENT_BITS,
        "state_bits_rule": STATE_BITS_RULE,
        "bias": BIAS,
        "reserved_exponent_code": RESERVED_EXPONENT_CODE,
        "decode": "value(M, E) = M * 2**(E - 126 - W_acc); M two's complement W_acc+1 bits",
        "zero": "canonical M = 0, E_code = 0; no signed zero in the state",
        "subnormal": "E_code = 0 with a non-normalized significand (gradual underflow floor)",
        "overflow": "E_code = 255 sticky marker; final conversion -> +/-Inf by sign(M); counted",
        "alignment": "w* = max(fit(state), fit(input)) clamped to the exponent floor; RNE ties-to-even",
        "state_to_operand": (
            "the W_acc+1 bit state significand enters the W_acc bit window through the same "
            "alignment; its dropped LSB is the per-update loss; there is no zero-input bypass, "
            "so every dot16 update narrows the state (an exact zero partial sum still narrows)"
        ),
        "normalization": "lossless left shift until the top two two's-complement bits differ; exponent decreases; floor at E_code = 0",
        "grs": "combinational view of the same RNE rounding inside one HWAdd; never persisted",
        "proposal_reference": "docs/mac_width_selection.md section 4.1 (F1-F7, confirmed)",
    }
