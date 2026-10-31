"""E4M3 element codec for MXFP8 tensors (OCP MX / OFP8 semantics).

The legacy :class:`Fp8E4M3` *input* adapter applies FTZ to ``raw_exp == 0``
(``mac-exp-*`` dot-product semantics).  MX element coding is different and is
implemented here explicitly:

* ``raw_exp == 0, frac == 0`` is signed zero (``0x00`` / ``0x80``);
* ``raw_exp == 0, frac > 0`` is a subnormal with value ``frac * 2**-9``;
* ``raw_exp == 15, frac == 7`` is NaN (``0x7F`` / ``0xFF``), all other
  ``raw_exp == 15`` codes are finite (largest magnitude 448);
* no infinities exist in E4M3.

Encoding is round-to-nearest-ties-to-even, keeps subnormals (no FTZ) and
saturates to the largest finite magnitude of the same sign when the rounded
magnitude would exceed 448.  Format authority: OCP 8-bit Floating Point
Specification (OFP8) and OCP Microscaling Formats (MX) Specification v1.0;
see ``docs/mx_semantics.md`` for the exact citations and the separation between
format rules and this experiment's quantization policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import rne_shift
from .base import DecodedOperand, Format

NAN_FRAC = 7
NAN_EXP = 15
MAX_FINITE_CODE = 0x7E
MAX_FINITE_MAGNITUDE = Dyadic(14, 5)  # 448 = 14 * 2**5
MIN_NORMAL_EXPONENT = -6  # 2**-6 = 8 * 2**-9
SUBNORMAL_LSB_EXPONENT = -9
MANTISSA_BITS = 4  # normal significand 8..15
NAN_CODES = (0x7F, 0xFF)


def is_nan_code(code: int) -> bool:
    return code in NAN_CODES


class E4M3Element(Format):
    """OCP E4M3 element adapter used by MXFP8 matrix experiments."""

    name = "fp8_e4m3_mx_element"
    code_bits = 8
    exp_bits = 4
    frac_bits = 3
    bias = 7
    window_offset = 14
    product_lsb_offset = 20
    sampling_exp_min = 1
    sampling_exp_max = 15
    gaussian_exp_center = 7
    gaussian_exp_sigma = 1.0
    value_exp_shift = -10

    def decode(self, code: int) -> DecodedOperand:
        if not 0 <= code <= 0xFF:
            raise ValueError(f"E4M3 code out of range: {code}")
        sign = (code >> 7) & 1
        raw_exp = (code >> 3) & 0xF
        frac = code & 0x7
        is_nan = raw_exp == NAN_EXP and frac == NAN_FRAC
        is_zero = raw_exp == 0 and frac == 0
        sig_int = frac if raw_exp == 0 else 8 + frac
        return DecodedOperand(
            code=code,
            sign=sign,
            raw_exp=raw_exp,
            frac=frac,
            sig_int=sig_int,
            is_zero=is_zero,
            is_special=is_nan,
        )

    def encode(self, sign: int, raw_exp: int, frac: int) -> int:
        if sign not in (0, 1) or not 0 <= raw_exp <= 0xF or not 0 <= frac <= 0x7:
            raise ValueError(f"invalid E4M3 field values: {sign}, {raw_exp}, {frac}")
        return (sign << 7) | (raw_exp << 3) | frac

    def is_legal_input(self, code: int) -> bool:
        operand = self.decode(code)
        return not operand.is_special

    def legal_fractions(self, raw_exp: int) -> tuple[int, ...]:
        if raw_exp == NAN_EXP:
            return tuple(range(0, NAN_FRAC))
        return tuple(range(8))

    def operand_value(self, operand: DecodedOperand) -> Dyadic:
        if operand.is_special:
            raise ValueError("E4M3 NaN codes (0x7F/0xFF) have no numeric value")
        if operand.is_zero:
            return Dyadic.zero()
        if operand.raw_exp == 0:
            magnitude = Dyadic(operand.frac, SUBNORMAL_LSB_EXPONENT)
        else:
            magnitude = Dyadic(8 + operand.frac, operand.raw_exp - 10)
        return -magnitude if operand.sign else magnitude

    def positive_normal_codes(self, exp_min: int | None = None, exp_max: int | None = None) -> list[tuple[int, Dyadic]]:
        lo = self.sampling_exp_min if exp_min is None else exp_min
        hi = self.sampling_exp_max if exp_max is None else exp_max
        codes = []
        for raw_exp in range(lo, hi + 1):
            for frac in self.legal_fractions(raw_exp):
                code = self.encode(0, raw_exp, frac)
                codes.append((code, Dyadic(8 + frac, raw_exp - 10)))
        codes.sort(key=lambda item: item[1])
        return codes

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update(
            {
                "kind": "mx_element",
                "input_ftz": "false (subnormals = frac * 2**-9 are decoded exactly)",
                "signed_zero": "0x00 / 0x80 are +0 / -0",
                "nan_codes": ["0x7F", "0xFF"],
                "has_infinity": False,
                "max_finite_magnitude": "448 (0x7E / 0xFE)",
                "encode_policy": "RNE ties-to-even, subnormals kept, saturating (no infinity)",
            }
        )
        return meta


def decode_element(code: int) -> Dyadic:
    """Exact value of a finite E4M3 element code; raises on NaN codes."""
    return E4M3Element().operand_value(E4M3Element().decode(code))


@dataclass(frozen=True)
class E4M3RoundResult:
    code: int
    saturated: bool
    overflowed: bool
    zeroed: bool


def round_dyadic_to_e4m3(x: Dyadic, zero_sign: int = 0) -> E4M3RoundResult:
    """Correctly round an exact dyadic to E4M3 with saturation to 448.

    ``zero_sign`` selects the sign of a zero result (exact zero input or a
    nonzero value that underflows to zero).  ``overflowed`` marks an input
    magnitude strictly above 448; ``saturated`` marks a result clamped to the
    largest finite magnitude of the same sign.
    """
    if zero_sign not in (0, 1):
        raise ValueError("zero_sign must be 0 or 1")
    if x.m == 0:
        return E4M3RoundResult(code=zero_sign << 7, saturated=False, overflowed=False, zeroed=False)
    negative = x.m < 0
    a = abs(x.m)
    overflowed = abs(x) > MAX_FINITE_MAGNITUDE
    nbits = a.bit_length()
    qexp = max(x.e + nbits - MANTISSA_BITS, SUBNORMAL_LSB_EXPONENT)
    t = rne_shift(a, qexp - x.e)
    if t == 0:
        return E4M3RoundResult(code=(negative << 7), saturated=False, overflowed=overflowed, zeroed=True)
    if t >= (1 << MANTISSA_BITS):
        t >>= 1
        qexp += 1
    if t >= (1 << (MANTISSA_BITS - 1)):
        biased = qexp + 10
        if biased >= 16 or (biased == 15 and t > 14):
            return E4M3RoundResult(
                code=(negative << 7) | MAX_FINITE_CODE, saturated=True, overflowed=overflowed, zeroed=False
            )
        if biased < 1:
            raise AssertionError("normal E4M3 result with subnormal exponent")
        code = (negative << 7) | (biased << 3) | (t - (1 << (MANTISSA_BITS - 1)))
    else:
        if qexp != SUBNORMAL_LSB_EXPONENT:
            raise AssertionError("subnormal E4M3 result must be quantised at 2**-9")
        code = (negative << 7) | t
    return E4M3RoundResult(code=code, saturated=False, overflowed=overflowed, zeroed=False)
