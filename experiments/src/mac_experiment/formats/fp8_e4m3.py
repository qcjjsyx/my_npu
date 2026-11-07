"""FP8 E4M3 format adapter.

Encoding: 1 sign bit, 4 exponent bits (bias 7), 3 fraction bits.
Normal significand integer is ``8 + fraction`` (8..15).  ``raw_exp == 0``
decodes to zero (input FTZ); ``raw_exp == 15, frac == 7`` is NaN, all other
``raw_exp == 15`` codes are finite (max magnitude 448).
"""

from __future__ import annotations

from ..arithmetic.dyadic import Dyadic
from .base import DecodedOperand, Format

NAN_FRAC = 7
NAN_EXP = 15


class Fp8E4M3(Format):
    name = "fp8_e4m3"
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
            raise ValueError(f"FP8 code out of range: {code}")
        sign = (code >> 7) & 1
        raw_exp = (code >> 3) & 0xF
        frac = code & 0x7
        is_nan = raw_exp == NAN_EXP and frac == NAN_FRAC
        return DecodedOperand(
            code=code,
            sign=sign,
            raw_exp=raw_exp,
            frac=frac,
            sig_int=0 if raw_exp == 0 else 8 + frac,
            is_zero=raw_exp == 0,
            is_special=is_nan,
        )

    def encode(self, sign: int, raw_exp: int, frac: int) -> int:
        if sign not in (0, 1) or not 0 <= raw_exp <= 0xF or not 0 <= frac <= 0x7:
            raise ValueError(f"invalid FP8 field values: {sign}, {raw_exp}, {frac}")
        return (sign << 7) | (raw_exp << 3) | frac

    def is_legal_input(self, code: int) -> bool:
        operand = self.decode(code)
        if operand.is_special or operand.is_zero:
            return False
        return operand.frac in self.legal_fractions(operand.raw_exp)

    def legal_fractions(self, raw_exp: int) -> tuple[int, ...]:
        if raw_exp == NAN_EXP:
            return tuple(range(0, NAN_FRAC))
        return tuple(range(8))

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
