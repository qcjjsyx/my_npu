"""TF32 format adapter.

TF32 is 1 sign bit, 8 exponent bits (bias 127), 10 fraction bits; the normal
significand integer is ``1024 + fraction`` (1024..2047).

Internal compact encoding (19 bits), used everywhere in this project:

    bit 18      : sign
    bits 17..10 : raw exponent (8 bits)
    bits 9..0   : fraction (10 bits)

This is *not* an FP32 bit pattern.  ``raw_exp == 0`` decodes to zero (input
FTZ); ``raw_exp == 255`` is a special encoding (NaN/Inf) and is excluded from
sampling.  ``from_fp32_bits`` is provided for test construction only and
truncates the low 13 FP32 fraction bits without rounding.
"""

from __future__ import annotations

from ..arithmetic.dyadic import Dyadic
from .base import DecodedOperand, Format


class Tf32(Format):
    name = "tf32"
    code_bits = 19
    exp_bits = 8
    frac_bits = 10
    bias = 127
    window_offset = 254
    product_lsb_offset = 274
    sampling_exp_min = 96
    sampling_exp_max = 158
    gaussian_exp_center = 127
    gaussian_exp_sigma = 4.0
    value_exp_shift = -137

    def decode(self, code: int) -> DecodedOperand:
        if not 0 <= code <= 0x7FFFF:
            raise ValueError(f"TF32 compact code out of range: {code}")
        sign = (code >> 18) & 1
        raw_exp = (code >> 10) & 0xFF
        frac = code & 0x3FF
        return DecodedOperand(
            code=code,
            sign=sign,
            raw_exp=raw_exp,
            frac=frac,
            sig_int=0 if raw_exp == 0 else 1024 + frac,
            is_zero=raw_exp == 0,
            is_special=raw_exp == 0xFF,
        )

    def encode(self, sign: int, raw_exp: int, frac: int) -> int:
        if sign not in (0, 1) or not 0 <= raw_exp <= 0xFF or not 0 <= frac <= 0x3FF:
            raise ValueError(f"invalid TF32 field values: {sign}, {raw_exp}, {frac}")
        return (sign << 18) | (raw_exp << 10) | frac

    def is_legal_input(self, code: int) -> bool:
        operand = self.decode(code)
        return not operand.is_special and not operand.is_zero

    def legal_fractions(self, raw_exp: int) -> tuple[int, ...]:
        return tuple(range(1024))

    def positive_normal_codes(self, exp_min: int | None = None, exp_max: int | None = None) -> list[tuple[int, Dyadic]]:
        lo = self.sampling_exp_min if exp_min is None else exp_min
        hi = self.sampling_exp_max if exp_max is None else exp_max
        codes = []
        for raw_exp in range(lo, hi + 1):
            for frac in self.legal_fractions(raw_exp):
                code = self.encode(0, raw_exp, frac)
                codes.append((code, Dyadic(1024 + frac, raw_exp - 137)))
        codes.sort(key=lambda item: item[1])
        return codes

    @staticmethod
    def from_fp32_bits(bits: int) -> int:
        sign = (bits >> 31) & 1
        raw_exp = (bits >> 23) & 0xFF
        frac = (bits >> 13) & 0x3FF
        return (sign << 18) | (raw_exp << 10) | frac
