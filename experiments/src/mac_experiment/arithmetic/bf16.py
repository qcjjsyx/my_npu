"""BF16 (bfloat16) values and exact RNE conversion from dyadics.

BF16 is 1 sign bit, 8 exponent bits (bias 127) and 7 fraction bits; the normal
significand integer is ``128 + fraction``.  Conversions here are computed
directly from exact dyadics with integer arithmetic: subnormals are kept, no
FTZ, overflow rounds to +/-Inf (no saturation).  Host float casts are never
used to define rounding semantics; ``to_float`` exists for reporting only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .dyadic import Dyadic
from .rounding import rne_shift

SIGN_MASK = 0x8000
EXP_MASK = 0x7F80
FRAC_MASK = 0x007F
QUIET_NAN = 0x7FC0
INF = 0x7F80
MANTISSA_BITS = 8
MIN_SUBNORMAL_EXP = -133


@dataclass(frozen=True)
class Bf16:
    bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", self.bits & 0xFFFF)

    @staticmethod
    def from_bits(bits: int) -> "Bf16":
        return Bf16(bits)

    @staticmethod
    def zero(sign: int = 0) -> "Bf16":
        return Bf16(SIGN_MASK if sign else 0)

    @staticmethod
    def inf(sign: int = 0) -> "Bf16":
        return Bf16(INF | (SIGN_MASK if sign else 0))

    @staticmethod
    def nan() -> "Bf16":
        return Bf16(QUIET_NAN)

    @staticmethod
    def from_fp32_bits(bits: int) -> "Bf16":
        """RNE conversion of an FP32 bit pattern (via its exact value).

        For finite inputs this is the same as the hardware-style
        round-to-nearest-even of the 32-bit pattern, including ties-to-even,
        subnormals and overflow to Inf.  It is implemented through the exact
        dyadic value to avoid relying on host float behaviour.
        """
        from .fp32 import Fp32

        value = Fp32.from_bits(bits)
        if value.is_nan():
            return Bf16.nan()
        if value.is_inf():
            return Bf16.inf(value.sign)
        if value.is_zero():
            return Bf16.zero(value.sign)
        return round_dyadic_to_bf16(value.exact_dyadic())[0]

    @property
    def sign(self) -> int:
        return 1 if self.bits & SIGN_MASK else 0

    @property
    def exponent_field(self) -> int:
        return (self.bits & EXP_MASK) >> 7

    @property
    def fraction_field(self) -> int:
        return self.bits & FRAC_MASK

    def is_nan(self) -> bool:
        return self.exponent_field == 0xFF and self.fraction_field != 0

    def is_inf(self) -> bool:
        return self.exponent_field == 0xFF and self.fraction_field == 0

    def is_finite(self) -> bool:
        return self.exponent_field != 0xFF

    def is_zero(self) -> bool:
        return (self.bits & ~SIGN_MASK) == 0

    def to_float(self) -> float:
        """Binary64 rendering for reporting/plotting only."""
        if not self.is_finite():
            return math.nan if self.is_nan() else math.copysign(math.inf, -1.0 if self.sign else 1.0)
        if self.is_zero():
            return -0.0 if self.sign else 0.0
        if self.exponent_field == 0:
            magnitude = math.ldexp(float(self.fraction_field), MIN_SUBNORMAL_EXP)
        else:
            magnitude = math.ldexp(float(128 + self.fraction_field), self.exponent_field - 134)
        return -magnitude if self.sign else magnitude

    def exact_dyadic(self) -> Dyadic:
        if not self.is_finite():
            raise ValueError("cannot convert non-finite BF16 to Dyadic")
        if self.is_zero():
            return Dyadic.zero()
        if self.exponent_field == 0:
            magnitude = Dyadic(self.fraction_field, MIN_SUBNORMAL_EXP)
        else:
            magnitude = Dyadic(128 + self.fraction_field, self.exponent_field - 134)
        return -magnitude if self.sign else magnitude

    def same_value(self, other: "Bf16") -> bool:
        return self.exact_dyadic() == other.exact_dyadic()

    def to_dict(self) -> dict:
        return {"bits": self.bits, "hex": f"0x{self.bits:04X}"}

    @staticmethod
    def from_dict(d: dict) -> "Bf16":
        return Bf16(int(d["bits"]))

    def __repr__(self) -> str:
        return f"Bf16(0x{self.bits:04X})"


def round_dyadic_to_bf16(x: Dyadic) -> tuple[Bf16, Dyadic | None]:
    """Correctly round an exact dyadic to BF16 (RNE, subnormals, Inf on overflow).

    Returns ``(bf16, exact_rounded_value)``; the exact value is ``None`` when
    the result is infinite.  Exact zero maps to +0; a nonzero value that
    rounds to zero keeps its IEEE sign.
    """
    if x.m == 0:
        return Bf16.zero(0), Dyadic.zero()
    negative = x.m < 0
    a = abs(x.m)
    nbits = a.bit_length()
    qexp = max(x.e + nbits - MANTISSA_BITS, MIN_SUBNORMAL_EXP)
    t = rne_shift(a, qexp - x.e)
    if t == 0:
        return Bf16.zero(1 if negative else 0), Dyadic.zero()
    if t >= (1 << MANTISSA_BITS):
        t >>= 1
        qexp += 1
    if t >= (1 << (MANTISSA_BITS - 1)):
        biased = qexp + 134
        if biased >= 0xFF:
            return Bf16.inf(1 if negative else 0), None
        bits = (SIGN_MASK if negative else 0) | (biased << 7) | (t - (1 << (MANTISSA_BITS - 1)))
    else:
        if qexp != MIN_SUBNORMAL_EXP:
            raise AssertionError("subnormal BF16 result must be quantised at 2**-133")
        bits = (SIGN_MASK if negative else 0) | t
    value = Dyadic(-t if negative else t, qexp)
    return Bf16.from_bits(bits), value
