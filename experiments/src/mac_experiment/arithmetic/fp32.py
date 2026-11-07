"""IEEE-754 binary32 values stored as bit patterns.

The accumulator state is an FP32 value.  Storing the raw 32-bit pattern keeps
the framework free of host binary64 arithmetic: values are only converted to
Python floats for reporting/plotting, where every FP32 value is exactly
representable in binary64.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .dyadic import Dyadic

SIGN_MASK = 0x8000_0000
EXP_MASK = 0x7F80_0000
FRAC_MASK = 0x007F_FFFF
QUIET_NAN = 0x7FC0_0000
INF = 0x7F80_0000


@dataclass(frozen=True)
class Fp32:
    bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", self.bits & 0xFFFF_FFFF)

    @staticmethod
    def from_bits(bits: int) -> "Fp32":
        return Fp32(bits)

    @staticmethod
    def zero(sign: int = 0) -> "Fp32":
        return Fp32(SIGN_MASK if sign else 0)

    @staticmethod
    def inf(sign: int = 0) -> "Fp32":
        return Fp32(INF | (SIGN_MASK if sign else 0))

    @staticmethod
    def nan() -> "Fp32":
        return Fp32(QUIET_NAN)

    @staticmethod
    def from_float(value: float) -> "Fp32":
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return Fp32(bits)

    @property
    def sign(self) -> int:
        return 1 if self.bits & SIGN_MASK else 0

    @property
    def exponent_field(self) -> int:
        return (self.bits & EXP_MASK) >> 23

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
        return struct.unpack("<f", struct.pack("<I", self.bits))[0]

    def exact_dyadic(self) -> Dyadic:
        """Exact value of a finite FP32 number (raises for inf/NaN)."""
        if not self.is_finite():
            raise ValueError("cannot convert non-finite FP32 to Dyadic")
        if self.is_zero():
            return Dyadic.zero()
        exp = self.exponent_field
        frac = self.fraction_field
        if exp == 0:
            magnitude = Dyadic(frac, -149)
        else:
            magnitude = Dyadic((1 << 23) | frac, exp - 150)
        return -magnitude if self.sign else magnitude

    def same_value(self, other: "Fp32") -> bool:
        """Numeric equality; +0 and -0 compare equal."""
        return self.exact_dyadic() == other.exact_dyadic()

    def to_dict(self) -> dict:
        return {"bits": self.bits, "hex": f"0x{self.bits:08X}"}

    @staticmethod
    def from_dict(d: dict) -> "Fp32":
        return Fp32(int(d["bits"]))

    def __repr__(self) -> str:
        return f"Fp32(0x{self.bits:08X})"
