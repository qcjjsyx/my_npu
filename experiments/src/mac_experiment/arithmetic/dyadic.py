"""Exact dyadic rationals.

A :class:`Dyadic` represents ``m * 2**e`` with an arbitrary-precision integer
``m``.  Values are kept in canonical form (``m`` odd or zero, ``e`` zero when
``m`` is zero) so structural equality is exact numeric equality.  No floating
point arithmetic is used anywhere in this module except for the final,
correctly rounded ``to_float`` conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Dyadic:
    m: int
    e: int

    def __post_init__(self) -> None:
        m = self.m
        e = self.e
        if m == 0:
            object.__setattr__(self, "e", 0)
            return
        tz = (m & -m).bit_length() - 1
        if tz:
            object.__setattr__(self, "m", m >> tz)
            object.__setattr__(self, "e", e + tz)

    @staticmethod
    def zero() -> "Dyadic":
        return Dyadic(0, 0)

    def is_zero(self) -> bool:
        return self.m == 0

    def sign(self) -> int:
        return 0 if self.m == 0 else (1 if self.m > 0 else -1)

    def __neg__(self) -> "Dyadic":
        return Dyadic(-self.m, self.e)

    def __abs__(self) -> "Dyadic":
        return Dyadic(abs(self.m), self.e)

    def __add__(self, other: "Dyadic") -> "Dyadic":
        if self.m == 0:
            return other
        if other.m == 0:
            return self
        e = min(self.e, other.e)
        m = (self.m << (self.e - e)) + (other.m << (other.e - e))
        return Dyadic(m, e)

    def __sub__(self, other: "Dyadic") -> "Dyadic":
        return self + (-other)

    def __mul__(self, other: "Dyadic") -> "Dyadic":
        return Dyadic(self.m * other.m, self.e + other.e)

    def __lt__(self, other: "Dyadic") -> bool:
        e = min(self.e, other.e)
        return (self.m << (self.e - e)) < (other.m << (other.e - e))

    def __le__(self, other: "Dyadic") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Dyadic") -> bool:
        return other < self

    def __ge__(self, other: "Dyadic") -> bool:
        return other <= self

    def __hash__(self) -> int:
        return hash((self.m, self.e))

    def to_fraction(self) -> Fraction:
        if self.e >= 0:
            return Fraction(self.m << self.e)
        return Fraction(self.m, 1 << (-self.e))

    def to_float(self) -> float:
        """Correctly rounded binary64 value (may overflow to +/-inf)."""
        if self.m == 0:
            return 0.0
        try:
            return float(self.to_fraction())
        except OverflowError:
            return math.inf if self.m > 0 else -math.inf

    def to_dict(self) -> dict:
        return {"m": str(self.m), "e": self.e}

    @staticmethod
    def from_dict(d: dict) -> "Dyadic":
        return Dyadic(int(d["m"]), int(d["e"]))

    def __repr__(self) -> str:
        return f"Dyadic({self.m}, {self.e})"
