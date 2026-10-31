"""Rounding primitives: signed RNE right shift and exact Dyadic -> FP32.

The only rounding step in the whole pipeline is the final conversion of an
exact dyadic value to binary32 (round-to-nearest, ties-to-even, gradual
underflow, overflow to infinity).  Partial sums are never pre-rounded.
"""

from __future__ import annotations

from .dyadic import Dyadic
from .fp32 import Fp32

MIN_SUBNORMAL_EXP = -149
MANTISSA_BITS = 24
SIGN_BIT = 0x8000_0000


def rne_shift(m: int, s: int) -> int:
    """Round ``m / 2**s`` to the nearest integer, ties to even.

    ``s`` may be negative, in which case the result is exact (left shift).
    """
    if m == 0:
        return 0
    sign = -1 if m < 0 else 1
    a = abs(m)
    if s <= 0:
        return sign * (a << (-s))
    q, r = divmod(a, 1 << s)
    half = 1 << (s - 1)
    if r > half or (r == half and (q & 1)):
        q += 1
    return sign * q


def round_dyadic_to_fp32(x: Dyadic) -> tuple[Fp32, Dyadic | None]:
    """Correctly round an exact dyadic to FP32.

    Returns ``(fp32, exact_rounded_value)``.  ``exact_rounded_value`` is the
    exact dyadic value of the produced FP32 number, or ``None`` when the
    result is infinite.  Exact zero maps to +0; a nonzero value that rounds to
    zero keeps its IEEE sign.
    """
    if x.m == 0:
        return Fp32.zero(0), Dyadic.zero()
    negative = x.m < 0
    a = abs(x.m)
    nbits = a.bit_length()
    qexp = max(x.e + nbits - MANTISSA_BITS, MIN_SUBNORMAL_EXP)
    t = rne_shift(a, qexp - x.e)
    if t == 0:
        return Fp32.zero(1 if negative else 0), Dyadic.zero()
    if t >= (1 << MANTISSA_BITS):
        t >>= 1
        qexp += 1
    if t >= (1 << (MANTISSA_BITS - 1)):
        biased = qexp + 150
        if biased >= 0xFF:
            return Fp32.inf(1 if negative else 0), None
        bits = (SIGN_BIT if negative else 0) | (biased << 23) | (t - (1 << (MANTISSA_BITS - 1)))
    else:
        if qexp != MIN_SUBNORMAL_EXP:
            raise AssertionError("subnormal result must be quantised at 2**-149")
        bits = (SIGN_BIT if negative else 0) | t
    value = Dyadic(-t if negative else t, qexp)
    return Fp32.from_bits(bits), value
