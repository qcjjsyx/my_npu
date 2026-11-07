"""ULP distance between finite floating-point encodings.

The distance is defined on the ordered sequence of representable finite
values of a format: identical values have distance 0, adjacent values have
distance 1, and crossing an exponent boundary still counts representable
steps.  +0 and -0 are merged (both map to index 0).  The computation uses
exact Python integers over the raw bit patterns and never subtracts unsigned
encodings directly or divides an error by an ULP.
"""

from __future__ import annotations


def ordered_index(bits: int, sign_bit: int) -> int:
    """Monotonic integer index of a finite encoding (zeros merged at 0)."""
    sign = (bits >> sign_bit) & 1
    magnitude = bits & ((1 << sign_bit) - 1)
    if magnitude == 0:
        return 0
    return magnitude if sign == 0 else -magnitude


def fp32_ordered_index(bits: int) -> int:
    if bits >> 23 & 0xFF == 0xFF:
        raise ValueError("ULP distance is defined for finite values only")
    return ordered_index(bits, 31)


def e4m3_ordered_index(bits: int) -> int:
    """Monotonic index of a finite E4M3 encoding (S.1111.111 is NaN)."""
    if bits & 0x7F == 0x7F:
        raise ValueError("ULP distance is defined for finite values only")
    return ordered_index(bits, 7)


def bf16_ordered_index(bits: int) -> int:
    if bits >> 7 & 0xFF == 0xFF:
        raise ValueError("ULP distance is defined for finite values only")
    return ordered_index(bits, 15)


def ulp_distance(bits_a: int, bits_b: int, index_fn) -> int:
    """Exact integer step distance between two finite encodings."""
    return abs(index_fn(bits_a) - index_fn(bits_b))
