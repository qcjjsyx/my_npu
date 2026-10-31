"""Independent exact-arithmetic oracle built on :class:`fractions.Fraction`.

Deliberately shares no code with the implementation under test.
"""

from __future__ import annotations

from fractions import Fraction


def round_fraction_to_int(x: Fraction) -> int:
    """Round half to even, using floor + fractional part (works for negatives)."""
    floor = x.numerator // x.denominator
    frac = x - floor
    half = Fraction(1, 2)
    if frac > half:
        return floor + 1
    if frac < half:
        return floor
    return floor if floor % 2 == 0 else floor + 1


def _floor_log2(a: Fraction) -> int:
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


def oracle_fp32_bits(x: Fraction) -> int:
    """Correctly rounded binary32 bits for an exact rational value."""
    if x == 0:
        return 0
    sign = 1 if x < 0 else 0
    a = abs(x)
    e = _floor_log2(a)
    if e >= -126:
        ulp = Fraction(2) ** (e - 23)
    else:
        ulp = Fraction(1, 2**149)
    n = round_fraction_to_int(a / ulp)
    if n == 0:
        return sign << 31
    if n >= 2**24:
        n //= 2
        e += 1
    if n >= 2**23:
        if e < -126:
            return (sign << 31) | (1 << 23)
        biased = e + 127
        if biased >= 255:
            return (sign << 31) | 0x7F800000
        return (sign << 31) | (biased << 23) | (n - 2**23)
    if ulp != Fraction(1, 2**149):
        raise AssertionError("subnormal oracle result with normal ulp")
    return (sign << 31) | n


def oracle_align_to_window(value: Fraction, window_lsb: int) -> int:
    """Signed RNE alignment of an exact value to multiples of 2**window_lsb."""
    return round_fraction_to_int(value / Fraction(2) ** window_lsb)


def fp8_product_fraction(sign_a: int, sig_a: int, sign_b: int, sig_b: int, c: int) -> Fraction:
    sign = -1 if (sign_a ^ sign_b) else 1
    return sign * Fraction(sig_a * sig_b) * Fraction(2) ** (c - 20)


def tf32_product_fraction(sign_a: int, sig_a: int, sign_b: int, sig_b: int, c: int) -> Fraction:
    sign = -1 if (sign_a ^ sign_b) else 1
    return sign * Fraction(sig_a * sig_b) * Fraction(2) ** (c - 274)


def oracle_tf32_code(x: Fraction) -> tuple[int, bool]:
    """Independent correctly-rounded TF32 code for an exact rational.

    Layout: sign(1) | raw_exp(8, bias 127) | frac(10); normal value
    ``(1024+frac) * 2**(raw_exp-137)``; subnormal ``frac * 2**-136``; specials
    use ``raw_exp = 255``.  Returns ``(code, overflowed)`` with RNE ties-to-even
    on the exact value.
    """
    if x == 0:
        return 0, False
    sign = 1 if x < 0 else 0
    a = abs(x)
    min_normal = Fraction(2) ** -126
    if a < min_normal:
        n = round_fraction_to_int(a / Fraction(2) ** -136)
        if n <= 1023:
            return (sign << 18) | n, False
        return (sign << 18) | (1 << 10), False
    e = _floor_log2(a)
    while True:
        ulp = Fraction(2) ** (e - 10)
        n = round_fraction_to_int(a / ulp)
        if n < 2048:
            break
        e += 1
    biased = e + 127
    if biased >= 255:
        return (sign << 18) | (0xFF << 10), True
    if n < 1024:
        raise AssertionError("normal oracle result below the minimum normal significand")
    return (sign << 18) | (biased << 10) | (n - 1024), False


def oracle_align_int(
    products: list[tuple[int, int]],
    h: int,
    w_align: int,
    k_shift: int,
    window_offset: int,
) -> tuple[int, Fraction]:
    """Independent ``(N_b, q_b as Fraction)`` of one aligned dot16."""
    if not products:
        return 0, Fraction(0)
    cmax = max(c for _, c in products)
    total = 0
    for p, c in products:
        shift = (cmax - c) + k_shift - w_align
        if shift <= 0:
            total += p << (-shift)
        else:
            total += round_fraction_to_int(Fraction(p) / Fraction(2) ** shift)
    return total, Fraction(total) * Fraction(2) ** (cmax - window_offset - w_align + h)


def oracle_s_b(products: list[tuple[int, int]], h: int, product_lsb_offset: int) -> Fraction:
    total = Fraction(0)
    for p, c in products:
        total += Fraction(p) * Fraction(2) ** (c - product_lsb_offset + h)
    return total
