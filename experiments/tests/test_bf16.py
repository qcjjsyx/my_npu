"""BF16 conversion and ULP distance tests with an independent Fraction oracle."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from mac_experiment.arithmetic.bf16 import Bf16, round_dyadic_to_bf16
from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.arithmetic.rounding import round_dyadic_to_fp32
from mac_experiment.arithmetic.ulp import bf16_ordered_index, fp32_ordered_index, ulp_distance

from oracle import _floor_log2, round_fraction_to_int


def oracle_bf16_bits(x: Fraction) -> int:
    """Independent correctly-rounded BF16 encoding of an exact rational."""
    if x == 0:
        return 0
    sign = 1 if x < 0 else 0
    a = abs(x)
    e = _floor_log2(a)
    if e >= -126:
        ulp = Fraction(2) ** (e - 7)
    else:
        ulp = Fraction(1, 2**133)
    n = round_fraction_to_int(a / ulp)
    if n == 0:
        return sign << 15
    if n >= 2**8:
        n //= 2
        e += 1
    if n >= 2**7:
        if e < -126:
            return (sign << 15) | (1 << 7)
        biased = e + 127
        if biased >= 255:
            return (sign << 15) | 0x7F80
        return (sign << 15) | (biased << 7) | (n - 2**7)
    if ulp != Fraction(1, 2**133):
        raise AssertionError("subnormal BF16 result with normal ulp")
    return (sign << 15) | n


def bits(x: Dyadic) -> int:
    return round_dyadic_to_bf16(x)[0].bits


def test_exact_representable_values():
    assert bits(Dyadic(1, 0)) == 0x3F80
    assert bits(Dyadic(-1, 0)) == 0xBF80
    assert bits(Dyadic(3, 0)) == 0x4040
    assert bits(Dyadic(5, -1)) == 0x4020
    assert bits(Dyadic(0, 0)) == 0x0000


def test_ties_to_even_both_sides():
    assert bits(Dyadic((1 << 8) + 1, -8)) == 0x3F80
    assert bits(Dyadic((1 << 8) + 3, -8)) == 0x3F82
    assert bits(Dyadic(-((1 << 8) + 1), -8)) == 0xBF80
    assert bits(Dyadic(-((1 << 8) + 3), -8)) == 0xBF82


def test_mantissa_and_exponent_carry():
    assert bits(Dyadic((1 << 8) - 1, -8)) == 0x3F7F
    assert bits(Dyadic((1 << 9) - 3, -8)) == 0x3FFE
    assert bits(Dyadic((1 << 9) - 1, 0)) == 0x4400
    assert bits(Dyadic((1 << 24) + 1, -24)) == 0x3F80


def test_subnormal_and_normal_boundary():
    assert bits(Dyadic(1, -133)) == 0x0001
    assert bits(Dyadic(1, -134)) == 0x0000
    assert bits(Dyadic(3, -134)) == 0x0002
    assert bits(Dyadic(3, -133)) == 0x0003
    assert bits(Dyadic((1 << 7) - 1, -133)) == 0x007F
    assert bits(Dyadic(1 << 7, -133)) == 0x0080
    assert bits(Dyadic(1, -126)) == 0x0080


def test_max_finite_and_overflow():
    max_finite = Dyadic((1 << 8) - 1, 120)
    assert bits(max_finite) == 0x7F7F
    assert bits(Dyadic((1 << 9) - 1, 119)) == 0x7F80
    assert bits(Dyadic((1 << 9) - 3, 119)) == 0x7F7E
    assert bits(Dyadic(-((1 << 9) - 1), 119)) == 0xFF80
    assert bits(Dyadic(1 << 8, 120)) == 0x7F80


def test_negative_tiny_rounds_to_negative_zero():
    value, exact = round_dyadic_to_bf16(Dyadic(-1, -200))
    assert value.bits == 0x8000
    assert value.is_zero()
    assert exact == Dyadic.zero()


def test_infinity_and_nan_encodings():
    assert Bf16.inf(0).bits == 0x7F80
    assert Bf16.inf(1).bits == 0xFF80
    assert Bf16.nan().is_nan()
    assert not Bf16.inf().is_finite()
    with pytest.raises(ValueError):
        Bf16.nan().exact_dyadic()


def test_from_fp32_bits_rne():
    assert Bf16.from_fp32_bits(0x3F800000).bits == 0x3F80
    assert Bf16.from_fp32_bits(0x3F808000).bits == 0x3F80
    assert Bf16.from_fp32_bits(0x3F818000).bits == 0x3F82
    assert Bf16.from_fp32_bits(0x7F800000).bits == 0x7F80
    assert Bf16.from_fp32_bits(0xFF800000).bits == 0xFF80
    assert Bf16.from_fp32_bits(0x7FC00000).is_nan()
    assert Bf16.from_fp32_bits(0x00000001).bits == 0x0000


def test_from_fp32_bits_preserves_signed_zero():
    assert Bf16.from_fp32_bits(0x00000000).bits == 0x0000
    assert Bf16.from_fp32_bits(0x80000000).bits == 0x8000
    assert Bf16.from_fp32_bits(0x80000000).is_zero()
    assert Bf16.from_fp32_bits(0x80000000).sign == 1


def test_to_float_roundtrip():
    for raw in (0x3F80, 0xBF80, 0x4040, 0x0001, 0x7F7F, 0x0080):
        value = Bf16.from_bits(raw)
        assert value.exact_dyadic().to_float() == value.to_float()


def test_against_fraction_oracle_random():
    rng = random.Random(20260917)
    for _ in range(500):
        m = rng.randrange(-(1 << rng.randrange(1, 60)), (1 << rng.randrange(1, 60)) + 1)
        e = rng.randrange(-150, 130)
        x = Dyadic(m, e)
        assert bits(x) == oracle_bf16_bits(x.to_fraction()), (m, e)


def test_double_rounding_counterexample():
    s_total = Dyadic(1, 0) + Dyadic(1, -8) + Dyadic(1, -26)
    direct = round_dyadic_to_bf16(s_total)[0]
    o32, o32_exact = round_dyadic_to_fp32(s_total)
    double_rounded = round_dyadic_to_bf16(o32_exact)[0]
    assert direct.bits == 0x3F81
    assert o32.bits == 0x3F808000
    assert double_rounded.bits == 0x3F80
    assert direct.bits != double_rounded.bits


def test_ulp_distance_basics():
    assert ulp_distance(0x3F800000, 0x3F800000, fp32_ordered_index) == 0
    assert ulp_distance(0x3F800000, 0x3F800001, fp32_ordered_index) == 1
    assert ulp_distance(0x00000000, 0x80000000, fp32_ordered_index) == 0
    assert ulp_distance(0x00000001, 0x80000001, fp32_ordered_index) == 2
    assert ulp_distance(0x3F800000, 0xBF800000, fp32_ordered_index) == 2 * 0x3F800000
    assert ulp_distance(0x3F7FFFFF, 0x3F800000, fp32_ordered_index) == 1
    assert ulp_distance(0x3F800000, 0x40000000, fp32_ordered_index) == 1 << 23


def test_ulp_distance_is_format_specific():
    assert ulp_distance(0x3F80, 0x3F81, bf16_ordered_index) == 1
    assert ulp_distance(0x3F80, 0x3F82, bf16_ordered_index) == 2
    assert ulp_distance(0x3F800000, 0x3F808000, fp32_ordered_index) == 0x8000
    assert ulp_distance(0x3F80, 0x4000, bf16_ordered_index) == 128
    assert ulp_distance(0x3F80, 0xBF80, bf16_ordered_index) == 2 * 0x3F80


def test_ulp_distance_rejects_non_finite():
    with pytest.raises(ValueError):
        ulp_distance(0x7F800000, 0x3F800000, fp32_ordered_index)
    with pytest.raises(ValueError):
        ulp_distance(0x7F80, 0x3F80, bf16_ordered_index)
    with pytest.raises(ValueError):
        ulp_distance(0x7FC00000, 0x3F800000, fp32_ordered_index)


def test_ulp_index_monotonic_on_positive_values():
    rng = random.Random(5)
    values = sorted({rng.randrange(0, 0x7F800000) for _ in range(200)})
    indices = [fp32_ordered_index(value) for value in values]
    assert indices == sorted(indices)
    assert all(b > a for a, b in zip(indices, indices[1:]))
    for base in (0, 1, 0x3F000000, 0x7F7FFFF0):
        run = [fp32_ordered_index(base + offset) for offset in range(16)]
        assert all(b - a == 1 for a, b in zip(run, run[1:]))


def test_fp32_bits_of_bf16_values_are_consistent():
    value = round_dyadic_to_bf16(Dyadic(3, -1))[0]
    fp32 = Fp32.from_float(value.to_float())
    assert fp32.exact_dyadic() == value.exact_dyadic()
