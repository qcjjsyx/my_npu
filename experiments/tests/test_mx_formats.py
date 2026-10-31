"""E4M3 element / E8M0 scale codec tests with an independent Fraction oracle."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.ulp import e4m3_ordered_index, ulp_distance
from mac_experiment.formats.e4m3_element import (
    E4M3Element,
    decode_element,
    is_nan_code,
    round_dyadic_to_e4m3,
)
from mac_experiment.formats.e8m0 import (
    MAX_EXPONENT,
    MIN_EXPONENT,
    clamp_exponent,
    code_to_exponent,
    exponent_to_code,
    is_finite_code,
    scale_value,
)

from oracle import _floor_log2, round_fraction_to_int

FINITE_CODES = [code for code in range(256) if not is_nan_code(code)]


def oracle_e4m3_bits(x: Fraction) -> int:
    """Independent saturating RNE E4M3 encoding of an exact rational."""
    if x == 0:
        return 0
    sign = 1 if x < 0 else 0
    a = abs(x)
    if a > 448:
        return (sign << 7) | 0x7E
    e = _floor_log2(a)
    if e >= -6:
        ulp = Fraction(2) ** (e - 3)
    else:
        ulp = Fraction(1, 2**9)
    n = round_fraction_to_int(a / ulp)
    if n == 0:
        return sign << 7
    if n >= 2**4:
        n //= 2
        e += 1
    if n >= 2**3:
        if e < -6:
            return (sign << 7) | 0x08
        biased = e + 7
        return (sign << 7) | (biased << 3) | (n - 2**3)
    if ulp != Fraction(1, 2**9):
        raise AssertionError("subnormal E4M3 result with normal ulp")
    return (sign << 7) | n


def test_finite_code_roundtrip():
    for code in FINITE_CODES:
        value = decode_element(code)
        zero_sign = (code >> 7) & 1
        result = round_dyadic_to_e4m3(value, zero_sign=zero_sign)
        assert result.code == code, (hex(code), value, hex(result.code))


def test_subnormal_and_normal_boundary():
    assert decode_element(0x01) == Dyadic(1, -9)
    assert decode_element(0x07) == Dyadic(7, -9)
    assert decode_element(0x08) == Dyadic(1, -6)
    assert decode_element(0x0F) == Dyadic(15, -9)
    assert decode_element(0x10) == Dyadic(1, -5)
    assert decode_element(0x38) == Dyadic(1, 0)
    assert decode_element(0x7E) == Dyadic(14, 5)
    assert decode_element(0x7E).to_float() == 448.0
    assert decode_element(0x00).is_zero()
    assert decode_element(0x80).is_zero()


def test_signed_zero_preserved():
    assert round_dyadic_to_e4m3(Dyadic.zero(), zero_sign=0).code == 0x00
    assert round_dyadic_to_e4m3(Dyadic.zero(), zero_sign=1).code == 0x80
    tiny = round_dyadic_to_e4m3(Dyadic(-1, -12))
    assert tiny.code == 0x80
    assert tiny.zeroed


def test_nan_codes_rejected():
    assert is_nan_code(0x7F) and is_nan_code(0xFF)
    fmt = E4M3Element()
    assert fmt.decode(0x7F).is_special
    assert fmt.decode(0xFF).is_special
    for code in (0x7F, 0xFF):
        with pytest.raises(ValueError):
            decode_element(code)
    assert fmt.decode(0x7E).is_special is False


def test_ties_to_even():
    assert round_dyadic_to_e4m3(Dyadic(1, 0) + Dyadic(1, -4)).code == 0x38  # 1.0 even
    assert round_dyadic_to_e4m3(Dyadic(1, 0) + Dyadic(3, -4)).code == 0x3A  # 1.25 even
    three_halves_quantum = Dyadic(3, -10)
    result = round_dyadic_to_e4m3(three_halves_quantum)
    assert result.code == 0x02  # ties to even subnormal quantum


def test_saturation_and_overflow_events():
    result = round_dyadic_to_e4m3(Dyadic(512, 0))
    assert result.code == 0x7E
    assert result.saturated and result.overflowed
    negative = round_dyadic_to_e4m3(Dyadic(-512, 0))
    assert negative.code == 0xFE
    assert negative.saturated
    between = round_dyadic_to_e4m3(Dyadic(460, 0))
    assert between.code == 0x7E
    assert between.overflowed
    assert not between.saturated


def test_zeroed_event_on_underflow():
    result = round_dyadic_to_e4m3(Dyadic(1, -20))
    assert result.code == 0x00
    assert result.zeroed and not result.saturated


def test_against_fraction_oracle_random():
    rng = random.Random(20260921)
    for _ in range(1000):
        m = rng.randrange(-(1 << rng.randrange(1, 40)), (1 << rng.randrange(1, 40)) + 1)
        e = rng.randrange(-30, 12)
        x = Dyadic(m, e)
        expected = oracle_e4m3_bits(x.to_fraction())
        actual = round_dyadic_to_e4m3(x).code
        assert actual == expected, (m, e)


def test_e8m0_coding_and_reserved_code():
    assert exponent_to_code(0) == 127
    assert exponent_to_code(-127) == 0
    assert exponent_to_code(127) == 254
    assert code_to_exponent(0) == -127
    assert code_to_exponent(254) == 127
    assert scale_value(127) == Dyadic(1, 0)
    assert scale_value(0) == Dyadic(1, -127)
    assert is_finite_code(254)
    assert not is_finite_code(255)
    with pytest.raises(ValueError):
        code_to_exponent(255)
    with pytest.raises(ValueError):
        scale_value(255)
    with pytest.raises(ValueError):
        exponent_to_code(128)
    with pytest.raises(ValueError):
        exponent_to_code(-128)


def test_e8m0_clamp():
    assert clamp_exponent(3) == (3, False, False)
    assert clamp_exponent(MIN_EXPONENT - 5) == (MIN_EXPONENT, True, False)
    assert clamp_exponent(MAX_EXPONENT + 5) == (MAX_EXPONENT, False, True)


def test_e4m3_ulp_index_finite_and_signed_zero():
    assert e4m3_ordered_index(0x00) == 0
    assert e4m3_ordered_index(0x80) == 0
    assert ulp_distance(0x00, 0x80, e4m3_ordered_index) == 0
    assert ulp_distance(0x01, 0x02, e4m3_ordered_index) == 1
    assert ulp_distance(0x38, 0x39, e4m3_ordered_index) == 1
    with pytest.raises(ValueError):
        e4m3_ordered_index(0x7F)
    with pytest.raises(ValueError):
        e4m3_ordered_index(0xFF)
    values = [code for code in range(0x00, 0x7F)] + [code for code in range(0x80, 0xFF)]
    indices = sorted(e4m3_ordered_index(code) for code in values)
    assert indices == list(range(-0x7E, 0)) + [0, 0] + list(range(1, 0x7F))
