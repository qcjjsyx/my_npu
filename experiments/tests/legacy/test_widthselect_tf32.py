"""TF32 RNE quantiser tests against an independent Fraction oracle."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.widthselect.tf32_rne import (
    MANTISSA_BITS,
    MAX_FINITE_MAGNITUDE,
    MIN_SUBNORMAL_EXPONENT,
    SIGN_BIT,
    decode_tf32_magnitude,
    decode_tf32_value,
    is_special_code,
    metadata,
    round_dyadic_to_tf32,
)
from oracle import oracle_tf32_code


def _denominator_exp(value: Fraction) -> int:
    denominator = value.denominator
    if denominator & (denominator - 1):
        raise AssertionError("test helper requires a dyadic fraction")
    return denominator.bit_length() - 1


def _round_exact(value: Fraction) -> tuple[int, bool]:
    result = round_dyadic_to_tf32(Dyadic(value.numerator, -_denominator_exp(value)))
    return result.code, result.overflowed


def test_roundtrip_all_finite_codes():
    for raw_exp in range(0, 255):
        fracs = range(0, 1024, 37) if raw_exp not in (0, 1, 254) else range(1024)
        for frac in fracs:
            code = (raw_exp << 10) | frac
            value = decode_tf32_value(code)
            back = round_dyadic_to_tf32(value)
            assert not back.overflowed
            assert back.code == code, (hex(code), hex(back.code))
            assert not is_special_code(code)


def test_special_codes_are_rejected():
    for code in ((0xFF << 10) | 0, (0xFF << 10) | 1, (1 << 18) | (0xFF << 10) | 5):
        with pytest.raises(ValueError):
            decode_tf32_value(code)
        with pytest.raises(ValueError):
            decode_tf32_magnitude(code)


def test_signed_zero_roundtrip():
    assert round_dyadic_to_tf32(Dyadic.zero()).code == 0
    assert round_dyadic_to_tf32(Dyadic.zero(), zero_sign=1).code == (1 << 18)
    assert decode_tf32_value(0) == Dyadic.zero()
    assert decode_tf32_value(1 << 18) == Dyadic.zero()


def test_random_values_match_oracle():
    rng = random.Random(20260926)
    checked = 0
    for _ in range(4000):
        nbits = rng.randint(1, 30)
        m = rng.getrandbits(nbits) or 1
        if rng.random() < 0.5:
            m = -m
        e = rng.randint(-180, 110)
        value = Fraction(m) * Fraction(2) ** e
        ours = _round_exact(value)
        oracle = oracle_tf32_code(value)
        assert ours == oracle, (value, ours, oracle)
        checked += 1
    assert checked == 4000


def test_carry_into_exponent():
    # (2**11 - 1) * 2**-10 + half an ulp rounds up to 2.0
    value = Fraction(4095, 2048)  # 1.99951171875
    code, overflowed = _round_exact(value)
    assert not overflowed
    raw_exp = (code >> 10) & 0xFF
    frac = code & 0x3FF
    assert raw_exp == 128 and frac == 0

    value = Fraction(3) * Fraction(2) ** -1  # 1.5 exactly representable
    code, _ = _round_exact(value)
    assert decode_tf32_value(code) == Dyadic(3, -1)


def test_subnormal_grid_and_underflow():
    lsb = Fraction(2) ** MIN_SUBNORMAL_EXPONENT
    # exactly 3 subnormal LSBs
    assert decode_tf32_value(_round_exact(3 * lsb)[0]) == Dyadic(3, MIN_SUBNORMAL_EXPONENT)
    # ties-to-even at half an LSB -> zero
    code, overflowed = _round_exact(lsb / 2)
    assert code == 0 and not overflowed
    # just above half an LSB -> one LSB
    code, _ = _round_exact(lsb / 2 + Fraction(2) ** -200)
    assert decode_tf32_value(code) == Dyadic(1, MIN_SUBNORMAL_EXPONENT)
    # FP32's smallest subnormal underflows to zero
    result = round_dyadic_to_tf32(Dyadic(1, -149))
    assert result.code == 0 and result.zeroed


def test_subnormal_normal_boundary_carry():
    # (2**10 - 0.5) * 2**-136 rounds up to the smallest normal
    value = (Fraction(1024) - Fraction(1, 2)) * Fraction(2) ** -136
    code, _ = _round_exact(value)
    assert ((code >> 10) & 0xFF) == 1 and (code & 0x3FF) == 0
    assert decode_tf32_value(code) == Dyadic(1, -126)


def test_overflow_near_fp32_max():
    # FP32 max is closer to 2**128 than to the largest finite TF32 -> +Inf
    fp32_max = Fraction((1 << 24) - 1) * Fraction(2) ** 104
    result = round_dyadic_to_tf32(Dyadic((1 << 24) - 1, 104))
    assert result.overflowed
    assert result.code == (0xFF << 10)
    assert not result.saturated
    # just below the rounding midpoint stays finite at the largest magnitude
    midpoint = Fraction(2) ** 128 - Fraction(2) ** 116
    below = midpoint - Fraction(2) ** 100
    code, overflowed = _round_exact(below)
    assert not overflowed
    assert ((code >> 10) & 0xFF) == 254 and (code & 0x3FF) == 1023
    assert oracle_tf32_code(fp32_max)[1] is True


def test_input_overflow_saturates_to_max_finite():
    fp32_max = Dyadic((1 << 24) - 1, 104)
    saturated = round_dyadic_to_tf32(fp32_max, overflow="saturate")
    assert saturated.overflowed and saturated.saturated
    assert saturated.raw_exp == 254 and saturated.frac == 1023
    assert decode_tf32_value(saturated.code) == MAX_FINITE_MAGNITUDE
    negative = round_dyadic_to_tf32(-fp32_max, overflow="saturate")
    assert negative.saturated and (negative.code >> SIGN_BIT) == 1
    assert decode_tf32_value(negative.code) == -MAX_FINITE_MAGNITUDE
    assert not is_special_code(saturated.code) and not is_special_code(negative.code)
    with pytest.raises(ValueError):
        round_dyadic_to_tf32(fp32_max, overflow="clamp")


def test_decode_magnitude_consistency():
    rng = random.Random(11)
    for _ in range(2000):
        code = rng.randrange(0, (0xFF << 10))
        sign, t, e = decode_tf32_magnitude(code)
        if t == 0:
            value = Dyadic.zero()
        else:
            magnitude = Dyadic(t, e - 137)
            value = -magnitude if sign else magnitude
        assert value == decode_tf32_value(code)


def test_metadata_policy():
    meta = metadata()
    assert meta["mantissa_bits"] == MANTISSA_BITS
    assert "RNE" in meta["rounding"]
    assert meta["policy"] == "fp32_significand_rne_11bit_kept_subnormal_input_overflow_saturated"
    assert "saturates" in meta["overflow"]
