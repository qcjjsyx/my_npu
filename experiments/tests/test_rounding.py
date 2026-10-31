import random

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.arithmetic.rounding import rne_shift, round_dyadic_to_fp32

from oracle import oracle_fp32_bits


def bits(x: Dyadic) -> int:
    return round_dyadic_to_fp32(x)[0].bits


def test_rne_shift_ties_to_even():
    assert rne_shift(1, 1) == 0
    assert rne_shift(3, 1) == 2
    assert rne_shift(5, 1) == 2
    assert rne_shift(7, 1) == 4
    assert rne_shift(-1, 1) == 0
    assert rne_shift(-3, 1) == -2
    assert rne_shift(-5, 1) == -2
    assert rne_shift(0, 10) == 0
    assert rne_shift(3, 0) == 3
    assert rne_shift(3, -2) == 12


def test_rne_shift_negative_left_shift_is_exact():
    assert rne_shift(-3, -4) == -48


def test_fp32_basic_values():
    assert bits(Dyadic(1, 0)) == 0x3F800000
    assert bits(Dyadic(-1, 0)) == 0xBF800000
    assert bits(Dyadic(3, 0)) == 0x40400000
    assert bits(Dyadic(0, 0)) == 0x00000000


def test_fp32_halfway_ties_to_even():
    assert bits(Dyadic((1 << 24) + 1, -24)) == 0x3F800000
    assert bits(Dyadic((1 << 24) + 3, -24)) == 0x3F800002
    assert bits(Dyadic(-((1 << 24) + 1), -24)) == 0xBF800000
    assert bits(Dyadic(-((1 << 24) + 3), -24)) == 0xBF800002


def test_fp32_subnormal_boundaries():
    assert bits(Dyadic(1, -149)) == 0x00000001
    assert bits(Dyadic(1, -150)) == 0x00000000
    assert bits(Dyadic(3, -150)) == 0x00000002
    assert bits(Dyadic(3, -151)) == 0x00000001
    assert bits(Dyadic(1, -151)) == 0x00000000
    assert bits(Dyadic(1 << 23, -149)) == 0x00800000
    assert bits(Dyadic((1 << 23) - 1, -149)) == 0x007FFFFF


def test_fp32_max_finite_and_overflow():
    max_finite = Dyadic((1 << 24) - 1, 104)
    assert bits(max_finite) == 0x7F7FFFFF
    assert bits(Dyadic((1 << 25) - 1, 103)) == 0x7F800000
    assert bits(Dyadic((1 << 25) - 3, 103)) == 0x7F7FFFFE
    assert bits(Dyadic(-((1 << 25) - 1), 103)) == 0xFF800000
    assert bits(Dyadic((1 << 24), 104)) == 0x7F800000


def test_fp32_negative_tiny_rounds_to_negative_zero():
    fp32, value = round_dyadic_to_fp32(Dyadic(-1, -200))
    assert fp32.bits == 0x80000000
    assert fp32.is_zero()
    assert value == Dyadic.zero()


def test_fp32_exact_rounded_value_returned():
    fp32, value = round_dyadic_to_fp32(Dyadic(1, -150))
    assert fp32.bits == 0
    assert value == Dyadic.zero()


def test_fp32_against_fraction_oracle_random():
    rng = random.Random(20260916)
    for _ in range(400):
        m = rng.randrange(-(1 << rng.randrange(1, 90)), (1 << rng.randrange(1, 90)) + 1)
        e = rng.randrange(-180, 140)
        x = Dyadic(m, e)
        assert bits(x) == oracle_fp32_bits(x.to_fraction()), (m, e)


def test_fp32_against_fraction_oracle_targeted_exponents():
    rng = random.Random(7)
    for e in (-160, -150, -149, -148, -127, -126, -125, -24, -23, -1, 0, 23, 103, 104, 105, 127, 128):
        for _ in range(40):
            m = rng.randrange(1, 1 << 26) * rng.choice((-1, 1))
            x = Dyadic(m, e)
            assert bits(x) == oracle_fp32_bits(x.to_fraction()), (m, e)


def test_fp32_zero_sign_conventions():
    assert round_dyadic_to_fp32(Dyadic.zero())[0].bits == 0
    negative_zero = Fp32.from_bits(0x80000000)
    assert negative_zero.is_zero()
    assert negative_zero.same_value(Fp32.zero())


def test_dyadic_addition_is_exact():
    assert Dyadic(1, -149) + Dyadic(1, -149) == Dyadic(1, -148)
    assert Dyadic(3, 2) + Dyadic(-3, 2) == Dyadic.zero()
    assert (Dyadic(5, -3) + Dyadic(1, -1)) == Dyadic(9, -3)


def test_dyadic_normalisation_and_ordering():
    assert Dyadic(4, 3) == Dyadic(1, 5)
    assert Dyadic(0, 100) == Dyadic.zero()
    assert Dyadic(1, -10) < Dyadic(1, -9)
    assert Dyadic(-1, 5) < Dyadic(1, -50)
    with pytest.raises(ValueError):
        Fp32.inf().exact_dyadic()
