"""Output MX quantisation tests: exact scale policy, thresholds, events."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.formats.e8m0 import exponent_to_code
from mac_experiment.matrix.quantize import (
    ceil_log2_fraction,
    decoded_group_values,
    quantize_matrix_independent,
    quantize_matrix_shared,
    select_scale_exponent,
)


def _values(*entries):
    return [Dyadic(m, e) for m, e in entries]


def test_ceil_log2_fraction_exact():
    assert ceil_log2_fraction(Fraction(1)) == 0
    assert ceil_log2_fraction(Fraction(1, 2)) == -1
    assert ceil_log2_fraction(Fraction(1, 2) + Fraction(1, 10**9)) == 0
    assert ceil_log2_fraction(Fraction(448, 448)) == 0
    assert ceil_log2_fraction(Fraction(449, 448)) == 1
    assert ceil_log2_fraction(Fraction(448 * (2**127), 448)) == 127


def test_scale_selection_thresholds_both_sides():
    assert select_scale_exponent(Dyadic(448, 0)) == (0, False, False)
    assert select_scale_exponent(Dyadic(449, 0)) == (1, False, False)
    assert select_scale_exponent(Dyadic(224, 0)) == (-1, False, False)
    assert select_scale_exponent(Dyadic(225, 0)) == (0, False, False)
    exact_power = Dyadic(448 * (2**5), 0)
    assert select_scale_exponent(exact_power) == (5, False, False)
    assert select_scale_exponent(exact_power + Dyadic(1, 0)) == (6, False, False)
    assert select_scale_exponent(Dyadic(1, -200)) == (-127, True, False)
    high = Dyadic(448 * (2**128), 0)
    assert select_scale_exponent(high) == (127, False, True)


def test_all_zero_group_canonical_scale_and_signed_zeros():
    values = [Dyadic.zero()] * 32
    signs = [0] * 32
    signs[3] = 1
    quant = quantize_matrix_independent(values, signs, 1, 32)
    group = quant.groups[0]
    assert group.scale_code == exponent_to_code(0)
    assert group.fully_zero
    assert set(group.element_codes) == {0x00, 0x80}
    assert group.element_codes[3] == 0x80
    assert quant.events["fully_zero_group"] == 1


def test_independent_encoding_and_decoding_round_trip():
    values = _values((448, 0), (-1, -9), (1, -20), (7, -9))
    values += [Dyadic.zero()] * 28
    quant = quantize_matrix_independent(values, None, 1, 32)
    group = quant.groups[0]
    assert group.scale_code == exponent_to_code(0)
    decoded = decoded_group_values(group)
    assert decoded[0] == Dyadic(448, 0)
    assert decoded[1] == Dyadic(-1, -9)
    assert decoded[2] == Dyadic.zero()  # 2**-20 underflows the E4M3 grid
    assert quant.events["element_zeroed"] >= 1
    assert quant.events["element_overflow"] == 0


def test_shared_scale_never_reselects_and_saturates():
    values = _values((1000, 0)) + [Dyadic.zero()] * 31
    scale_code = exponent_to_code(0)
    quant = quantize_matrix_shared(values, None, 1, 32, [scale_code])
    group = quant.groups[0]
    assert group.scale_code == scale_code
    assert group.element_codes[0] == 0x7E  # saturated +448
    assert quant.events["element_overflow"] == 1
    assert quant.events["element_saturated"] == 1

    negative = quantize_matrix_shared(_values((-1000, 0)) + [Dyadic.zero()] * 31, None, 1, 32, [scale_code])
    assert negative.groups[0].element_codes[0] == 0xFE


def test_reserved_scale_code_invalid_group():
    values = [Dyadic(1, 0)] * 32
    quant = quantize_matrix_shared(values, None, 1, 32, [0xFF])
    group = quant.groups[0]
    assert group.invalid and group.invalid_reason == "reserved_scale_code"
    assert quant.events["reserved_scale_code"] == 1


def test_nonfinite_value_invalid_group():
    values = [Dyadic(1, 0)] * 31 + [None]
    quant = quantize_matrix_independent(values, None, 1, 32)
    group = quant.groups[0]
    assert group.invalid and group.invalid_reason == "nonfinite_value"
    assert quant.events["nonfinite_group"] == 1
    assert quant.events["nonfinite_element"] == 1


def test_clamping_events_recorded():
    high = quantize_matrix_independent([Dyadic(448 * (2**128), 0)] + [Dyadic.zero()] * 31, None, 1, 32)
    assert high.groups[0].clamp_high
    assert high.events["scale_clamped_high"] == 1
    low = quantize_matrix_independent([Dyadic(1, -200)] + [Dyadic.zero()] * 31, None, 1, 32)
    assert low.groups[0].clamp_low
    assert low.events["scale_clamped_low"] == 1


def test_group_axis_zero_layout():
    rows, cols = 32, 1
    values = [Dyadic(1, 0)] * 32
    quant = quantize_matrix_independent(values, None, rows, cols, group_axis=0)
    assert quant.group_grid == (1, 1)
    assert quant.groups[0].element_codes is not None
    assert quant.group_index(5, 0) == 0
    assert quant.decoded_value(5, 0) == Dyadic(1, 0)


def test_rejects_incomplete_group():
    with pytest.raises(ValueError):
        quantize_matrix_independent([Dyadic(1, 0)] * 31, None, 1, 31)
