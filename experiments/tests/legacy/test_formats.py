import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.formats import get_format


def test_fp8_constants_match_brief_table():
    fmt = get_format("fp8_e4m3")
    assert fmt.product_lsb(25) == 25 - 20
    assert fmt.window_lsb(25, 18) == 25 - 14 - 18
    assert fmt.window_container_bits(18) == 25
    assert fmt.bias == 7
    assert fmt.frac_bits == 3
    assert fmt.exp_bits == 4


def test_tf32_constants_match_brief_table():
    fmt = get_format("tf32")
    assert fmt.product_lsb(274) == 0
    assert fmt.product_lsb(316) == 316 - 274
    assert fmt.window_lsb(316, 20) == 316 - 254 - 20
    assert fmt.window_container_bits(20) == 27
    assert fmt.bias == 127
    assert fmt.frac_bits == 10
    assert fmt.exp_bits == 8
    assert fmt.code_bits == 19


def test_fp8_decode_zero_nan_and_finite_max():
    fmt = get_format("fp8_e4m3")
    zero = fmt.decode(0x00)
    assert zero.is_zero and not zero.is_special
    assert fmt.operand_value(zero) == Dyadic.zero()
    assert fmt.decode(0x80).is_zero
    nan = fmt.decode(0x7F)
    assert nan.is_special
    assert fmt.decode(0xFF).is_special
    top = fmt.decode(0x7E)
    assert not top.is_special
    assert fmt.operand_value(top) == Dyadic(14, 5)
    assert fmt.operand_value(top).to_float() == 448.0
    one = fmt.decode(fmt.encode(0, 7, 0))
    assert fmt.operand_value(one).to_float() == 1.0


def test_fp8_legal_fractions_exclude_nan_code():
    fmt = get_format("fp8_e4m3")
    assert fmt.legal_fractions(15) == (0, 1, 2, 3, 4, 5, 6)
    assert len(fmt.legal_fractions(7)) == 8
    assert fmt.is_legal_input(fmt.encode(0, 15, 6))
    assert not fmt.is_legal_input(0x7F)
    assert not fmt.is_legal_input(0x00)


def test_fp8_product_uses_raw_exponent_sum():
    fmt = get_format("fp8_e4m3")
    a = fmt.decode(fmt.encode(0, 7, 0))
    b = fmt.decode(fmt.encode(1, 7, 0))
    product = fmt.product(a, b)
    assert product.c == 14
    assert product.dyadic == Dyadic(-64, -6)
    assert product.dyadic.to_float() == -1.0


def test_tf32_compact_encoding_roundtrip():
    fmt = get_format("tf32")
    code = fmt.encode(1, 127, 512)
    assert code == (1 << 18) | (127 << 10) | 512
    operand = fmt.decode(code)
    assert operand.sign == 1 and operand.raw_exp == 127 and operand.frac == 512
    assert fmt.operand_value(operand).to_float() == -1.5


def test_tf32_zero_special_and_fp32_bridge():
    fmt = get_format("tf32")
    assert fmt.decode(0).is_zero
    assert fmt.decode(0xFF << 10).is_special
    assert fmt.decode((0xFF << 10) | 1).is_special
    assert not fmt.decode(fmt.encode(0, 254, 1023)).is_special
    fp32_one = 0x3F800000
    assert fmt.from_fp32_bits(fp32_one) == fmt.encode(0, 127, 0)
    fp32_two = 0x40000000
    assert fmt.decode(fmt.from_fp32_bits(fp32_two)).raw_exp == 128


def test_tf32_product_matches_fraction():
    fmt = get_format("tf32")
    a = fmt.decode(fmt.encode(0, 130, 0))
    b = fmt.decode(fmt.encode(1, 128, 1023))
    product = fmt.product(a, b)
    expected = -((1024) * (2047)) * (2 ** (258 - 274))
    assert product.dyadic.to_fraction() == expected


def test_positive_normal_code_sets_are_legal_and_sorted():
    for name, expected_count in (("fp8_e4m3", 119), ("tf32", 63 * 1024)):
        fmt = get_format(name)
        codes = fmt.positive_normal_codes()
        assert len(codes) == expected_count
        values = [value for _, value in codes]
        assert values == sorted(values)
        for code, value in codes:
            operand = fmt.decode(code)
            assert not operand.is_zero and not operand.is_special
            assert fmt.operand_value(operand) == value
    fp8 = get_format("fp8_e4m3")
    with pytest.raises(ValueError):
        fp8.decode(256)
    tf32 = get_format("tf32")
    with pytest.raises(ValueError):
        tf32.decode(1 << 19)
