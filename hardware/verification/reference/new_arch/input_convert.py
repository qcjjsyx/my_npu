"""Independent raw-word to 22-bit bundle reference for IP_DESIGN §2.

Decode floats as exact dyadics (coefficient times a power of two), then
encode each element in the destination field layout.
"""

from enum import IntEnum


class SourceFormat(IntEnum):
    FP4 = 0
    E4M3FN = 1
    E5M2 = 2
    BF16 = 3
    FP16 = 4
    INT4 = 5
    INT8 = 6
    INT16 = 7


def _float_dyadic(code: int, exp_bits: int, frac_bits: int, bias: int,
                  *, e4m3fn: bool = False) -> tuple[int, int]:
    sign, payload = divmod(code, 2 ** (exp_bits + frac_bits))
    exponent, fraction = divmod(payload, 2 ** frac_bits)
    max_exponent = 2 ** exp_bits - 1
    if exponent == max_exponent:
        if e4m3fn:
            if fraction == 2 ** frac_bits - 1:
                return 0, 0
        elif fraction:
            return 0, 0
        else:
            exponent, fraction = max_exponent - 1, 2 ** frac_bits - 1

    if exponent:
        coefficient = 2 ** frac_bits + fraction
    else:
        coefficient = fraction
    return (-coefficient if sign else coefficient, max(1, exponent) - bias - frac_bits)


def _fold(coefficient: int, exponent: int, width: int) -> int:
    if coefficient == 0:
        return 0
    coarse, remainder = divmod(exponent, 4)
    mantissa = coefficient * 2 ** remainder
    return coarse * 2 ** width + mantissa % 2 ** width


def _fp8_field(code: int, fmt: SourceFormat) -> int:
    if fmt == SourceFormat.E4M3FN:
        coefficient, power = _float_dyadic(code, 4, 3, 7, e4m3fn=True)
    else:
        coefficient, power = _float_dyadic(code, 5, 2, 15)
        coefficient *= 2  # Express the coefficient with three fraction bits.
        power -= 1
    return _fold(coefficient, power + 18, 8)


def _nibble_field(code: int, fmt: SourceFormat) -> int:
    if fmt == SourceFormat.INT4:
        coefficient = code - 16 if code >= 8 else code
    else:
        sign, magnitude = divmod(code, 8)
        exponent, fraction = divmod(magnitude, 2)
        # Multiply the E2M1 dyadic by two to obtain the integer coefficient.
        coefficient = (2 + fraction) * 2 ** (exponent - 1) if exponent else fraction
        if sign:
            coefficient = -coefficient
    return coefficient % 32


def convert_raw(raw: int, fmt: SourceFormat | int) -> int:
    """Return the packed bundle for one 16-bit word and a converter fmt code."""
    fmt = SourceFormat(fmt)
    if not 0 <= raw < 2 ** 16:
        raise ValueError(f"raw word outside unsigned16: {raw}")
    if fmt in (SourceFormat.FP4, SourceFormat.INT4):
        return sum(_nibble_field((raw >> (4 * lane)) % 16, fmt) << (5 * lane)
                   for lane in range(4))
    if fmt in (SourceFormat.E4M3FN, SourceFormat.E5M2):
        g1, g0 = divmod(raw, 256)
        return (_fp8_field(g1, fmt) << 11) | _fp8_field(g0, fmt)
    if fmt == SourceFormat.BF16:
        coefficient, power = _float_dyadic(raw, 8, 7, 127)
        return ((power + 134) << 9) | (coefficient % 512) if coefficient else 0
    if fmt == SourceFormat.FP16:
        coefficient, power = _float_dyadic(raw, 5, 10, 15)
        return _fold(coefficient, power + 25, 15)
    return raw
