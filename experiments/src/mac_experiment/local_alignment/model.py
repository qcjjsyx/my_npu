"""Exact folded operand decoding and finite local alignment windows."""
from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import rne_shift, round_dyadic_to_fp32

# element exponent bits, fraction bits, bias, largest positive finite code
FORMATS = {
    "e4m3fn": (4, 3, 7, 0x7e),
    "e5m2": (5, 2, 15, 0x7b),
    "fp16": (5, 10, 15, 0x7bff),
    "bf16": (8, 7, 127, 0x7f7f),
}
# signed alignment-coefficient container C, exponent granularity g, product bias b, K
# bf16_postfold uses an 18-bit complete product, then exact 21-bit recoding.
LAYOUTS = {"fp8": (16, 4, 36, 16), "fp16": (30, 4, 50, 8), "bf16": (18, 1, 268, 16),
           "bf16_postfold": (21, 4, 268, 16)}
PAIRS = {
    "mxfp8_e4m3fn": ("e4m3fn", "e4m3fn"),
    "mxfp8_e5m2": ("e5m2", "e5m2"),
    "mxfp8_e4m3fn_e5m2": ("e4m3fn", "e5m2"),
    "mxfp8_e5m2_e4m3fn": ("e5m2", "e4m3fn"),
    "fp16": ("fp16", "fp16"), "bf16": ("bf16", "bf16"),
    "bf16_postfold": ("bf16", "bf16"),
}


def family(mode):
    if mode not in PAIRS:
        raise ValueError(f"unknown mode {mode!r}")
    return "fp8" if mode.startswith("mxfp8") else mode


def mapped_code(code, fmt):
    eb, fb, _, maximum = FORMATS[fmt]
    if not isinstance(code, int) or not 0 <= code < 1 << (eb + fb + 1):
        raise ValueError("element code out of range")
    sign = code >> (eb + fb)
    magnitude = code & ((1 << (eb + fb)) - 1)
    exp, frac = magnitude >> fb, magnitude & ((1 << fb) - 1)
    if fmt == "e4m3fn":
        return 0 if magnitude == 0x7f else code
    if exp == (1 << eb) - 1:
        return 0 if frac else (sign << (eb + fb)) | maximum
    return code


def decode(code, fmt):
    """Return (signed coefficient M, coarse exponent Q), after entry mapping."""
    code = mapped_code(code, fmt)
    eb, fb, _, _ = FORMATS[fmt]
    sign = code >> (eb + fb)
    exp = (code >> fb) & ((1 << eb) - 1)
    frac = code & ((1 << fb) - 1)
    m = ((1 << fb) if exp else 0) + frac
    if not m:
        return 0, 0
    e = exp or 1
    if fmt == "e4m3fn":
        e += 8
    elif fmt == "e5m2":
        m <<= 1
    if fmt != "bf16":
        m <<= e % 4
        e //= 4
    return (-m if sign else m), e


def element_value(code, fmt):
    m, q = decode(code, fmt)
    g, bias = (1, 134) if fmt == "bf16" else (4, 25 if fmt == "fp16" else 18)
    return Dyadic(m, g * q - bias)


def check_signed(n, width):
    if not -(1 << (width - 1)) <= n < 1 << (width - 1):
        raise ArithmeticError(f"{n} overflows signed{width}")


def postfold_product(p, t):
    """Exact BF16 post-multiply recoding P'=P*2**(T mod 4), Q=T//4."""
    check_signed(p, 18)
    if not p:
        return 0, 0
    coefficient = p << (t % 4)
    check_signed(coefficient, 21)
    return coefficient, t // 4


@dataclass
class Block:
    q: Dyadic
    s: Dyadic
    l1: Dyadic
    tmax: int | None
    integer_sum: int
    changed: int
    zeroed: int
    terms: list


def align(products, layout, F, rounding, h=0):
    """n=floor/RNE(P*2**(F-g*(Tmax-T))); Q=sum(n)*2**(g*Tmax-b-F+h)."""
    if type(F) is not int or F < 0 or rounding not in ("floor", "rne"):
        raise ValueError("F must be nonnegative integer; rounding must be floor or rne")
    C, g, b, K = LAYOUTS[layout]
    if len(products) != K:
        raise ValueError(f"expected exactly K={K} products (externally pad tail with zero)")
    nonzero = [(p, t) for p, t in products if p]
    tmax = max((t for p, t in nonzero), default=None)
    s = l1 = Dyadic.zero()
    total = changed = zeroed = 0
    terms = []
    for p, t in products:
        check_signed(p, C)
        value = Dyadic(p, g * t - b + h)
        s, l1 = s + value, l1 + abs(value)
        shift = g * (tmax - t) - F if p else 0
        n = (p >> shift if shift > 0 else p << -shift) if rounding == "floor" else rne_shift(p, shift)
        check_signed(n, C + F)
        recovered = Dyadic(n, g * tmax - b - F + h) if tmax is not None else Dyadic.zero()
        changed += recovered != value
        zeroed += p != 0 and n == 0
        total += n
        terms.append({"P": p, "T": t, "shift": shift, "n": n})
    check_signed(total, C + F + (K - 1).bit_length())
    q = Dyadic(total, g * tmax - b - F + h) if tmax is not None else Dyadic.zero()
    return Block(q, s, l1, tmax, total, changed, zeroed, terms)


def output_fp32(value):
    """RNE, gradual underflow, signed tiny zero, exact zero +0, finite saturation."""
    fp, exact = round_dyadic_to_fp32(value)
    saturated = exact is None
    if saturated:
        from ..arithmetic.fp32 import Fp32
        fp = Fp32((fp.sign << 31) | 0x7f7fffff)
    return fp, saturated
