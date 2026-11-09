"""FP32 source -> saturating RNE target codes, and per-side K-axis MX scales."""
from collections import Counter
from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..arithmetic.rounding import rne_shift
from ..matrix.quantize import ceil_log2_fraction
from .model import FORMATS, element_value


@dataclass
class Quantized:
    codes: list
    scales: list
    events: dict


def encode(value, fmt, zero_sign=0):
    eb, fb, bias, maximum = FORMATS[fmt]
    sign = int(value.m < 0) if value.m else zero_sign
    if not value.m:
        return sign << (eb + fb), False
    a = abs(value.m)
    qe = max(value.e + a.bit_length() - fb - 1, 1 - bias - fb)
    n = rne_shift(a, qe - value.e)
    if n >= 1 << (fb + 1):
        n >>= 1
        qe += 1
    if n < 1 << fb:
        magnitude = n
    else:
        exponent = qe + bias + fb
        magnitude = (exponent << fb) | (n - (1 << fb))
    saturated = magnitude > maximum
    return (sign << (eb + fb)) | min(magnitude, maximum), saturated


def quantize(bits, fmt, mx=False):
    if mx and fmt not in ("e4m3fn", "e5m2"):
        raise ValueError("MX quantization is supported only for FP8")
    values = [Fp32(x).exact_dyadic() for x in bits]
    codes, scales, events = [], [], Counter(total=len(bits), saturated=0, zeroed=0, subnormal=0,
                                             scale_clamped_low=0, scale_clamped_high=0)
    maximum = abs(element_value(FORMATS[fmt][3], fmt))
    for start in range(0, len(bits), 32 if mx else max(1, len(bits))):
        block = values[start:start + 32] if mx else values
        amax = max(map(abs, block), default=Dyadic.zero())
        raw = ceil_log2_fraction(amax.to_fraction() / maximum.to_fraction()) if mx and amax.m else 0
        e = max(-127, min(127, raw)) if mx else 0
        if mx:
            scales.append(e + 127)
            events["scale_clamped_low"] += raw < -127
            events["scale_clamped_high"] += raw > 127
        for offset, value in enumerate(block):
            code, sat = encode(Dyadic(value.m, value.e - e), fmt, bits[start + offset] >> 31)
            fb = FORMATS[fmt][1]
            mag = code & ((1 << (FORMATS[fmt][0] + fb)) - 1)
            events["saturated"] += sat
            events["zeroed"] += bool(value.m and not mag)
            events["subnormal"] += 0 < mag < (1 << fb)
            codes.append(code)
    return Quantized(codes, scales, dict(events))
