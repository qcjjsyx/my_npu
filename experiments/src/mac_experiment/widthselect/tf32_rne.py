"""TF32 input quantisation with explicit RNE (new policy, not the legacy adapter).

TF32 is 1 sign bit + 8 exponent bits (bias 127) + 10 fraction bits; the normal
significand integer is ``1024 + fraction`` (11 significant bits of precision).

This module defines the *new experiment's* input semantics, which differ from
the historical dot-product adapter in ``formats/tf32.py`` (that adapter truncates
low bits and flushes subnormals to zero and is left untouched):

* the FP32 exponent range is kept;
* the significand is rounded to 11 significant bits with round-to-nearest,
  ties-to-even, computed on the exact dyadic value (never by truncating bits);
* mantissa carry propagates into the exponent;
* gradual underflow is kept: ``raw_exp == 0`` with ``frac > 0`` decodes to
  ``frac * 2**-136`` (the FP32-range subnormal grid, narrower than FP32's);
* signed zeros are kept;
* values that round to a biased exponent >= 255 are handled explicitly:
  the generic conversion (``overflow="inf"``) produces the special ``+/-Inf``
  encoding, while the input-quantization policy (``overflow="saturate"``)
  saturates to the maximum finite TF32 value and counts the event, so every
  recorded target-format input is finite and the exact-dyadic MAC engine never
  receives a special code.

Policy identifier: ``fp32_significand_rne_11bit_kept_subnormal_input_overflow_saturated``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import rne_shift

CODE_BITS = 19
SIGN_BIT = 18
EXP_BITS = 8
FRAC_BITS = 10
BIAS = 127
MANTISSA_BITS = 11
MIN_SUBNORMAL_EXPONENT = -136  # 2**-126 / 2**10
VALUE_EXP_SHIFT = -137
SUBNORMAL_RAW_EXP = 0
SPECIAL_RAW_EXP = 0xFF
MAX_FINITE_RAW_EXP = 0xFE
MAX_FINITE_MAGNITUDE = Dyadic((1 << 11) - 1, 117)  # 2047 * 2**117


@dataclass(frozen=True)
class Tf32RoundResult:
    code: int
    overflowed: bool = False
    zeroed: bool = False
    saturated: bool = False

    @property
    def is_inf(self) -> bool:
        return self.overflowed and not self.saturated

    @property
    def raw_exp(self) -> int:
        return (self.code >> FRAC_BITS) & 0xFF

    @property
    def frac(self) -> int:
        return self.code & 0x3FF


def round_dyadic_to_tf32(x: Dyadic, zero_sign: int = 0, overflow: str = "inf") -> Tf32RoundResult:
    """Correctly round an exact dyadic to TF32 with RNE ties-to-even.

    ``overflow``:
    * ``"inf"`` (default, generic correctly-rounded conversion): a rounded
      biased exponent >= 255 produces the special ``+/-Inf`` encoding;
    * ``"saturate"``: it saturates to ``+/-`` the maximum finite TF32 value
      (``raw_exp = 254``, ``frac = 1023``) with ``saturated=True``.  This is the
      input-quantization policy: every recorded input stays finite.

    ``zero_sign`` selects the sign of exact-zero and underflow-to-zero results.
    """
    if zero_sign not in (0, 1):
        raise ValueError("zero_sign must be 0 or 1")
    if overflow not in ("inf", "saturate"):
        raise ValueError(f"overflow must be 'inf' or 'saturate', got {overflow!r}")
    if x.m == 0:
        return Tf32RoundResult(code=zero_sign << SIGN_BIT)
    negative = x.m < 0
    a = abs(x.m)
    nbits = a.bit_length()
    qexp = max(x.e + nbits - MANTISSA_BITS, MIN_SUBNORMAL_EXPONENT)
    t = rne_shift(a, qexp - x.e)
    if t == 0:
        return Tf32RoundResult(code=negative << SIGN_BIT, zeroed=True)
    if t >= (1 << MANTISSA_BITS):
        t >>= 1
        qexp += 1
    if t >= (1 << (MANTISSA_BITS - 1)):
        biased = qexp + 137
        if biased >= SPECIAL_RAW_EXP:
            if overflow == "saturate":
                return Tf32RoundResult(
                    code=(negative << SIGN_BIT) | (MAX_FINITE_RAW_EXP << FRAC_BITS) | 0x3FF,
                    overflowed=True,
                    saturated=True,
                )
            return Tf32RoundResult(code=(negative << SIGN_BIT) | (SPECIAL_RAW_EXP << FRAC_BITS), overflowed=True)
        code = (negative << SIGN_BIT) | (biased << FRAC_BITS) | (t - (1 << (MANTISSA_BITS - 1)))
    else:
        if qexp != MIN_SUBNORMAL_EXPONENT:
            raise AssertionError("subnormal TF32 result must be quantised at 2**-136")
        code = (negative << SIGN_BIT) | t
    return Tf32RoundResult(code=code)


def decode_tf32_value(code: int) -> Dyadic:
    """Exact dyadic value of a finite TF32 code; specials raise."""
    if not 0 <= code <= 0x7FFFF:
        raise ValueError(f"TF32 compact code out of range: {code}")
    sign = (code >> SIGN_BIT) & 1
    raw_exp = (code >> FRAC_BITS) & 0xFF
    frac = code & 0x3FF
    if raw_exp == SPECIAL_RAW_EXP:
        raise ValueError(f"TF32 code 0x{code:05X} has a special exponent field (255)")
    if raw_exp == SUBNORMAL_RAW_EXP:
        magnitude = Dyadic(frac, MIN_SUBNORMAL_EXPONENT)
    else:
        magnitude = Dyadic(1024 + frac, raw_exp + VALUE_EXP_SHIFT)
    return -magnitude if sign else magnitude


def decode_tf32_magnitude(code: int) -> tuple[int, int, int]:
    """``(sign, t, E)`` with value ``(-1)**sign * t * 2**(E - 137)``.

    Subnormals use ``E = 1`` and ``t = frac`` (so the shared exponent formula
    covers them); zero returns ``(sign, 0, 0)``; specials raise.
    """
    if not 0 <= code <= 0x7FFFF:
        raise ValueError(f"TF32 compact code out of range: {code}")
    sign = (code >> SIGN_BIT) & 1
    raw_exp = (code >> FRAC_BITS) & 0xFF
    frac = code & 0x3FF
    if raw_exp == SPECIAL_RAW_EXP:
        raise ValueError(f"TF32 code 0x{code:05X} has a special exponent field (255)")
    if raw_exp == SUBNORMAL_RAW_EXP:
        if frac == 0:
            return sign, 0, 0
        return sign, frac, 1
    return sign, 1024 + frac, raw_exp


def is_special_code(code: int) -> bool:
    return ((code >> FRAC_BITS) & 0xFF) == SPECIAL_RAW_EXP


def code_fingerprint_payload(code: int) -> bytes:
    return int(code).to_bytes(4, "little")


def metadata() -> dict:
    return {
        "name": "tf32",
        "layout": "sign(1) | raw_exp(8, bias 127) | frac(10)",
        "mantissa_bits": MANTISSA_BITS,
        "precision": "11 significant bits (1 implicit + 10 explicit)",
        "rounding": "RNE ties-to-even on the exact dyadic; no bit truncation",
        "carry": "mantissa carry propagates into the exponent",
        "subnormal": "raw_exp=0 with frac>0 decodes to frac * 2**-136 (gradual underflow kept)",
        "signed_zero": "+0 / -0 kept",
        "overflow": (
            "generic conversion rounds a biased exponent >= 255 to +/-Inf; input quantization "
            "saturates to +/-max finite (raw_exp=254, frac=1023) and counts the event"
        ),
        "fp32_exponent_range": "kept",
        "policy": "fp32_significand_rne_11bit_kept_subnormal_input_overflow_saturated",
        "legacy_adapter_note": (
            "formats/tf32.py (truncation + FTZ) is a different historical adapter and is not used here"
        ),
    }
