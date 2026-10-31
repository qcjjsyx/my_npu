"""E8M0 scale format for MXFP8 tensors (OCP MX semantics).

E8M0 is an 8-bit exponent-only format with bias 127 and no sign bit:

* code ``c`` in ``0..254`` encodes the exact value ``2**(c - 127)``;
* code ``255`` (``0xFF``) is reserved as NaN and is never a finite scale.

Scales are applied exactly by shifting the dyadic exponent; no host floating
point multiplication is used anywhere.  Authority: OCP Microscaling Formats
(MX) Specification v1.0; see ``docs/mx_semantics.md``.
"""

from __future__ import annotations

from ..arithmetic.dyadic import Dyadic

NAN_CODE = 0xFF
BIAS = 127
MIN_EXPONENT = -127
MAX_EXPONENT = 127
MIN_FINITE_CODE = 0
MAX_FINITE_CODE = 254


def is_finite_code(code: int) -> bool:
    if not 0 <= code <= 0xFF:
        raise ValueError(f"E8M0 code out of range: {code}")
    return code != NAN_CODE


def exponent_to_code(exponent: int) -> int:
    if not MIN_EXPONENT <= exponent <= MAX_EXPONENT:
        raise ValueError(f"E8M0 exponent out of finite range [{MIN_EXPONENT}, {MAX_EXPONENT}]: {exponent}")
    return exponent + BIAS


def code_to_exponent(code: int) -> int:
    if not is_finite_code(code):
        raise ValueError("E8M0 code 0xFF is reserved (NaN) and is not a finite scale")
    return code - BIAS


def scale_value(code: int) -> Dyadic:
    """Exact power of two of a finite E8M0 code."""
    return Dyadic(1, code_to_exponent(code))


def clamp_exponent(exponent: int) -> tuple[int, bool, bool]:
    """Clamp to the finite E8M0 exponent range; returns ``(e, low, high)``."""
    if exponent < MIN_EXPONENT:
        return MIN_EXPONENT, True, False
    if exponent > MAX_EXPONENT:
        return MAX_EXPONENT, False, True
    return exponent, False, False


def metadata() -> dict:
    return {
        "name": "e8m0",
        "code_bits": 8,
        "bias": 127,
        "finite_codes": [MIN_FINITE_CODE, MAX_FINITE_CODE],
        "reserved_code": NAN_CODE,
        "reserved_semantics": "NaN; never used as a finite scale",
        "value_rule": "2**(code - 127)",
        "exponent_range": [MIN_EXPONENT, MAX_EXPONENT],
        "application": "exact dyadic exponent shift; no host float multiplication",
    }
