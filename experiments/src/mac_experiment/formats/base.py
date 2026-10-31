"""Format adapter interface.

A :class:`Format` owns everything format specific: legal encodings, exact
decoding, exact products, and the two window constants used by the generic
reducer.  The reducer itself contains no format branches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic


@dataclass(frozen=True)
class DecodedOperand:
    code: int
    sign: int
    raw_exp: int
    frac: int
    sig_int: int
    is_zero: bool
    is_special: bool


@dataclass(frozen=True)
class ProductValue:
    dyadic: Dyadic
    c: int

    @property
    def is_zero(self) -> bool:
        return self.dyadic.is_zero()


class Format(ABC):
    """Base class for input format adapters."""

    name: str
    code_bits: int
    exp_bits: int
    frac_bits: int
    bias: int
    window_offset: int
    product_lsb_offset: int
    sampling_exp_min: int
    sampling_exp_max: int
    gaussian_exp_center: int
    gaussian_exp_sigma: float
    value_exp_shift: int

    @abstractmethod
    def decode(self, code: int) -> DecodedOperand: ...

    @abstractmethod
    def encode(self, sign: int, raw_exp: int, frac: int) -> int: ...

    @abstractmethod
    def is_legal_input(self, code: int) -> bool: ...

    @abstractmethod
    def legal_fractions(self, raw_exp: int) -> tuple[int, ...]: ...

    @abstractmethod
    def positive_normal_codes(self, exp_min: int | None = None, exp_max: int | None = None) -> list[tuple[int, Dyadic]]: ...

    def operand_value(self, operand: DecodedOperand) -> Dyadic:
        if operand.is_zero:
            return Dyadic.zero()
        magnitude = Dyadic(operand.sig_int, operand.raw_exp + self.value_exp_shift)
        return -magnitude if operand.sign else magnitude

    def product(self, a: DecodedOperand, b: DecodedOperand) -> ProductValue:
        """Exact product ``signed_sig_a * signed_sig_b * 2**(C - product_lsb_offset)``."""
        c = a.raw_exp + b.raw_exp
        if a.is_zero or b.is_zero:
            return ProductValue(Dyadic.zero(), c)
        sign = a.sign ^ b.sign
        magnitude = Dyadic(a.sig_int * b.sig_int, c - self.product_lsb_offset)
        return ProductValue(-magnitude if sign else magnitude, c)

    def product_lsb(self, c: int) -> int:
        return c - self.product_lsb_offset

    def window_lsb(self, cmax: int, w: int) -> int:
        return cmax - self.window_offset - w

    def window_container_bits(self, w: int) -> int:
        """Signed integer width needed for the 16-term aligned window."""
        return w + 7

    def container_bounds(self, w: int) -> tuple[int, int]:
        bits = self.window_container_bits(w)
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "code_bits": self.code_bits,
            "sign_bits": 1,
            "exp_bits": self.exp_bits,
            "frac_bits": self.frac_bits,
            "bias": self.bias,
            "product_lsb": "C - %d" % self.product_lsb_offset,
            "window_lsb": "Cmax - %d - W" % self.window_offset,
            "sampling_exp_range": [self.sampling_exp_min, self.sampling_exp_max],
            "gaussian_exp_center": self.gaussian_exp_center,
            "gaussian_exp_sigma": self.gaussian_exp_sigma,
            "input_ftz": "raw_exp == 0 decodes to zero",
            "zero_path": "all-zero block reduces to exact zero with window_lsb=None",
        }
