"""Input format adapters."""

from .base import DecodedOperand, Format, ProductValue
from .fp8_e4m3 import Fp8E4M3
from .tf32 import Tf32

FORMATS = {
    Fp8E4M3.name: Fp8E4M3,
    Tf32.name: Tf32,
}


def get_format(name: str) -> Format:
    try:
        return FORMATS[name]()
    except KeyError:
        raise KeyError(f"unknown format {name!r}; known: {sorted(FORMATS)}") from None


__all__ = ["DecodedOperand", "Format", "ProductValue", "Fp8E4M3", "Tf32", "FORMATS", "get_format"]
