"""Mathematical forward reference from 22-bit fields to the ACC contribution.

Use full integer products, unbounded floor division and direct summation.
The RTL's multiplier split, tree packing, and pipeline are checked separately.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from .acc_fixed import normalize_contribution


class Mode(IntEnum):
    FP4 = 0
    FP8 = 1
    BF16 = 2
    FP16 = 3
    INT4 = 4
    INT8 = 5
    INT16 = 6


ALIGNMENT = {
    Mode.FP8: (16, 36),
    Mode.BF16: (11, 268),
    Mode.FP16: (8, 50),
}


def signed_field(word: int, start: int, width: int) -> int:
    """Extract a signed field with its own sign bit."""
    value = (word >> start) % (1 << width)
    return value - (1 << width) if value >= (1 << (width - 1)) else value


def decode_groups(mode: Mode, fields: Sequence[int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Read each active element as (signed coefficient, exponent field)."""
    if len(fields) != 16:
        raise ValueError("expected 16 physical input slots per side")
    if mode == Mode.FP8:
        return tuple(tuple((signed_field(word, bit, 8), (word >> (bit + 8)) & 7)
                           for word in fields) for bit in (0, 11))
    if mode in (Mode.FP4, Mode.INT4):
        return tuple(tuple((signed_field(word, bit, 5), 0)
                           for word in fields for bit in bits)
                     for bits in ((0, 5), (10, 15)))
    if mode == Mode.INT8:
        return tuple(tuple((signed_field(word, bit, 8), 0) for word in fields)
                     for bit in (0, 8))
    if mode == Mode.BF16:
        return (tuple((signed_field(word, 0, 9), (word >> 9) & 255) for word in fields),)
    if mode == Mode.FP16:
        return (tuple((signed_field(word, 0, 15), (word >> 15) & 7)
                      for word in fields[::2]),)
    return (tuple((signed_field(word, 0, 16), 0) for word in fields[::2]),)


@dataclass(frozen=True)
class LocalGroup:
    products: tuple[tuple[int, int], ...]
    max_q: int
    terms: tuple[int, ...]
    align_e: int
    scale: int

    @property
    def n(self) -> int:
        return sum(self.terms)

    def sum_record(self) -> tuple[int, int, int]:
        n = self.n
        return (n, self.align_e, self.scale) if n else (0, 0, 0)

    def mx_record(self, is_float: bool) -> tuple[int, int, int]:
        n, e_loc, h = self.sum_record()
        if is_float:
            m_v, h_v = normalize_contribution(n, e_loc + h)
            return m_v, h_v, 0
        return 0, 0, n


def forward_groups(
    mode: Mode, mx: bool, a: Sequence[int], b: Sequence[int],
    scale_a: int = 0, scale_b: int = 0,
) -> tuple[LocalGroup, LocalGroup]:
    """Return both group slots, including the zero slot for G=1."""
    mode = Mode(mode)
    if mx and mode not in (Mode.FP4, Mode.FP8, Mode.INT8):
        raise ValueError(f"MX is undefined for {mode.name}")
    groups = []
    for group, (elements_a, elements_b) in enumerate(zip(decode_groups(mode, a),
                                                       decode_groups(mode, b), strict=True)):
        scale = 0
        if mx:
            sa, sb = ((word >> (8 * group)) & 255 for word in (scale_a, scale_b))
            if 255 in (sa, sb):
                raise ValueError(f"group {group}: invalid E8M0 scale")
            scale = (sa - 127) + (sb - 127)

        products = []
        for (ma, qa), (mb, qb) in zip(elements_a, elements_b, strict=True):
            p, q = ma * mb, qa + qb
            if mode == Mode.BF16:
                q, remainder = divmod(q, 4)
                p, q = (p * (2 ** remainder), q) if p else (0, 0)
            products.append((p, q))

        max_q = 0
        if mode in ALIGNMENT:
            f, bias = ALIGNMENT[mode]
            max_q = max((q for p, q in products if p), default=0)
            terms = tuple((p * (2 ** f)) // (2 ** (4 * (max_q - q))) if p else 0
                          for p, q in products)
            e_loc = 4 * max_q - bias - f
        else:
            terms = tuple(p for p, _ in products)
            e_loc = {Mode.FP4: -2, Mode.INT8: -12 if mx else 0}.get(mode, 0)
        groups.append(LocalGroup(tuple(products), max_q, terms, e_loc, scale))

    if len(groups) == 1:
        groups.append(LocalGroup((), 0, (), 0, 0))
    return groups[0], groups[1]
