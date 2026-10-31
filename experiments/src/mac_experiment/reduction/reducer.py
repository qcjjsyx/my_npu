"""Generic windowed reducer: 16 exact products -> one exact partial sum.

The reducer knows nothing about formats: it receives ``ProductTerm`` values
(exact dyadic + raw exponent sum ``C``) and a ``window_offset`` constant from
the format adapter.  All 16 terms of a block are aligned to a common finite
window LSB and summed with a wide signed integer container.  The returned
partial sum is exact: no FP32 (or other) rounding happens here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import rne_shift


@dataclass(frozen=True)
class ProductTerm:
    dyadic: Dyadic
    c: int

    @property
    def is_zero(self) -> bool:
        return self.dyadic.is_zero()


class WindowOverflowError(ArithmeticError):
    def __init__(self, total: int, lo: int, hi: int, w: int):
        super().__init__(
            f"aligned window sum {total} outside {w + 7}-bit signed range [{lo}, {hi}] for W={w}"
        )
        self.total = total
        self.lo = lo
        self.hi = hi
        self.w = w


@dataclass
class BlockReduction:
    exact_sum: Dyadic
    aligned_sum: Dyadic
    window_lsb: int | None
    cmax: int | None
    zeroed_terms: int = 0
    all_zero: bool = False
    alignment_exact: bool = True


class RNEAlignmentPolicy:
    """Signed round-to-nearest-ties-to-even alignment to the window LSB."""

    name = "rne_ties_even"

    def align_term(self, term: ProductTerm, window_lsb: int) -> tuple[int, bool]:
        if term.dyadic.is_zero():
            return 0, False
        shifted = rne_shift(term.dyadic.m, window_lsb - term.dyadic.e)
        return shifted, shifted == 0


@dataclass
class BlockReducer:
    window_offset: int
    alignment: RNEAlignmentPolicy = field(default_factory=RNEAlignmentPolicy)

    def reduce(self, terms: Sequence[ProductTerm], w: int) -> BlockReduction:
        nonzero = [t for t in terms if not t.is_zero]
        exact_sum = Dyadic.zero()
        for term in terms:
            exact_sum = exact_sum + term.dyadic
        if not nonzero:
            return BlockReduction(
                exact_sum=Dyadic.zero(),
                aligned_sum=Dyadic.zero(),
                window_lsb=None,
                cmax=None,
                zeroed_terms=0,
                all_zero=True,
                alignment_exact=True,
            )
        cmax = max(t.c for t in nonzero)
        window_lsb = cmax - self.window_offset - w
        total = 0
        zeroed = 0
        exact = True
        for term in terms:
            if term.is_zero:
                continue
            value, was_zeroed = self.alignment.align_term(term, window_lsb)
            total += value
            if was_zeroed:
                zeroed += 1
            if term.dyadic.e < window_lsb:
                exact = False
        lo, hi = container_bounds(w)
        if total < lo or total > hi:
            raise WindowOverflowError(total, lo, hi, w)
        return BlockReduction(
            exact_sum=exact_sum,
            aligned_sum=Dyadic(total, window_lsb),
            window_lsb=window_lsb,
            cmax=cmax,
            zeroed_terms=zeroed,
            all_zero=False,
            alignment_exact=exact,
        )


def container_bounds(w: int) -> tuple[int, int]:
    bits = w + 7
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
