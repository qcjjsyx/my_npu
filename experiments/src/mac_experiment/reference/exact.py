"""Exact reference paths.

The reference side is organised independently from the candidate reducer: the
mathematical exact sum and the window-loss-free FP32 recurrence are computed
here without calling the candidate reducer.
"""

from __future__ import annotations

from typing import Iterable

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..arithmetic.rounding import round_dyadic_to_fp32


def exact_sum(values: Iterable[Dyadic]) -> Dyadic:
    total = Dyadic.zero()
    for value in values:
        total = total + value
    return total


def l1_sum(values: Iterable[Dyadic]) -> Dyadic:
    total = Dyadic.zero()
    for value in values:
        total = total + abs(value)
    return total


class ReferencePath:
    """FP32 recurrence ``r_b = RN32(exact_value(r_(b-1)) + s_b)``."""

    def __init__(self) -> None:
        self._state = Fp32.zero(0)

    @property
    def state(self) -> Fp32:
        return self._state

    def update(self, exact_block_sum: Dyadic) -> Fp32:
        if not self._state.is_finite():
            return self._state
        exact = self._state.exact_dyadic() + exact_block_sum
        self._state, _ = round_dyadic_to_fp32(exact)
        return self._state
