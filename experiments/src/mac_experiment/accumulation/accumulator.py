"""FP32 accumulator with a single correctly rounded RNE step per update.

The partial sum is exact (dyadic).  The update computes
``exact_value(state) + partial`` exactly and performs exactly one
round-to-nearest-ties-to-even conversion to FP32 with gradual underflow.
No intermediate or early FP32 rounding of the partial sum is performed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..arithmetic.rounding import round_dyadic_to_fp32


@dataclass
class UpdateOutcome:
    fp32: Fp32
    exact_sum: Dyadic
    rounded_value: Dyadic | None
    delta: Dyadic | None
    overflowed: bool
    absorbed: bool
    non_finite: bool


class Fp32Accumulator:
    """Running FP32 state updated from exact dyadic partial sums."""

    def __init__(self) -> None:
        self._state = Fp32.zero(0)

    @property
    def state(self) -> Fp32:
        return self._state

    def update(self, partial: Dyadic) -> UpdateOutcome:
        previous = self._state
        if not previous.is_finite():
            return UpdateOutcome(
                fp32=previous,
                exact_sum=Dyadic.zero(),
                rounded_value=None,
                delta=None,
                overflowed=False,
                absorbed=False,
                non_finite=True,
            )
        exact = previous.exact_dyadic() + partial
        new, rounded = round_dyadic_to_fp32(exact)
        self._state = new
        overflowed = new.is_inf()
        delta = None if rounded is None else rounded - exact
        absorbed = (not partial.is_zero()) and new.is_finite() and previous.is_finite() and new.same_value(previous)
        return UpdateOutcome(
            fp32=new,
            exact_sum=exact,
            rounded_value=rounded,
            delta=delta,
            overflowed=overflowed,
            absorbed=absorbed,
            non_finite=new.is_inf() or new.is_nan(),
        )
