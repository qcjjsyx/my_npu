"""Exact dyadic addition then one RNE: numerical reference, not fixed-GRS RTL."""
from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..local_alignment.model import align, family, output_fp32
from ..local_alignment.runner import blocks, ordered

FLOORS = {"mxfp8_e4m3fn": 16, "mxfp8_e5m2": 16,
          "mxfp8_e4m3fn_e5m2": 16, "mxfp8_e5m2_e4m3fn": 16,
          "fp16": 8, "bf16_postfold": 11}


@dataclass(frozen=True)
class State:
    p: int
    M: int = 0
    H: int = 0

    def __post_init__(self):
        if type(self.p) is not int or self.p < 2:
            raise ValueError("p must be an integer >=2")
        if not -512 <= self.H <= 511:
            raise OverflowError("state H outside signed10")
        if (not self.M and self.H) or (self.M and not (1 << (self.p-1) <= abs(self.M) < 1 << self.p)):
            raise ValueError("noncanonical state")

    @property
    def value(self):
        return Dyadic(self.M, self.H-(self.p-1))

    def to_dict(self):
        return dict(p=self.p, M=str(self.M), H=self.H)

    @classmethod
    def from_dict(cls, data):
        return cls(data["p"], int(data["M"]), data["H"])

    def update(self, contribution):
        exact = self.value + contribution
        if not exact.m:
            return State(self.p)
        a = abs(exact.m)
        shift = a.bit_length()-self.p
        if shift > 0:
            q, r = divmod(a, 1 << shift)
            half = 1 << (shift-1)
            a = q + int(r > half or (r == half and q & 1))
        else:
            a <<= -shift
        h = exact.e + shift + self.p-1
        if a == 1 << self.p:
            a >>= 1
            h += 1
        return State(self.p, -a if exact.m < 0 else a, h)


def prepare_blocks(a, b, mode, length):
    return [align(products, family(mode), FLOORS[mode], "floor", h)
            for products, h in blocks(a, b, mode, length)]


def evaluate(a, b, mode, length, p, local_blocks=None):
    local_blocks = prepare_blocks(a, b, mode, length) if local_blocks is None else local_blocks
    state = State(p)
    S = V = L1 = Dyadic.zero()
    trace = []
    absorbed = 0
    heights = []
    for block in local_blocks:
        old = state
        state = state.update(block.q)
        absorbed += bool(block.q.m and old == state)
        S, V, L1 = S+block.s, V+block.q, L1+block.l1
        if state.M:
            heights.append(state.H)
        trace.append(dict(contribution=block.q.to_dict(), **state.to_dict()))
    C = state.value
    yp, ps = output_fp32(C)
    yr, rs = output_fp32(S)
    yl, ls = output_fp32(V)
    result = dict(mode=mode, length=length, p_acc=p, F=FLOORS[mode],
                  S=S.to_dict(), V=V.to_dict(), C=C.to_dict(), L1=L1.to_dict(),
                  local_error=(V-S).to_dict(), state_error=(C-V).to_dict(), total_error=(C-S).to_dict(),
                  updates=len(local_blocks), absorbed=absorbed, H_min=min(heights, default=None),
                  H_max=max(heights, default=None), state=state.to_dict(), output_bits=yp.bits,
                  reference_bits=yr.bits, local_bits=yl.bits, output_saturated=ps,
                  reference_saturated=rs, local_saturated=ls, trace=trace,
                  cancellation_ratio=float(abs(S).to_fraction()/L1.to_fraction()) if L1.m else None)
    for label, ref, sat in (("reference", yr, rs), ("local", yl, ls)):
        result[label+"_comparison"] = dict(bitmatch=yp.bits == ref.bits,
            ulp=abs(ordered(yp.bits)-ordered(ref.bits)), saturated=ps or sat,
            signed_zero_mismatch=yp.is_zero() and ref.is_zero() and yp.sign != ref.sign)
    result["output_zero"] = yp.is_zero()
    result["output_subnormal"] = 0 < (yp.bits & 0x7fffffff) < 0x800000
    return result
