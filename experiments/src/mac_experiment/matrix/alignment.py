"""Fixed-W_align dot16 alignment for MXFP8 (explicit experiment semantics).

For one output element, one dot16 block (16 MXFP8 products) and a common E8M0
scale exponent ``h = alpha + beta`` (both input scales are constant over the
G32 group, and each dot16 lies inside one group):

* every finite E4M3 operand is decomposed without bias subtraction::

      E = max(raw_exp, 1)
      t = frac                if raw_exp == 0   (subnormal, t >= 1)
      t = 8 + frac            if raw_exp > 0

  so that ``u = (-1)**sign * t * 2**(E - 10)``; ``raw_exp == 0, frac == 0`` is
  zero and ``0x7F`` / ``0xFF`` are rejected;

* products are formed in the element domain::

      P_i = (-1)**(sign_x XOR sign_w) * t_x * t_w
      C_i = E_x + E_w

  with exact value ``v_i = P_i * 2**(C_i - 20)`` and exact scaled product
  ``v_i * 2**h``.  Bias terms cancel in ``C_i``, and ``h`` is not needed for
  the alignment decision because all 16 terms share it;

* ``Cmax = max(C_i)`` over the non-zero products.  With a fixed alignment
  fraction ``w_align`` the per-term aligned integer is::

      n_i = RNE_integer(P_i * 2**(w_align - 6 - (Cmax - C_i)))

  i.e. ``shift_i = (Cmax - C_i) + 6 - w_align``: a signed RNE ties-to-even
  right shift when ``shift_i > 0`` and an exact left shift otherwise;

* ``N_b = sum_i n_i`` is formed with exact integer addition, then the real
  partial sum is recovered as::

      q_b = N_b * 2**(Cmax - 14 - w_align + h)

  The ``14`` is the sum of the two E4M3 exponent biases and must not be
  dropped.  ``q_b`` is never rounded to FP32 here.

Aligned integers use a signed container of ``w_align + 7`` bits (28 bits at
``w_align = 21``); per-term and sum range violations raise instead of silently
wrapping, truncating or saturating.

Nothing in this module changes the legacy window no-window path in
``matrix/engine.py``; it is a separate candidate path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import rne_shift
from ..formats.e4m3_element import E4M3Element, is_nan_code
from ..version import MX_ALIGN_CONTAINER_EXTRA_BITS, MX_ALIGN_DEFAULT

ALIGNMENT_POLICY = "fixed_w_align_dot16_rne_integer_alignment"

_E4M3 = E4M3Element()


def decode_magnitude(code: int) -> tuple[int, int, int]:
    """``(sign, t, E)`` of a finite E4M3 code; zero returns ``(sign, 0, 0)``.

    ``t`` and ``E`` are the scale-free magnitude fields defined in the module
    docstring.  NaN codes raise; subnormals use ``E = max(raw_exp, 1)`` and are
    decoded exactly (no FTZ).
    """
    if not 0 <= code <= 0xFF:
        raise ValueError(f"E4M3 code out of range: {code}")
    if is_nan_code(code):
        raise ValueError("E4M3 NaN codes (0x7F/0xFF) have no magnitude")
    operand = _E4M3.decode(code)
    sign = 1 if operand.sign else 0
    if operand.is_zero:
        return sign, 0, 0
    if operand.raw_exp == 0:
        return sign, operand.frac, 1
    return sign, 8 + operand.frac, operand.raw_exp


def aligned_container_bounds(w_align: int) -> tuple[int, int]:
    """Signed range of the ``w_align + 7`` bit alignment container."""
    _validate_w_align(w_align)
    bits = w_align + MX_ALIGN_CONTAINER_EXTRA_BITS
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _validate_w_align(w_align: int) -> None:
    if not isinstance(w_align, int) or w_align < 1 or w_align > 60:
        raise ValueError(f"w_align must be an integer in [1, 60], got {w_align!r}")


def check_container(value: int, w_align: int, label: str) -> None:
    low, high = aligned_container_bounds(w_align)
    if not low <= value <= high:
        bits = w_align + MX_ALIGN_CONTAINER_EXTRA_BITS
        raise ArithmeticError(
            f"{label} {value} outside the signed {bits}-bit alignment container "
            f"[{low}, {high}] at w_align={w_align}; refusing to wrap/truncate/saturate"
        )


@dataclass
class AlignedTerm:
    p: int
    c: int
    shift: int
    n: int
    changed: bool
    zeroed: bool


@dataclass
class AlignedDot16:
    all_zero: bool
    cmax: int | None
    integer_sum: int
    q_b: Dyadic
    s_b: Dyadic
    l1_b: Dyadic
    terms: list[AlignedTerm] = field(default_factory=list)
    cmax_minus_ci: list[int] = field(default_factory=list)

    @property
    def changed_count(self) -> int:
        return sum(1 for term in self.terms if term.changed)

    @property
    def zeroed_count(self) -> int:
        return sum(1 for term in self.terms if term.zeroed)


def align_term(p: int, c: int, cmax: int, w_align: int) -> tuple[int, bool, bool]:
    """Align one non-zero product ``(p, c)``; returns ``(n, changed, zeroed)``.

    ``changed`` is exact: it compares the aligned integer against the exact
    target on its alignment grid and does not merely test ``shift > 0``.
    """
    if p == 0:
        raise ValueError("align_term requires a non-zero product")
    shift = (cmax - c) + 6 - w_align
    n = rne_shift(p, shift)
    if shift <= 0:
        changed = False  # exact left shift
    else:
        # right RNE shift; changed iff low bits were actually discarded
        changed = (n << shift) != p
    check_container(n, w_align, "aligned product")
    return n, changed, n == 0


def align_dot16(
    products: list[tuple[int, int]],
    h: int,
    w_align: int = MX_ALIGN_DEFAULT,
) -> AlignedDot16:
    """Align and sum one dot16.

    ``products`` are the non-zero ``(P_i, C_i)`` in any order; ``h`` is the
    common scale exponent (``alpha + beta``) used only to recover the real
    partial sum.  An empty list is the explicit all-zero dot16 path: ``q_b = 0``
    and ``cmax``/window exponent are ``None``.  A non-zero product set that
    cancels to ``N_b == 0`` is *not* an all-zero dot16.
    """
    _validate_w_align(w_align)
    if not products:
        return AlignedDot16(
            all_zero=True,
            cmax=None,
            integer_sum=0,
            q_b=Dyadic.zero(),
            s_b=Dyadic.zero(),
            l1_b=Dyadic.zero(),
        )
    cmax = max(c for _, c in products)
    terms: list[AlignedTerm] = []
    integer_sum = 0
    s_b = Dyadic.zero()
    l1_b = Dyadic.zero()
    for p, c in products:
        n, changed, zeroed = align_term(p, c, cmax, w_align)
        integer_sum += n
        terms.append(AlignedTerm(p=p, c=c, shift=(cmax - c) + 6 - w_align, n=n, changed=changed, zeroed=zeroed))
        exact_value = Dyadic(p, c - 20 + h)
        s_b = s_b + exact_value
        l1_b = l1_b + abs(exact_value)
    check_container(integer_sum, w_align, "aligned dot16 integer sum")
    q_b = Dyadic(integer_sum, cmax - 14 - w_align + h)
    return AlignedDot16(
        all_zero=False,
        cmax=cmax,
        integer_sum=integer_sum,
        q_b=q_b,
        s_b=s_b,
        l1_b=l1_b,
        terms=terms,
        cmax_minus_ci=[cmax - c for _, c in products],
    )


def recover_partial(integer_sum: int, cmax: int, h: int, w_align: int) -> Dyadic:
    """``q_b = N_b * 2**(Cmax - 14 - w_align + h)`` (exact, no FP32 step)."""
    _validate_w_align(w_align)
    return Dyadic(integer_sum, cmax - 14 - w_align + h)


def summarize_trace(dot16: AlignedDot16, w_align: int) -> dict:
    """Serializable per-dot16 trace (for the filtered diagnostic export)."""
    return {
        "all_zero": dot16.all_zero,
        "cmax": dot16.cmax,
        "qq": "q_b = N_b * 2**(Cmax - 14 - w_align + h)",
        "integer_sum": dot16.integer_sum,
        "counts": {
            "products_nonzero": len(dot16.terms),
            "changed": dot16.changed_count,
            "zeroed": dot16.zeroed_count,
        },
        "cmax_minus_ci_histogram": {str(k): v for k, v in sorted(Counter(dot16.cmax_minus_ci).items())},
        "terms": [
            {
                "P": term.p,
                "C": term.c,
                "shift": term.shift,
                "n": term.n,
                "changed": term.changed,
                "zeroed": term.zeroed,
            }
            for term in dot16.terms
        ],
        "s_b": dot16.s_b.to_dict(),
        "q_b": dot16.q_b.to_dict(),
        "q_b_equals_s_b": dot16.q_b == dot16.s_b,
    }
