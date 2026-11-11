"""Independent exact reference for one accumulator update.

The state is ``C = M * 2**(H - (p-1))``.  This module computes the exact sum of
the old state and the contribution with integers and then applies one RNE to
``p`` magnitude bits.  It is deliberately written without the fixed-point
alignment/normalization structure of ``acc_fixed`` so that a bug in one does
not hide in the other.
"""

from __future__ import annotations


def exact_update(M_old: int, H_old: int, M_new: int, H_new: int,
                 p: int = 40) -> tuple[int, int]:
    """Return ``(M, H)`` of ``RNE_p(C_old + v)`` or ``(0, 0)``."""
    if M_old == 0 and M_new == 0:
        return 0, 0
    Hm = min(H_old, H_new)
    X = (M_old << (H_old - Hm)) + (M_new << (H_new - Hm))
    if X == 0:
        return 0, 0
    negative = X < 0
    a = -X if negative else X
    H = Hm + (a.bit_length() - 1) - (p - 1)
    shift = a.bit_length() - p
    if shift > 0:
        q, r = divmod(a, 1 << shift)
        half = 1 << (shift - 1)
        if r > half or (r == half and (q & 1)):
            q += 1
    else:
        q = a << (-shift)
    if q == (1 << p):
        q >>= 1
        H += 1
    return (-q if negative else q), H
