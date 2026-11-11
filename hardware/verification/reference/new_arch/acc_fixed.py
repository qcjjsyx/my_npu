"""Structural fixed-point model of the new-architecture accumulator.

This model mirrors the intended RTL data path instead of reusing the exact
numerical reference: two's-complement alignment with floor, E=2 real low bits
plus one sticky bit, post-add normalization and RNE, signed re-normalization,
and the signed-11 exponent update.  The independent arbiter is ``oracle.py``;
this module must not import it.

Widths are derived from ``p`` so the same code runs at a small precision for
exhaustive checks:

    A, B       p + E + 1 bits   (43 at p=40)
    T          p + E + 2 bits   (44 at p=40)
    T'         p + E + 1 bits   (43 at p=40, -2^(p+2) included)
    M_pre      p + 1 bits
    M_r        p + 2 bits

State format: ``C = M * 2**(H - (p-1))`` with ``2**(p-1) <= abs(M) < 2**p``
for nonzero states and ``M = H = 0`` for zero.  ``M != -2**p`` is required by
the contract; the model raises instead of wrapping on any illegal value.
"""

from __future__ import annotations

E = 2


class AccFixedError(ArithmeticError):
    """Raised when the model is asked to compute outside the stated contract."""


def is_canonical(M: int, p: int) -> bool:
    return (1 << (p - 1)) <= abs(M) < (1 << p)


def check_state(M: int, H: int, p: int, what: str = "state") -> None:
    if M == 0:
        if H != 0:
            raise AccFixedError(f"{what}: zero coeff with nonzero H={H}")
        return
    if not is_canonical(M, p):
        raise AccFixedError(f"{what}: noncanonical M={M} at p={p}")
    if not -512 <= H <= 511:
        raise AccFixedError(f"{what}: H={H} outside signed10")


def normalize_contribution(N: int, e: int, p: int = 40) -> tuple[int, int]:
    """Exact re-encoding of ``v = N * 2**e`` as ``(M_v, H_v)``.

    Requires ``abs(N) < 2**p`` so the shift is a left shift and no bit is lost.
    """
    if N == 0:
        return 0, 0
    if not -(1 << p) < N < (1 << p):
        raise AccFixedError(f"contribution |N|={abs(N)} does not fit |N|<2^{p}")
    k = abs(N).bit_length() - 1
    shift = (p - 1) - k
    if shift < 0:
        raise AccFixedError(f"contribution needs a right shift (k={k})")
    M_v = N << shift
    if not is_canonical(M_v, p):
        raise AccFixedError(f"normalized contribution out of range: M_v={M_v}")
    return M_v, e + k


def _msb_position(T: int) -> int:
    """Position of the magnitude MSB via the sign-differing bit rule.

    For T >= 0 this is the highest set bit.  For T < 0 it is the highest zero
    bit of the two's-complement pattern, i.e. ``bit_length(~T) - 1``.  This is
    one lower than the magnitude MSB when abs(T) is a power of two; the signed
    re-normalization absorbs that case.
    """
    if T >= 0:
        return T.bit_length() - 1
    return (~T).bit_length() - 1


def update(M_old: int, H_old: int, M_new: int, H_new: int,
           p: int = 40, debug: dict | None = None) -> tuple[int, int]:
    """One state update ``C_next = RNE_p(C_old + v)``.

    ``(M_old, H_old)`` and ``(M_new, H_new)`` are canonical state/contribution
    pairs.  Zero state or zero contribution is bypassed, never aligned; the
    H=0 encoding of zero is not a numeric exponent.
    """
    if p < 3 or E >= p:
        raise AccFixedError(f"unsupported (p={p}, E={E})")

    if M_old == 0:
        check_state(M_old, H_old, p, "old state")
        if M_new == 0:
            check_state(M_new, H_new, p, "contribution")
            return 0, 0
        check_state(M_new, H_new, p, "contribution")
        return M_new, H_new
    if M_new == 0:
        check_state(M_new, H_new, p, "contribution")
        check_state(M_old, H_old, p, "old state")
        return M_old, H_old
    check_state(M_old, H_old, p, "old state")
    check_state(M_new, H_new, p, "contribution")

    if H_old >= H_new:
        H_a, a, b, d = H_old, M_old, M_new, H_old - H_new
    else:
        H_a, a, b, d = H_new, M_new, M_old, H_new - H_old
    if d > 1023:
        raise AccFixedError(f"exponent difference {d} outside signed11")

    A = a << E
    B = (b << E) >> d
    tau = ((b << E) - (B << d)) != 0
    T = A + B
    if abs(T) > (1 << (p + E + 1)) - (1 << (E + 1)):
        raise AccFixedError(f"|T|={abs(T)} overflows the {p + E + 2}-bit adder")
    if T == 0:
        # T == 0 is exactly the zero sum: d == 0 and a == -b, hence tau == 0.
        if tau:
            raise AccFixedError("zero sum with nonzero sticky (unreachable)")
        if debug is not None:
            debug.update(dict(H_a=H_a, d=d, A=A, B=B, tau=tau, T=T,
                              M=0, H=0))
        return 0, 0

    t = _msb_position(T)
    if t < 0:
        raise AccFixedError("zero sum reached the datapath; it must be bypassed")
    sigma = (p + 1) - t
    if not -1 <= sigma <= p + 1:
        raise AccFixedError(f"normalization shift sigma={sigma} out of range")
    if sigma >= 0:
        Tp = T << sigma
        dropped = 0
    else:
        Tp = T >> 1
        dropped = T & 1
    if not -(1 << (p + 2)) <= Tp <= (1 << (p + 2)) - 1:
        raise AccFixedError(f"T'={Tp} outside p+3 signed range")

    M_pre = Tp >> E
    G = (Tp >> (E - 1)) & 1
    R = Tp & 1
    S = tau or dropped
    inc = 1 if (G and (R or S or (M_pre & 1))) else 0
    M_r = M_pre + inc
    if not -(1 << p) <= M_r <= (1 << p):
        raise AccFixedError(f"M_r={M_r} outside renormalization range")

    c = 0
    if M_r == (1 << p) or M_r == -(1 << p):
        M_r >>= 1
        c = 1
    H_next = H_a - sigma + c
    if not -512 <= H_next <= 511:
        raise AccFixedError(f"H_next={H_next} outside signed10")

    if M_r == 0 or not is_canonical(M_r, p):
        raise AccFixedError(f"result M={M_r} not normalized")

    if debug is not None:
        debug.update(dict(H_a=H_a, d=d, A=A, B=B, tau=tau, T=T, t=t,
                          sigma=sigma, Tp=Tp, dropped=dropped,
                          M_pre=M_pre, G=G, R=R, S=S, inc=inc,
                          M_r0=M_pre + inc, c=c, M=M_r, H=H_next))
    return M_r, H_next
