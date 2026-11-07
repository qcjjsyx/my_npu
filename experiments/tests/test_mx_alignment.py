"""Numerical-semantics tests for the fixed-W_align alignment path.

The independent Fraction oracle below never calls ``matrix/alignment.py``; it
decodes E4M3 with its own bit arithmetic, rounds to the alignment grid with
ties-to-even, accumulates S/L1/T/A exactly and converts to FP32 with
``tests/oracle.oracle_fp32_bits``.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.formats.fp8_e4m3 import Fp8E4M3
from mac_experiment.matrix.aligned_engine import evaluate_mx_pair_aligned, trace_dot16
from mac_experiment.matrix.alignment import (
    align_dot16,
    align_term,
    aligned_container_bounds,
    check_container,
    decode_magnitude,
)
from mac_experiment.matrix.engine import MxMatrixEngine, evaluate_mx_pair
from mac_experiment.matrix.tensor import MxTensor

from oracle import oracle_fp32_bits, round_fraction_to_int

W = 21


# ---------------------------------------------------------------------------
# independent Fraction oracle
# ---------------------------------------------------------------------------


def oracle_element(code: int) -> Fraction:
    sign = -1 if (code >> 7) & 1 else 1
    raw_exp = (code >> 3) & 0xF
    frac = code & 0x7
    if raw_exp == 15 and frac == 7:
        raise ValueError("NaN code")
    if raw_exp == 0 and frac == 0:
        return Fraction(0)
    E = raw_exp if raw_exp > 0 else 1
    t = frac if raw_exp == 0 else 8 + frac
    return sign * t * Fraction(2) ** (E - 10)


def oracle_products(x_code: int, w_code: int) -> tuple[int, int] | None:
    """(P, C) of one product in the element domain, or None for zero."""
    x_val = oracle_element(x_code)
    w_val = oracle_element(w_code)
    if x_val == 0 or w_val == 0:
        return None
    def fields(code):
        raw_exp = (code >> 3) & 0xF
        frac = code & 0x7
        E = raw_exp if raw_exp > 0 else 1
        t = frac if raw_exp == 0 else 8 + frac
        sign = -1 if (code >> 7) & 1 else 1
        return sign, t, E
    sx, tx, ex = fields(x_code)
    sw, tw, ew = fields(w_code)
    return sx * sw * tx * tw, ex + ew


def oracle_align(products: list[tuple[int, int]], h: int, w_align: int) -> tuple[Fraction, Fraction, int | None, list[int]]:
    if not products:
        return Fraction(0), Fraction(0), None, []
    cmax = max(c for _, c in products)
    ns = [round_fraction_to_int(Fraction(p) * Fraction(2) ** (w_align - 6 - (cmax - c))) for p, c in products]
    q_b = Fraction(sum(ns)) * Fraction(2) ** (cmax - 14 - w_align + h)
    s_b = sum(Fraction(p) * Fraction(2) ** (c - 20 + h) for p, c in products)
    return q_b, s_b, cmax, ns


def oracle_pair(x_codes, w_codes, x_scales, w_scales, w_align, M, N, K, dot16=16, group=32):
    """S/L1/T and R/A as Fractions and FP32 bits, per output element."""
    grid = M * N
    S = [Fraction(0)] * grid
    L1 = [Fraction(0)] * grid
    T = [Fraction(0)] * grid
    r_values = [Fraction(0)] * grid
    a_values = [Fraction(0)] * grid
    r_bits = [0] * grid
    a_bits = [0] * grid
    for block in range(K // dot16):
        g = (block * dot16) // group
        for m in range(M):
            alpha = x_scales[m * (K // group) + g] - 127
            for n in range(N):
                beta = w_scales[g * N + n] - 127
                products = []
                for i in range(dot16):
                    k = block * dot16 + i
                    pair = oracle_products(x_codes[m * K + k], w_codes[k * N + n])
                    if pair is not None:
                        products.append(pair)
                q_b, s_b, _, _ = oracle_align(products, alpha + beta, w_align)
                index = m * N + n
                S[index] += s_b
                T[index] += q_b
                L1[index] += sum(abs(Fraction(p) * Fraction(2) ** (c - 20 + alpha + beta)) for p, c in products)
                r_bits[index] = oracle_fp32_bits(r_values[index] + s_b)
                r_values[index] = fraction_bits_value(r_bits[index])
                a_bits[index] = oracle_fp32_bits(a_values[index] + q_b)
                a_values[index] = fraction_bits_value(a_bits[index])
    return S, L1, T, r_bits, a_bits, r_values, a_values


def fraction_bits_value(bits: int) -> Fraction:
    return Fp32.from_bits(bits).exact_dyadic().to_fraction()


def fraction_to_dyadic(value: Fraction) -> Dyadic:
    if value == 0:
        return Dyadic.zero()
    denominator = value.denominator
    assert denominator & (denominator - 1) == 0
    return Dyadic(value.numerator, -(denominator.bit_length() - 1))


def make_tensor(name, rows, cols, axis, codes, scales):
    return MxTensor(name, (rows, cols), axis, list(codes), list(scales))


# ---------------------------------------------------------------------------
# 1-2: decoding and signed RNE
# ---------------------------------------------------------------------------


def test_decode_magnitude_normals_subnormals_zeros_nan():
    assert decode_magnitude(0x38) == (0, 8, 7)
    assert decode_magnitude(0xB8) == (1, 8, 7)
    assert decode_magnitude(0x01) == (0, 1, 1)
    assert decode_magnitude(0x07) == (0, 7, 1)
    assert decode_magnitude(0x08) == (0, 8, 1)
    assert decode_magnitude(0x00) == (0, 0, 0)
    assert decode_magnitude(0x80) == (1, 0, 0)
    for code in (0x7F, 0xFF):
        with pytest.raises(ValueError):
            decode_magnitude(code)


def test_signed_rne_shift_ties_to_even():
    # force shift = 1: (cmax - c) + 6 - 21 = 1 -> cmax = c + 16
    for p, expected in ((3, 2), (-3, -2), (5, 2), (-5, -2), (1, 0), (-1, 0)):
        n, changed, zeroed = align_term(p, 10, 26, 21)
        assert n == expected, (p, n)
        assert changed
    # exact even value: 4 / 2 = 2 without any lost bit
    n, changed, zeroed = align_term(4, 10, 26, 21)
    assert (n, changed) == (2, False)


# ---------------------------------------------------------------------------
# 3: 15/16 boundary, 4: zeroing, 5: cancellation, 6: all-zero distinction
# ---------------------------------------------------------------------------


def test_cmax_difference_boundary_15_and_16():
    # diff = 15 -> shift = 0 -> exact, never changed
    dot16 = align_dot16([(81, 14), (64, 29)], h=0, w_align=21)
    assert dot16.cmax == 29
    assert [t.shift for t in dot16.terms] == [0, -15]
    assert [t.changed for t in dot16.terms] == [False, False]
    # diff = 16 -> shift = 1: odd P loses a bit, even P stays exact
    dot16 = align_dot16([(81, 14), (64, 30)], h=0, w_align=21)
    assert [t.shift for t in dot16.terms] == [1, -15]
    assert [t.changed for t in dot16.terms] == [True, False]
    assert dot16.terms[0].n == 40
    oracle_q, oracle_s, _, _ = oracle_align([(81, 14), (64, 30)], 0, 21)
    assert dot16.q_b.to_fraction() == oracle_q
    assert dot16.q_b != dot16.s_b


def test_single_product_aligned_to_zero_and_cancelling_terms():
    dot16 = align_dot16([(1, 0), (1, 16)], h=0, w_align=21)
    zeroed_terms = [t for t in dot16.terms if t.c == 0]
    assert zeroed_terms[0].n == 0 and zeroed_terms[0].zeroed
    assert dot16.q_b != dot16.s_b

    # two changed terms cancel exactly: q_b == s_b while both products changed
    cancel = align_dot16([(1, 26), (3, 10), (-3, 10)], h=0, w_align=21)
    changed_terms = [t for t in cancel.terms if t.c in (10,)]
    assert [t.changed for t in changed_terms] == [True, True]
    assert cancel.q_b == cancel.s_b


def test_all_zero_dot16_is_not_nonzero_cancellation():
    empty = align_dot16([], h=5, w_align=21)
    assert empty.all_zero and empty.cmax is None and empty.q_b == Dyadic.zero()
    cancelling = align_dot16([(3, 10), (-3, 10)], h=5, w_align=21)
    assert not cancelling.all_zero
    assert cancelling.cmax == 10
    assert cancelling.integer_sum == 0
    assert cancelling.q_b == Dyadic.zero() == cancelling.s_b


# ---------------------------------------------------------------------------
# 7: container bounds
# ---------------------------------------------------------------------------


def test_container_bounds_and_overflow_guard():
    low, high = aligned_container_bounds(21)
    assert (low, high) == (-(2**27), 2**27 - 1)
    check_container(high, 21, "test")
    check_container(low, 21, "test")
    with pytest.raises(ArithmeticError):
        check_container(high + 1, 21, "test")
    with pytest.raises(ArithmeticError):
        check_container(low - 1, 21, "test")
    with pytest.raises(ValueError):
        aligned_container_bounds(0)


# ---------------------------------------------------------------------------
# 8: element-domain vs exact scaled-domain reduction
# ---------------------------------------------------------------------------


def test_element_domain_matches_exact_scaled_domain_oracle():
    rng = random.Random(20260922)
    code_pool = [0x00, 0x01, 0x02, 0x07, 0x08, 0x0F, 0x10, 0x38, 0x39, 0x70, 0x78, 0xB9, 0xF0]
    for _ in range(200):
        terms = []
        while not terms:
            terms = []
            for _ in range(rng.randint(1, 16)):
                xc = rng.choice(code_pool)
                wc = rng.choice(code_pool)
                pair = oracle_products(xc, wc)
                if pair is not None:
                    terms.append(pair)
        h = rng.randint(-40, 40)
        dot16 = align_dot16(terms, h=h, w_align=W)
        oracle_q, oracle_s, cmax, ns = oracle_align(terms, h=h, w_align=W)
        assert dot16.cmax == cmax
        assert [t.n for t in dot16.terms] == ns
        assert dot16.q_b.to_fraction() == oracle_q
        assert dot16.s_b.to_fraction() == oracle_s


# ---------------------------------------------------------------------------
# 9: large W_align == baseline no-window path; 10: subnormals not FTZ
# ---------------------------------------------------------------------------


def test_large_w_align_equals_no_window_baseline():
    x = make_tensor("x", 2, 64, 1, [0x38] * 128, [127, 130, 127, 130])
    w = make_tensor("w", 64, 2, 0, [0x39] * 128, [127] * 4)
    baseline = evaluate_mx_pair(x, w, [32, 64])
    aligned = evaluate_mx_pair_aligned(x, w, [32, 64], w_align=40)
    for base, cand in zip(baseline.snapshots, aligned.snapshots):
        assert base["S"] == cand["S"]
        assert base["L1"] == cand["L1"]
        assert cand["T"] == cand["S"]
        assert base["r_bits"] == cand["a_bits"]
        assert cand["events_cumulative"]["dot16_qb_diff"] == 0

    # scale = 1 (all E8M0 codes 127): exact alignment equals the legacy path
    x_unit = make_tensor("x", 2, 64, 1, [0x38] * 128, [127] * 4)
    w_unit = make_tensor("w", 64, 2, 0, [0x39] * 128, [127] * 4)
    base_unit = evaluate_mx_pair(x_unit, w_unit, [64])
    cand_unit = evaluate_mx_pair_aligned(x_unit, w_unit, [64], w_align=40)
    assert base_unit.snapshots[0]["S"] == cand_unit.snapshots[0]["S"]
    assert base_unit.snapshots[0]["r_bits"] == cand_unit.snapshots[0]["a_bits"]


def test_mx_subnormal_input_does_not_inherit_ftz():
    # legacy input adapter FTZes raw_exp == 0; the alignment path must not
    x = make_tensor("x", 1, 32, 1, [0x01] + [0x00] * 31, [127])
    w = make_tensor("w", 32, 1, 0, [0x38] + [0x00] * 31, [127])
    assert decode_magnitude(0x01) == (0, 1, 1)
    result = evaluate_mx_pair_aligned(x, w, [32], w_align=W)
    assert Dyadic.from_dict(result.snapshots[0]["S"][0]) == Dyadic(1, -9)
    legacy = Fp8E4M3()
    assert legacy.operand_value(legacy.decode(0x01)).is_zero()  # old FTZ semantics
    baseline = evaluate_mx_pair(x, w, [32])
    assert Dyadic.from_dict(baseline.snapshots[0]["S"][0]) == Dyadic(1, -9)


# ---------------------------------------------------------------------------
# 11: separate Cmax per dot16 inside one G32
# ---------------------------------------------------------------------------


def test_each_dot16_selects_its_own_cmax_within_a_g32():
    x_codes = [0x00] * 32
    w_codes = [0x00] * 32
    x_codes[0] = 0x78  # E=15, t=8
    w_codes[0] = 0x78
    x_codes[16] = 0x39  # E=7, t=9
    w_codes[16] = 0x39
    x = make_tensor("x", 1, 32, 1, x_codes, [127])
    w = make_tensor("w", 32, 1, 0, w_codes, [127])
    block0 = trace_dot16(x, w, 0, 0, 0)
    block1 = trace_dot16(x, w, 0, 0, 1)
    assert block0["cmax"] == 30
    assert block1["cmax"] == 14
    assert block1["terms"][0]["shift"] == -15  # own Cmax, exact left shift
    assert block1["terms"][0]["changed"] is False
    # a wrong shared cmax (block0's) would right-shift block1 and change it
    n_wrong, changed_wrong, _ = align_term(81, 14, cmax=30, w_align=21)
    assert changed_wrong and n_wrong != 81 << 15


# ---------------------------------------------------------------------------
# 12: q_b must not be pre-rounded to FP32 (subnormal double-rounding sample)
# ---------------------------------------------------------------------------


def test_qb_is_not_pre_rounded_to_fp32():
    x_codes = [0x00] * 32
    w_codes = [0x00] * 32
    x_codes[0] = x_codes[1] = x_codes[16] = 0x01  # t=1, E=1
    x_codes[17] = 0x02  # t=2, E=1
    w_codes[0] = w_codes[1] = w_codes[16] = w_codes[17] = 0x01
    x = make_tensor("x", 1, 32, 1, x_codes, [61])  # exponent -66
    w = make_tensor("w", 32, 1, 0, w_codes, [61])
    block0 = trace_dot16(x, w, 0, 0, 0)
    block1 = trace_dot16(x, w, 0, 0, 1)
    assert block0["q_b"] == Dyadic(1, -149).to_dict()
    assert block1["q_b"] == Dyadic(3, -150).to_dict()
    result = evaluate_mx_pair_aligned(x, w, [32], w_align=W)
    a_bits = result.snapshots[0]["a_bits"][0]
    assert a_bits == 0x00000002  # exact q1: RN(2.5 ulp) ties to even
    # wrong double rounding of q1 to FP32 first would give 3 * 2**-149
    pre_rounded = oracle_fp32_bits(Fraction(3, 2**150))
    state = Fp32.from_bits(0x00000001).exact_dyadic().to_fraction()
    wrong = oracle_fp32_bits(state + fraction_bits_value(pre_rounded))
    assert wrong == 0x00000003
    assert a_bits != wrong


# ---------------------------------------------------------------------------
# 13 (covered by test_large_w_align_equals_no_window_baseline)
# 14: identity and delta sum with the independent oracle
# 16: paired-prefix
# 20: deterministic W=21 loss
# ---------------------------------------------------------------------------


def test_oracle_row_matches_engine_and_identity():
    rng = random.Random(7)
    code_pool = [0x00, 0x01, 0x02, 0x03, 0x07, 0x08, 0x10, 0x38, 0x39, 0x48, 0x70, 0xB9]
    x_codes = [rng.choice(code_pool) for _ in range(32)]
    w_codes = [rng.choice(code_pool) for _ in range(32)]
    x_scales = [127]
    w_scales = [127]
    x = make_tensor("x", 1, 32, 1, x_codes, x_scales)
    w = make_tensor("w", 32, 1, 0, w_codes, w_scales)
    aligned = evaluate_mx_pair_aligned(x, w, [32], w_align=W).snapshots[-1]
    baseline = evaluate_mx_pair(x, w, [32]).snapshots[-1]
    S, L1, T, r_bits, a_bits, r_values, a_values = oracle_pair(
        x_codes, w_codes, x_scales, w_scales, W, 1, 1, 32
    )
    assert Dyadic.from_dict(aligned["S"][0]).to_fraction() == S[0]
    assert Dyadic.from_dict(aligned["L1"][0]).to_fraction() == L1[0]
    assert Dyadic.from_dict(aligned["T"][0]).to_fraction() == T[0]
    assert aligned["r_bits"] == r_bits == baseline["r_bits"]
    assert aligned["a_bits"] == a_bits
    # identity per element
    lhs = a_values[0] - S[0]
    rhs = (T[0] - S[0]) + (a_values[0] - T[0])
    assert lhs == rhs
    # delta sum equals A - T: emulate per-update deltas with the oracle
    delta_sum = Fraction(0)
    state = Fraction(0)
    for block in range(2):
        products = []
        for i in range(16):
            pair = oracle_products(x_codes[block * 16 + i], w_codes[block * 16 + i])
            if pair is not None:
                products.append(pair)
        q_b, _, _, _ = oracle_align(products, 0, W)
        new_bits = oracle_fp32_bits(state + q_b)
        new_value = fraction_bits_value(new_bits)
        delta_sum += new_value - (state + q_b)
        state = new_value
    assert delta_sum == a_values[0] - T[0]


def test_paired_prefix_matches_standalone_short_run():
    short_engine = MxMatrixEngine(1, 32, [32], "gaussian", "gaussian")
    long_engine = MxMatrixEngine(1, 32, [32, 64], "gaussian", "gaussian")
    short = short_engine.build_sample(5, 0)
    long = long_engine.build_sample(5, 0)
    short_result = evaluate_mx_pair_aligned(short.x, short.w, [32], w_align=W)
    long_result = evaluate_mx_pair_aligned(long.x, long.w, [32, 64], w_align=W)
    first = short_result.snapshots[0]
    other = long_result.snapshots[0]
    for key in ("S", "L1", "T", "r_bits", "a_bits", "events_cumulative", "cmax_minus_ci_histogram_cumulative"):
        assert first[key] == other[key]


def test_deterministic_w21_loss_at_boundary():
    x_codes = [0x00] * 32
    w_codes = [0x00] * 32
    x_codes[0] = 0x78  # C=30 with w_code 0x78
    w_codes[0] = 0x78
    x_codes[1] = 0x39  # C=14 with w_code 0x39
    w_codes[1] = 0x39
    x = make_tensor("x", 1, 32, 1, x_codes, [127])
    w = make_tensor("w", 32, 1, 0, w_codes, [127])
    result = evaluate_mx_pair_aligned(x, w, [32], w_align=W).snapshots[0]
    assert result["events_cumulative"]["dot16_qb_diff"] == 1
    assert result["events_cumulative"]["products_changed"] == 1
    assert Dyadic.from_dict(result["T"][0]) != Dyadic.from_dict(result["S"][0])
