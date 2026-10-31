from fractions import Fraction

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.formats import get_format
from mac_experiment.reduction import BlockReducer, ProductTerm, WindowOverflowError, container_bounds

from oracle import oracle_align_to_window


def make_terms(fmt, samples):
    terms = []
    for sign_a, exp_a, frac_a, sign_b, exp_b, frac_b in samples:
        a = fmt.decode(fmt.encode(sign_a, exp_a, frac_a))
        b = fmt.decode(fmt.encode(sign_b, exp_b, frac_b))
        product = fmt.product(a, b)
        terms.append(ProductTerm(product.dyadic, product.c))
    return terms


def fp8_terms(rng, count=16, exp_range=(1, 15)):
    samples = []
    for _ in range(count):
        samples.append(
            (
                rng.randrange(2),
                rng.randrange(exp_range[0], exp_range[1] + 1),
                rng.randrange(8),
                rng.randrange(2),
                rng.randrange(exp_range[0], exp_range[1] + 1),
                rng.randrange(8),
            )
        )
    return samples


def test_reducer_matches_fraction_oracle_fp8():
    import random

    rng = random.Random(1)
    fmt = get_format("fp8_e4m3")
    reducer = BlockReducer(window_offset=fmt.window_offset)
    for w in (18, 24, 32, 41):
        for _ in range(30):
            terms = make_terms(fmt, fp8_terms(rng))
            reduction = reducer.reduce(terms, w)
            cmax = max(t.c for t in terms if not t.is_zero)
            window_lsb = cmax - 14 - w
            expected_total = sum(
                oracle_align_to_window(t.dyadic.to_fraction(), window_lsb) for t in terms if not t.is_zero
            )
            assert reduction.window_lsb == window_lsb
            assert reduction.aligned_sum.to_fraction() == Fraction(expected_total) * Fraction(2) ** window_lsb
            assert reduction.exact_sum.to_fraction() == sum(t.dyadic.to_fraction() for t in terms)


def test_reducer_matches_fraction_oracle_tf32():
    import random

    rng = random.Random(2)
    fmt = get_format("tf32")
    reducer = BlockReducer(window_offset=fmt.window_offset)
    for w in (18, 30, 41):
        for _ in range(20):
            samples = [
                (
                    rng.randrange(2),
                    rng.randrange(96, 159),
                    rng.randrange(1024),
                    rng.randrange(2),
                    rng.randrange(96, 159),
                    rng.randrange(1024),
                )
                for _ in range(16)
            ]
            terms = make_terms(fmt, samples)
            reduction = reducer.reduce(terms, w)
            cmax = max(t.c for t in terms if not t.is_zero)
            window_lsb = cmax - 254 - w
            expected_total = sum(
                oracle_align_to_window(t.dyadic.to_fraction(), window_lsb) for t in terms if not t.is_zero
            )
            assert reduction.aligned_sum.to_fraction() == Fraction(expected_total) * Fraction(2) ** window_lsb


def test_window_container_range_proof():
    for name, max_sig in (("fp8_e4m3", 15 * 15), ("tf32", 2047 * 2047)):
        fmt = get_format(name)
        reducer = BlockReducer(window_offset=fmt.window_offset)
        for w in range(18, 42):
            lo, hi = container_bounds(w)
            c = 500
            terms = [ProductTerm(Dyadic(max_sig, c - fmt.product_lsb_offset), c) for _ in range(16)]
            reduction = reducer.reduce(terms, w)
            if reduction.aligned_sum.is_zero():
                total = 0
            else:
                total = reduction.aligned_sum.m << (reduction.aligned_sum.e - reduction.window_lsb)
            assert lo <= total <= hi
            assert reduction.aligned_sum.to_fraction() <= Fraction(hi) * Fraction(2) ** reduction.window_lsb


def test_reducer_overflow_detection_raises():
    fmt = get_format("fp8_e4m3")
    reducer = BlockReducer(window_offset=fmt.window_offset)
    huge = 1 << 60
    terms = [ProductTerm(Dyadic(huge, -20), 0) for _ in range(16)]
    with pytest.raises(WindowOverflowError):
        reducer.reduce(terms, 18)


def test_reducer_zero_path_and_zeroed_terms():
    fmt = get_format("fp8_e4m3")
    reducer = BlockReducer(window_offset=fmt.window_offset)
    zero = fmt.decode(0)
    terms = [ProductTerm(fmt.product(zero, zero).dyadic, 0) for _ in range(16)]
    reduction = reducer.reduce(terms, 18)
    assert reduction.all_zero
    assert reduction.window_lsb is None
    assert reduction.aligned_sum.m == 0

    a = fmt.decode(fmt.encode(0, 1, 0))
    b = fmt.decode(fmt.encode(0, 1, 0))
    tiny = fmt.product(a, b)
    big_a = fmt.decode(fmt.encode(0, 15, 6))
    big_b = fmt.decode(fmt.encode(0, 15, 6))
    big = fmt.product(big_a, big_b)
    terms = [ProductTerm(big.dyadic, big.c)] + [ProductTerm(tiny.dyadic, tiny.c) for _ in range(15)]
    reduction = reducer.reduce(terms, 18)
    assert reduction.zeroed_terms == 15


def test_alignment_exact_flag():
    fmt = get_format("fp8_e4m3")
    reducer = BlockReducer(window_offset=fmt.window_offset)
    a = fmt.decode(fmt.encode(0, 15, 0))
    b = fmt.decode(fmt.encode(0, 15, 0))
    product = fmt.product(a, b)
    terms = [ProductTerm(product.dyadic, product.c) for _ in range(16)]
    assert reducer.reduce(terms, 18).alignment_exact
    low = fmt.decode(fmt.encode(0, 1, 0))
    mixed = terms + [ProductTerm(fmt.product(low, low).dyadic, fmt.product(low, low).c)]
    reduction = reducer.reduce(mixed, 18)
    assert not reduction.alignment_exact
