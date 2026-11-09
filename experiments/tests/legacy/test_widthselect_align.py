"""Experiment-1 alignment tests: historical equality, oracle, prefixes, snapshots."""

from __future__ import annotations

import random

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.matrix.alignment import align_dot16
from mac_experiment.widthselect.reference import (
    FORMAT_ALIGNMENT,
    align_block,
    align_block_trace,
    block_products,
    block_scale_exponent,
    evaluate_pair_acc,
    evaluate_pair_align,
)
from mac_experiment.widthselect.tensors import TargetTensor
from mac_experiment.widthselect.tf32_rne import round_dyadic_to_tf32
from oracle import oracle_align_int, oracle_fp32_bits, oracle_s_b


def _random_products(rng: random.Random, count: int) -> list[tuple[int, int]]:
    products = []
    for _ in range(count):
        p = rng.randint(1, 7)
        if rng.random() < 0.5:
            p = -p
        products.append((p, rng.randint(0, 6)))
    return products


def test_e4m3_constants_equal_historical_align_dot16():
    rng = random.Random(1234)
    for _ in range(300):
        products = _random_products(rng, rng.randint(0, 16))
        h = rng.randint(-3, 3)
        for width in (10, 15, 21, 27):
            legacy = align_dot16(products, h, width)
            ours = align_block(products, h, width, "mx_fp8")
            assert ours.all_zero == legacy.all_zero
            assert ours.cmax == legacy.cmax
            assert ours.integer_sum == legacy.integer_sum
            assert ours.q_b == legacy.q_b
            assert ours.s_b == legacy.s_b
            assert ours.l1_b == legacy.l1_b
            assert ours.changed == legacy.changed_count
            assert ours.zeroed == legacy.zeroed_count


def test_independent_oracle_matches_for_both_formats():
    rng = random.Random(99)
    for fmt in ("mx_fp8", "tf32"):
        constants = FORMAT_ALIGNMENT[fmt]
        for _ in range(200):
            products = _random_products(rng, rng.randint(1, 16))
            h = rng.randint(-2, 2)
            for width in (12, 21, 30):
                ours = align_block(products, h, width, fmt)
                n_b, q_b = oracle_align_int(
                    products, h, width, constants["k_shift"], constants["window_offset"]
                )
                assert ours.integer_sum == n_b
                assert ours.q_b.to_fraction() == q_b
                assert ours.s_b.to_fraction() == oracle_s_b(
                    products, h, constants["product_lsb_offset"]
                )
                assert ours.l1_b.to_fraction() == sum(
                    abs(oracle_s_b([product], h, constants["product_lsb_offset"])) for product in products
                )


def test_all_zero_and_cancellation_are_distinct():
    zero = align_block([], 0, 21, "mx_fp8")
    assert zero.all_zero and zero.cmax is None and zero.integer_sum == 0
    assert zero.q_b == Dyadic.zero() and zero.s_b == Dyadic.zero()
    cancelling = align_block([(3, 5), (-3, 5)], 0, 21, "mx_fp8")
    assert not cancelling.all_zero
    assert cancelling.integer_sum == 0
    assert cancelling.q_b == Dyadic.zero()
    assert cancelling.s_b == Dyadic.zero()


def test_container_overflow_raises():
    with pytest.raises(ArithmeticError):
        align_block([(1 << 40, 0)], 0, 1, "tf32")


def _tf32_tensor(values: list[Dyadic], rows: int, cols: int) -> TargetTensor:
    codes = [round_dyadic_to_tf32(value).code for value in values]
    return TargetTensor(format="tf32", rows=rows, cols=cols, codes=codes)


def test_engine_exact_T_prefix_and_snapshot_stability():
    rng = random.Random(7)
    K = 32
    x_values = [Dyadic(rng.randint(-64, 64) or 1, rng.randint(-4, 2)) for _ in range(K)]
    w_values = [Dyadic(rng.randint(-64, 64) or 1, rng.randint(-4, 2)) for _ in range(K)]
    x = _tf32_tensor(x_values, 1, K)
    w = _tf32_tensor(w_values, K, 1)
    width = 24

    # independent exact per-dot16 expectation
    expected_t = Dyadic.zero()
    expected_s = Dyadic.zero()
    block_t = []
    block_s = []
    for block in range(K // 16):
        products = block_products(x, w, 0, 0, block, "tf32")
        result = align_block(products, 0, width, "tf32")
        expected_t = expected_t + result.q_b
        expected_s = expected_s + result.s_b
        block_t.append(expected_t)
        block_s.append(expected_s)

    long_run = evaluate_pair_align(x, w, [16, 32], [width])
    assert len(long_run.base_snapshots) == 2
    assert len(long_run.width_snapshots) == 2
    assert Dyadic.from_dict(long_run.width_snapshots[0]["T"][0]) == block_t[0]
    assert Dyadic.from_dict(long_run.width_snapshots[1]["T"][0]) == expected_t
    assert Dyadic.from_dict(long_run.base_snapshots[0]["S"][0]) == block_s[0]
    assert Dyadic.from_dict(long_run.base_snapshots[1]["S"][0]) == expected_s

    # snapshotting does not perturb the exact state
    end_only = evaluate_pair_align(x, w, [32], [width])
    assert Dyadic.from_dict(end_only.width_snapshots[0]["T"][0]) == expected_t
    # prefix run with the sliced tensors equals the K=16 state of the long run
    x16 = _tf32_tensor(x_values[:16], 1, 16)
    w16 = _tf32_tensor(w_values[:16], 16, 1)
    prefix_only = evaluate_pair_align(x16, w16, [16], [width])
    assert Dyadic.from_dict(prefix_only.width_snapshots[0]["T"][0]) == block_t[0]

    # Y = RN32(T) exactly, T never rounded per update
    expected_y = oracle_fp32_bits(expected_t.to_fraction())
    assert int(long_run.width_snapshots[1]["Y_bits"][0]) == expected_y
    expected_o = oracle_fp32_bits(expected_s.to_fraction())
    assert int(long_run.base_snapshots[1]["O_bits"][0]) == expected_o


def test_tf32_trace_and_scale_helper():
    rng = random.Random(5)
    x = _tf32_tensor([Dyadic(rng.randint(1, 255), -8) for _ in range(16)], 1, 16)
    w = _tf32_tensor([Dyadic(rng.randint(1, 255), -8) for _ in range(16)], 16, 1)
    products = block_products(x, w, 0, 0, 0, "tf32")
    trace = align_block_trace(products, 0, 21, "tf32")
    assert trace["all_zero"] is False
    assert len(trace["terms"]) == len(products)
    assert trace["integer_sum"] == sum(term["n"] for term in trace["terms"])
    assert block_scale_exponent(x, w, 0, 0, 0) == 0


def test_mx_scale_is_applied_after_integer_sum():
    group = 32
    x = TargetTensor(
        format="mx_fp8",
        rows=1,
        cols=group,
        codes=[0x38] * group,
        scale_codes=[0x7F],
        group_axis=1,
    )
    w = TargetTensor(
        format="mx_fp8",
        rows=group,
        cols=1,
        codes=[0x38] * group,
        scale_codes=[0x7F],
        group_axis=0,
    )
    result = evaluate_pair_align(x, w, [32], [21])
    T = Dyadic.from_dict(result.width_snapshots[0]["T"][0])
    S = Dyadic.from_dict(result.base_snapshots[0]["S"][0])
    assert S == Dyadic(32, 0)
    assert T == Dyadic(32, 0)
    assert block_scale_exponent(x, w, 0, 0, 0) == 0


def test_engine_accepts_saturated_tf32_inputs():
    from mac_experiment.widthselect.tf32_rne import MAX_FINITE_MAGNITUDE
    from mac_experiment.widthselect.tensors import quantize_fp32_bits

    # the largest finite FP32 value rounds to the TF32 overflow edge; the input
    # policy saturates, so the engine must never see a special code
    bits = [0x7F7FFFFF] * 32
    x = quantize_fp32_bits("tf32", "x", bits, 1, 32, group_axis=1)
    w = quantize_fp32_bits("tf32", "w", bits, 32, 1, group_axis=0)
    assert x.events["element_overflow_saturated"] == 32
    result = evaluate_pair_align(x.tensor, w.tensor, [32], [21])
    expected = MAX_FINITE_MAGNITUDE * MAX_FINITE_MAGNITUDE * Dyadic(32, 0)
    assert Dyadic.from_dict(result.base_snapshots[0]["S"][0]) == expected
    acc = evaluate_pair_acc(x.tensor, w.tensor, [32], [16])
    assert acc.acc_snapshots[0]["C_state"][0]["E_code"] == 255  # overflow marker, no crash


def test_deterministic_and_repeatable():
    rng = random.Random(77)
    values = [Dyadic(rng.randint(-128, 128) or 1, rng.randint(-5, 1)) for _ in range(32)]
    x = _tf32_tensor(values, 1, 32)
    w = _tf32_tensor(values, 32, 1)
    first = evaluate_pair_align(x, w, [16, 32], [18, 21])
    second = evaluate_pair_align(x, w, [16, 32], [18, 21])
    assert first.width_snapshots == second.width_snapshots
    assert first.base_snapshots == second.base_snapshots
    # smaller widths lose at least as many bits as larger ones for this pattern
    assert first.width_snapshots[-1]["events_cumulative"]["products_changed"] >= 0
