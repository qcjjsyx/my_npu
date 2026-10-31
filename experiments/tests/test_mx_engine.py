"""MX matrix engine tests: grouping, dot16 semantics, exact oracle, prefixes."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mac_experiment.accumulation.accumulator import Fp32Accumulator
from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.formats.e4m3_element import decode_element
from mac_experiment.matrix.engine import MxMatrixEngine, evaluate_mx_pair
from mac_experiment.matrix.tensor import MxTensor
from mac_experiment.reference.exact import ReferencePath

from oracle import oracle_fp32_bits

X_CODE_1 = 0x38  # 1.0
ZERO = 0x00


def tensor(name, shape, group_axis, element_codes, scale_codes):
    return MxTensor(name, shape, group_axis, element_codes, list(scale_codes))


def dyadic_from_fraction(value: Fraction) -> Dyadic:
    if value == 0:
        return Dyadic.zero()
    denominator = value.denominator
    assert denominator & (denominator - 1) == 0, "not a dyadic"
    return Dyadic(value.numerator, -(denominator.bit_length() - 1))


def decoded_scaled_dyadic(matrix: MxTensor, row: int, col: int) -> Dyadic:
    element = decode_element(matrix.element_code(row, col))
    if element.is_zero():
        return element
    return Dyadic(element.m, element.e + matrix.scale_exponent(row, col))


def oracle_exact(x: MxTensor, w: MxTensor):
    M, N = x.rows, w.cols
    K = x.cols
    sums = [[Fraction(0)] * N for _ in range(M)]
    l1 = [[Fraction(0)] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            total = Fraction(0)
            total_l1 = Fraction(0)
            for k in range(K):
                product = decoded_scaled_dyadic(x, m, k).to_fraction() * decoded_scaled_dyadic(w, k, n).to_fraction()
                total += product
                total_l1 += abs(product)
            sums[m][n] = total
            l1[m][n] = total_l1
    return sums, l1


def oracle_fp32(x: MxTensor, w: MxTensor, k_value: int):
    """Independent dot16 FP32 recurrence on exact products."""
    M, N = x.rows, w.cols
    state = [Fraction(0)] * (M * N)
    bits = [0] * (M * N)
    for block_index in range(k_value // 16):
        for m in range(M):
            for n in range(N):
                total = Fraction(0)
                for i in range(16):
                    k = block_index * 16 + i
                    total += decoded_scaled_dyadic(x, m, k).to_fraction() * decoded_scaled_dyadic(w, k, n).to_fraction()
                index = m * N + n
                exact = state[index] + total
                new_bits = oracle_fp32_bits(exact)
                bits[index] = new_bits
                from mac_experiment.arithmetic.fp32 import Fp32

                state[index] = Fp32.from_bits(new_bits).exact_dyadic().to_fraction()
    return bits


def uniform_tensors(M, N, K, scales_x=None, scales_w=None, x_code=X_CODE_1, w_code=X_CODE_1):
    x_codes = [x_code] * (M * K)
    w_codes = [w_code] * (K * N)
    if scales_x is None:
        scales_x = [127] * (M * (K // 32))
    if scales_w is None:
        scales_w = [127] * ((K // 32) * N)
    return tensor("x", (M, K), 1, x_codes, scales_x), tensor("w", (K, N), 0, w_codes, scales_w)


def test_exact_reference_matches_fraction_oracle():
    M, N, K = 2, 3, 64
    x = tensor("x", (M, K), 1, [X_CODE_1] * (M * K), [127, 130, 127, 130])
    w = tensor("w", (K, N), 0, [X_CODE_1] * (K * N), [127] * N + [133] * N)
    result = evaluate_mx_pair(x, w, [32, 64])
    record = result.snapshots[-1]
    sums, l1 = oracle_exact(x, w)
    for index, payload in enumerate(record["S"]):
        m, n = divmod(index, N)
        assert Dyadic.from_dict(payload).to_fraction() == sums[m][n]
    for index, payload in enumerate(record["L1"]):
        m, n = divmod(index, N)
        assert Dyadic.from_dict(payload).to_fraction() == l1[m][n]


def test_group_scale_applies_to_both_dot16_and_across_groups():
    # K=64: group 0 elements 1.0 at scale 2**0, group 1 at scale 2**3.
    x = tensor("x", (1, 64), 1, [X_CODE_1] * 64, [127, 130])
    w = tensor("w", (64, 1), 0, [X_CODE_1] * 64, [127, 127])
    result = evaluate_mx_pair(x, w, [64])
    value = Dyadic.from_dict(result.snapshots[0]["S"][0])
    assert value == Dyadic(288, 0)  # 32 * 1 + 32 * 8

    # Second dot16 of a group must use the group scale, not "no scale".
    second_half_only = [ZERO] * 16 + [X_CODE_1] * 16
    x2 = tensor("x", (1, 32), 1, second_half_only, [130])
    w2 = tensor("w", (32, 1), 0, [X_CODE_1] * 32, [127])
    result2 = evaluate_mx_pair(x2, w2, [32])
    assert Dyadic.from_dict(result2.snapshots[0]["S"][0]) == Dyadic(128, 0)  # 16 * 8


def test_dot16_separate_updates_distinguish_wrong_dot32_merge():
    """E4M3-level counterexample: dot16 #1 = 0.5 + 2**-25, dot16 #2 = 2**-25.

    Correct: RN(0.5 + 2**-25) = 0.5 (tie to even), then RN(0.5 + 2**-25) = 0.5.
    Wrong merge: RN(0.5 + 2**-24) = 0.5 + 2**-24.
    """
    x_codes = [ZERO] * 32
    w_codes = [ZERO] * 32
    x_codes[0] = 0x68  # exp 13, sig 8
    w_codes[0] = 0x70  # exp 14, sig 8 -> 0.5 with scale 2**-7
    x_codes[1] = 0x08  # exp 1
    w_codes[1] = 0x10  # exp 2 -> 2**-25 with scale 2**-7
    x_codes[16] = 0x08
    w_codes[16] = 0x10
    x = tensor("x", (1, 32), 1, x_codes, [120])  # exponent -7
    w = tensor("w", (32, 1), 0, w_codes, [120])
    log: list = []
    result = evaluate_mx_pair(x, w, [32], update_log=log)
    assert len(log) == 2
    assert Dyadic.from_dict(result.snapshots[0]["S"][0]) == Dyadic(1, -1) + Dyadic(1, -24)
    assert result.snapshots[0]["r_bits"][0] == 0x3F000000  # 0.5

    merged = Fp32Accumulator()
    merged.update(log[0] + log[1])
    assert merged.state.bits == 0x3F000001  # 0.5 + 2**-24, the wrong dot32 result
    two_step = Fp32Accumulator()
    two_step.update(log[0])
    two_step.update(log[1])
    assert two_step.state.bits == result.snapshots[0]["r_bits"][0]


def test_update_log_replay_matches_every_element():
    M, N, K = 2, 2, 64
    x, w = uniform_tensors(M, N, K, scales_x=[127, 130, 127, 130], scales_w=[127] * ((K // 32) * N))
    log: list = []
    result = evaluate_mx_pair(x, w, [32, 64], update_log=log)
    grid = M * N
    assert len(log) == (K // 16) * grid
    for snapshot in result.snapshots:
        blocks = snapshot["K"]
        for index in range(grid):
            acc = Fp32Accumulator()
            for block in range(blocks // 16):
                acc.update(log[block * grid + index])
            assert acc.state.bits == snapshot["r_bits"][index]


def test_scale_one_matches_existing_reference_path():
    M, N, K = 2, 1, 96
    x = tensor("x", (M, K), 1, [X_CODE_1] * (M * K), [127] * (M * (K // 32)))
    w = tensor("w", (K, N), 0, [X_CODE_1] * (K * N), [127] * (K // 32))
    result = evaluate_mx_pair(x, w, [32, 64, 96])
    per_block = []
    for block in range(K // 16):
        block_sum = Dyadic.zero()
        for k in range(block * 16, block * 16 + 16):
            block_sum = block_sum + decoded_scaled_dyadic(x, 0, k) * decoded_scaled_dyadic(w, k, 0)
        per_block.append(block_sum)
    for snapshot in result.snapshots:
        paths = [ReferencePath(), ReferencePath()]
        for block in range(snapshot["K"] // 16):
            for path in paths:
                path.update(per_block[block])
        for index, path in enumerate(paths):
            assert snapshot["r_bits"][index] == path.state.bits


def test_fp32_path_matches_fraction_oracle():
    M, N, K = 2, 3, 64
    x = tensor("x", (M, K), 1, [0x41] * (M * K), [129, 126, 129, 126])
    w = tensor("w", (K, N), 0, [0x39] * (K * N), [127] * N + [131] * N)
    result = evaluate_mx_pair(x, w, [32, 64])
    expected = oracle_fp32(x, w, 64)
    assert result.snapshots[-1]["r_bits"] == expected


def test_paired_prefix_equals_separate_k_run():
    M, N, K = 1, 1, 64
    x, w = uniform_tensors(M, N, K, scales_x=[129, 126], scales_w=[127, 131])
    long_result = evaluate_mx_pair(x, w, [32, 64])
    x_prefix = tensor("x", (M, 32), 1, x.element_codes[:32], x.scale_codes[: (M * 32 // 32)])
    w_prefix = tensor("w", (32, N), 0, [w.element_code(k, 0) for k in range(32)], [w.scale_code(0, 0)])
    short_result = evaluate_mx_pair(x_prefix, w_prefix, [32])
    assert long_result.snapshots[0] == short_result.snapshots[0]


def test_engine_rejects_non_multiple_k_and_nan_codes():
    with pytest.raises(ValueError):
        MxMatrixEngine(2, 32, [40], "gaussian", "gaussian")
    engine = MxMatrixEngine(1, 32, [32], "gaussian", "gaussian")
    sample = engine.build_sample(1, 0)
    sample.x.element_codes[0] = 0x7F
    with pytest.raises(ValueError):
        engine.evaluate(sample)


def test_tensor_validation_rejects_incomplete_groups():
    with pytest.raises(ValueError):
        MxTensor("x", (1, 31), 1, [0] * 31, [127])
    with pytest.raises(ValueError):
        MxTensor("x", (1, 32), 1, [0] * 32, [0xFF])
    with pytest.raises(ValueError):
        MxTensor("x", (1, 32), 2, [0] * 32, [127])


def test_build_sample_pairs_reproducible_distinct_and_fp32_prefix_stable():
    engine = MxMatrixEngine(2, 32, [32, 64], "gaussian", "laplace")
    first = engine.build_sample(seed=3, pair_index=0)
    again = engine.build_sample(seed=3, pair_index=0)
    assert first.x_fp32_bits == again.x_fp32_bits
    assert first.w_fp32_bits == again.w_fp32_bits
    assert first.x.element_codes == again.x.element_codes
    assert first.x.scale_codes == again.x.scale_codes

    other = engine.build_sample(seed=3, pair_index=1)
    assert first.x_fp32_bits != other.x_fp32_bits
    assert first.w_fp32_bits != other.w_fp32_bits
    assert first.x.element_codes != other.x.element_codes
    assert first.w.element_codes != other.w.element_codes
    assert first.sample_id != other.sample_id

    # a short-K run is the prefix of the long-K run at every level
    short_engine = MxMatrixEngine(2, 32, [32], "gaussian", "laplace")
    long_engine = MxMatrixEngine(2, 32, [32, 64], "gaussian", "laplace")
    short = short_engine.build_sample(seed=3, pair_index=0)
    long = long_engine.build_sample(seed=3, pair_index=0)
    for row in range(short.x.rows):
        assert short.x_fp32_bits[row * 32 : (row + 1) * 32] == long.x_fp32_bits[
            row * 64 : row * 64 + 32
        ]
        assert short.x.element_codes[row * 32 : (row + 1) * 32] == long.x.element_codes[
            row * 64 : row * 64 + 32
        ]
        assert short.x.scale_codes[row] == long.x.scale_codes[2 * row]
    assert short.w_fp32_bits == long.w_fp32_bits[: 32 * short.w.cols]
    assert short.w.element_codes == long.w.element_codes[: 32 * short.w.cols]
    assert short.w.scale_codes == long.w.scale_codes[: short.w.cols]

    # the K=32 snapshot from the long scan equals the standalone short run
    long_result = long_engine.evaluate(long)
    short_result = short_engine.evaluate(short)
    assert long_result.snapshots[0] == short_result.snapshots[0]


def test_build_sample_distributions_are_independent_and_order_independent():
    engine = MxMatrixEngine(1, 32, [32], "gaussian", "outlier")
    baseline = engine.build_sample(seed=8, pair_index=1)
    # changing the weight distribution cannot change the activation stream
    other_engine = MxMatrixEngine(1, 32, [32], "gaussian", "laplace")
    other = other_engine.build_sample(seed=8, pair_index=1)
    assert other.x_fp32_bits == baseline.x_fp32_bits
    assert other.x.element_codes == baseline.x.element_codes
    assert other.w_fp32_bits != baseline.w_fp32_bits

    for pair_index in (0, 9, 2, 4):
        engine.build_sample(seed=8, pair_index=pair_index)
    repeated = engine.build_sample(seed=8, pair_index=1)
    assert (baseline.x_fp32_bits, baseline.w_fp32_bits) == (
        repeated.x_fp32_bits,
        repeated.w_fp32_bits,
    )
    assert (baseline.x.element_codes, baseline.w.element_codes) == (
        repeated.x.element_codes,
        repeated.w.element_codes,
    )


def test_exact_reference_uses_decoded_mx_values_not_raw_fp32():
    """1.1 quantises to 1.125; S must use 1.125, not the raw FP32 1.1."""
    from mac_experiment.matrix.inputs import fp32_bits_from_float, quantize_fp32_bits_to_mx

    x_bits = [fp32_bits_from_float(1.1)] * 32
    w_bits = [fp32_bits_from_float(1.0)] * 32
    x = quantize_fp32_bits_to_mx("x", x_bits, 1, 32, group_axis=1).tensor
    w = quantize_fp32_bits_to_mx("w", w_bits, 32, 1, group_axis=0).tensor
    result = evaluate_mx_pair(x, w, [32])
    s_value = Dyadic.from_dict(result.snapshots[0]["S"][0])
    assert s_value == Dyadic(9 * 32, -3)  # 32 * 1.125
    raw = Dyadic(11 * 32, -4)  # 32 * 1.1
    assert s_value != raw
