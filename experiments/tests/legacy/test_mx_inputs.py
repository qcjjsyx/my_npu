"""Numeric-domain FP32 input sampling and G32 MXFP8 input quantisation tests."""

from __future__ import annotations

import math

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.formats.e4m3_element import decode_element
from mac_experiment.matrix.engine import MxMatrixEngine
from mac_experiment.matrix.inputs import (
    DISTRIBUTION_METADATA,
    INPUT_DISTRIBUTIONS,
    Fp32InputSampler,
    fp32_bits_from_float,
    fp32_bits_to_dyadics,
    quantize_fp32_bits_to_mx,
)


def test_fp32_conversion_is_explicit_rne_and_rejects_nonfinite():
    assert fp32_bits_from_float(1.0) == 0x3F800000
    assert fp32_bits_from_float(1.1) == 0x3F8CCCCD
    assert fp32_bits_from_float(0.0) == 0x00000000
    assert fp32_bits_from_float(-0.0) == 0x80000000
    # 1 + 2**-24 is a tie between 1.0 and 1+2**-23; ties-to-even picks 1.0
    assert fp32_bits_from_float(1.0 + 2.0**-24) == 0x3F800000
    assert fp32_bits_from_float(1.0 + 3.0 * 2.0**-24) == 0x3F800002
    # value just below 2**-150 rounds to zero without becoming non-finite
    assert fp32_bits_from_float(2.0**-160) == 0x00000000
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            fp32_bits_from_float(bad)


def test_custom_distribution_parameters_are_rejected():
    with pytest.raises(ValueError, match="custom distribution parameters"):
        Fp32InputSampler("gaussian", {"mean": 100.0, "std": 0.0})
    with pytest.raises(ValueError, match="custom distribution parameters"):
        Fp32InputSampler("laplace", {"scale": 2.0})
    sampler = Fp32InputSampler("gaussian")
    assert sampler.metadata()["parameters_configurable"] is False
    assert sampler.metadata()["parameters"] == {"mean": 0.0, "std": 1.0}


def test_distribution_definitions_and_metadata():
    assert set(DISTRIBUTION_METADATA) == set(INPUT_DISTRIBUTIONS)
    for name in INPUT_DISTRIBUTIONS:
        meta = Fp32InputSampler(name).metadata()
        assert meta["theoretical_mean"] == 0.0
        assert meta["theoretical_rms"] == 1.0
        assert meta["algorithm"]
        assert meta["parameters"]
    assert DISTRIBUTION_METADATA["laplace"]["parameters"]["scale"] == pytest.approx(1 / math.sqrt(2))
    assert DISTRIBUTION_METADATA["student_t3"]["parameters"]["dof"] == 3
    assert DISTRIBUTION_METADATA["outlier"]["parameters"]["p_large"] == pytest.approx(0.01)
    assert DISTRIBUTION_METADATA["outlier"]["parameters"]["variance"] == pytest.approx(1.63)


def test_distribution_empirical_mean_and_rms_loose_tolerance():
    """Statistically stable checks with wide tolerances (no flaky tight bounds)."""
    for name in INPUT_DISTRIBUTIONS:
        sampler = Fp32InputSampler(name)
        bits = sampler.sample_matrix(seed=2026, matrix="x", rows=200, cols=100, pair_index=0)
        values, _ = fp32_bits_to_dyadics(bits)
        floats = [value.to_float() for value in values]
        mean = sum(floats) / len(floats)
        rms = math.sqrt(sum(value * value for value in floats) / len(floats))
        assert abs(mean) < 0.05, (name, mean)
        assert abs(rms - 1.0) < 0.05, (name, rms)


def test_same_pair_reproducible_different_pair_distinct():
    sampler = Fp32InputSampler("gaussian")
    first = sampler.sample_matrix(seed=3, matrix="x", rows=4, cols=64, pair_index=0)
    again = sampler.sample_matrix(seed=3, matrix="x", rows=4, cols=64, pair_index=0)
    assert first == again
    other = sampler.sample_matrix(seed=3, matrix="x", rows=4, cols=64, pair_index=1)
    assert first != other
    assert len(set(first)) > 8  # non-degenerate


def test_k_prefix_stable_and_pair_count_order_independent():
    sampler = Fp32InputSampler("outlier")
    short = sampler.sample_matrix(seed=5, matrix="w", rows=6, cols=64, pair_index=2)
    long = sampler.sample_matrix(seed=5, matrix="w", rows=6, cols=128, pair_index=2)
    for row in range(6):
        assert short[row * 64 : (row + 1) * 64] == long[row * 128 : row * 128 + 64]
    # building other pairs and shapes in any order cannot change pair 2
    sampler.sample_matrix(seed=5, matrix="w", rows=2, cols=32, pair_index=7)
    sampler.sample_matrix(seed=5, matrix="x", rows=6, cols=128, pair_index=2)
    sampler.sample_matrix(seed=5, matrix="w", rows=6, cols=64, pair_index=0)
    assert sampler.sample_matrix(seed=5, matrix="w", rows=6, cols=64, pair_index=2) == short


def test_activation_and_weight_streams_are_independent():
    activation = Fp32InputSampler("gaussian")
    weight = Fp32InputSampler("gaussian")
    x_bits = activation.sample_matrix(seed=4, matrix="x", rows=2, cols=64, pair_index=1)
    w_bits = weight.sample_matrix(seed=4, matrix="w", rows=2, cols=64, pair_index=1)
    assert x_bits != w_bits
    # changing the weight distribution does not perturb the activation stream
    other_weight = Fp32InputSampler("laplace")
    other_weight.sample_matrix(seed=4, matrix="w", rows=2, cols=64, pair_index=1)
    assert activation.sample_matrix(seed=4, matrix="x", rows=2, cols=64, pair_index=1) == x_bits


def test_input_quantization_axes_g32_scales_and_prefix():
    # X: (M, K) groups along K (axis 1); W: (K, N) groups along K (axis 0)
    x_rows, x_cols = 2, 96
    x_bits = [0x3F800000] * (x_rows * x_cols)  # all 1.0
    x_input = quantize_fp32_bits_to_mx("x", x_bits, x_rows, x_cols, group_axis=1)
    assert x_input.tensor.group_grid == (2, 3)
    assert x_input.tensor.scale_code(0, 0) == 119
    assert x_input.tensor.scale_code(1, 95) == 119
    assert x_input.tensor.element_code(0, 95) == 0x78

    w_rows, w_cols = 96, 5
    w_bits = [0x3F800000] * (w_rows * w_cols)
    w_input = quantize_fp32_bits_to_mx("w", w_bits, w_rows, w_cols, group_axis=0)
    assert w_input.tensor.group_grid == (3, 5)
    assert w_input.tensor.scale_code(0, 0) == 119
    assert w_input.tensor.scale_code(95, 4) == 119

    # prefix stability of the MX codes/scales: quantise K=32 separately
    prefix_x = quantize_fp32_bits_to_mx(
        "x", [0x3F800000] * (x_rows * 32), x_rows, 32, group_axis=1
    )
    for row in range(x_rows):
        assert (
            prefix_x.tensor.element_codes[row * 32 : (row + 1) * 32]
            == x_input.tensor.element_codes[row * 96 : row * 96 + 32]
        )
        assert prefix_x.tensor.scale_codes[row] == x_input.tensor.scale_codes[row * 3]
    prefix_w = quantize_fp32_bits_to_mx(
        "w", [0x3F800000] * (32 * w_cols), 32, w_cols, group_axis=0
    )
    assert prefix_w.tensor.element_codes == w_input.tensor.element_codes[: 32 * w_cols]
    assert prefix_w.tensor.scale_codes == w_input.tensor.scale_codes[:w_cols]


def test_input_quantization_is_exact_and_all_zero_group_canonical():
    # 1.1 rounds to E4M3 1.125; 0.5 + tiny is representable; zeros stay zero
    bits = [fp32_bits_from_float(1.1), fp32_bits_from_float(-0.0)] + [0x00000000] * 30
    result = quantize_fp32_bits_to_mx("x", bits, 1, 32, group_axis=1)
    assert result.tensor.element_code(0, 0) == 0x79  # scaled 1.125 * 2**8 = 288
    assert result.tensor.element_code(0, 1) == 0x80  # signed zero preserved
    assert result.tensor.scale_code(0, 0) == 119  # smallest e with 1.125 <= 448 * 2**e
    assert result.diagnostics["mx_zero_elements"] == 31
    assert result.diagnostics["definition"].startswith("input diagnostics")
    decoded = result.tensor.decoded_value(0, 0)
    assert decoded == Dyadic(9, -3)  # 1.125
    assert decoded != Dyadic(11, -4)  # not the raw FP32 1.1


def test_engine_input_record_requantizes_to_same_mx_codes():
    engine = MxMatrixEngine(2, 32, [32, 64], "gaussian", "outlier")
    sample = engine.build_sample(seed=1, pair_index=1)
    record = engine.input_record(sample)
    assert record["schema"] == "mac-mx-exp-v2"
    assert record["input_sampling_version"] == "mx-fp32-input-v1"
    x = quantize_fp32_bits_to_mx("x", record["x_fp32_bits"], 2, 64, group_axis=1)
    w = quantize_fp32_bits_to_mx("w", record["w_fp32_bits"], 64, 32, group_axis=0)
    assert x.tensor.element_codes == sample.x.element_codes
    assert x.tensor.scale_codes == sample.x.scale_codes
    assert w.tensor.element_codes == sample.w.element_codes
    assert w.tensor.scale_codes == sample.w.scale_codes
    assert record["x"]["element_fingerprint"] == sample.x.element_fingerprint()


def test_nonfinite_saved_bits_are_rejected():
    with pytest.raises(ValueError):
        quantize_fp32_bits_to_mx("x", [0x7F800000] + [0] * 31, 1, 32, group_axis=1)
