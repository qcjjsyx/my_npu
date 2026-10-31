"""Input pipeline tests: distributions, streams, prefix stability, shared FP32."""

from __future__ import annotations

import math
import random
import struct

import pytest

from mac_experiment.matrix.inputs import (
    GENERATORS as LEGACY_GENERATORS,
    fp32_bits_from_float,
    quantize_fp32_bits_to_mx,
)
from mac_experiment.sampling.streams import mx_input_stream_rng
from mac_experiment.widthselect.distributions import (
    BITS_GENERATORS,
    DISTRIBUTION_METADATA,
    LEGACY_DISTRIBUTIONS,
    WIDTH_DISTRIBUTIONS,
    distribution_metadata,
    sample_fp32_matrix,
    validate_distribution,
)
from mac_experiment.widthselect.tensors import quantize_fp32_bits
from mac_experiment.widthselect.tf32_rne import is_special_code


def test_encoding_uniform_moments():
    uniform_meta = distribution_metadata("uniform")
    positive_meta = distribution_metadata("positive")

    def _empirical_rms(generator, seed: int, count: int) -> float:
        rng = random.Random(seed)
        total = 0.0
        for _ in range(count):
            value = struct.unpack("<f", struct.pack("<I", generator(rng)))[0]
            total += value * value
        return math.sqrt(total / count)

    assert _empirical_rms(BITS_GENERATORS["uniform"], 7, 100_000) == pytest.approx(
        uniform_meta["theoretical_rms"], rel=0.25
    )
    assert _empirical_rms(BITS_GENERATORS["positive"], 8, 100_000) == pytest.approx(
        positive_meta["theoretical_rms"], rel=0.25
    )
    # the fair-sign denominator must not be doubled: E[v^2] uses the magnitude
    # count, not twice the magnitude count
    magnitude_count = 255 * (1 << 23)
    ratio = positive_meta["theoretical_rms"] / uniform_meta["theoretical_rms"]
    assert ratio == pytest.approx(math.sqrt(magnitude_count / (magnitude_count - 1)), rel=1e-12)


def test_tf32_input_quantization_saturates_and_stays_finite():
    bits = [0x7F7FFFFF, 0xFF7FFFFF, 0x00000001, 0x80000001]
    quantized = quantize_fp32_bits("tf32", "x", bits, 1, 4, group_axis=1)
    max_finite = (254 << 10) | 1023
    assert quantized.tensor.codes[0] == max_finite
    assert quantized.tensor.codes[1] == (1 << 18) | max_finite
    assert all(not is_special_code(code) for code in quantized.tensor.codes)
    assert quantized.events["element_overflow_saturated"] == 2
    assert quantized.events["element_zeroed"] == 2
    assert quantized.diagnostics["tf32_saturated_elements"] == 2
    decoded_zero = quantized.tensor.decoded_value(0, 2)
    assert decoded_zero.is_zero()
    assert quantized.tensor.decoded_value(0, 3).is_zero()


def test_six_distribution_metadata_and_rejections():
    assert set(WIDTH_DISTRIBUTIONS) == set(DISTRIBUTION_METADATA)
    assert set(LEGACY_DISTRIBUTIONS) == {"gaussian", "laplace", "student_t3", "outlier"}
    for name in LEGACY_DISTRIBUTIONS:
        meta = distribution_metadata(name)
        assert meta["theoretical_mean"] == 0.0
        assert meta["theoretical_rms"] == 1.0
        assert meta["parameters_configurable"] is False
        assert meta["algorithm"]
    uniform = distribution_metadata("uniform")
    assert uniform["family"] == "uniform_finite_encoding"
    assert uniform["theoretical_mean"] == 0.0
    assert uniform["theoretical_rms"] > 1e30
    assert uniform["parameters"]["support_size"] == 2 * 255 * (1 << 23)
    positive = distribution_metadata("positive")
    assert positive["family"] == "uniform_finite_encoding_positive"
    assert positive["theoretical_mean"] > 1e30
    assert positive["theoretical_rms"] > 1e30
    assert positive["parameters"]["support_size"] == 255 * (1 << 23) - 1
    assert "encoding-domain" in uniform["fp32_conversion"]
    assert "RNE" in distribution_metadata("gaussian")["fp32_conversion"]
    assert DISTRIBUTION_METADATA["gaussian"]["source"].startswith("reused verbatim")
    with pytest.raises(ValueError, match="custom distribution parameters"):
        validate_distribution("gaussian", {"std": 2.0})
    with pytest.raises(ValueError, match="unknown input distribution"):
        validate_distribution("cauchy")


def test_streams_reproducible_and_coordinate_isolated():
    first = sample_fp32_matrix("gaussian", 7, "x", 2, 8, 0)
    again = sample_fp32_matrix("gaussian", 7, "x", 2, 8, 0)
    assert first == again
    assert len(set(first)) > 4
    assert sample_fp32_matrix("gaussian", 7, "x", 2, 8, 1) != first
    assert sample_fp32_matrix("gaussian", 8, "x", 2, 8, 0) != first
    assert sample_fp32_matrix("gaussian", 7, "w", 2, 8, 0) != first
    assert sample_fp32_matrix("laplace", 7, "x", 2, 8, 0) != first


def test_encoding_uniform_distributions():
    uniform = sample_fp32_matrix("uniform", 5, "x", 4, 512, 0)
    assert uniform == sample_fp32_matrix("uniform", 5, "x", 4, 512, 0)
    assert all(((bits >> 23) & 0xFF) != 0xFF for bits in uniform)
    assert any(bits >> 31 for bits in uniform) and any(not (bits >> 31) for bits in uniform)
    exponent_fields = {(bits >> 23) & 0xFF for bits in uniform}
    assert max(exponent_fields) > 200 and min(exponent_fields) < 20

    positive = sample_fp32_matrix("positive", 5, "x", 4, 512, 0)
    assert positive == sample_fp32_matrix("positive", 5, "x", 4, 512, 0)
    assert all((bits >> 31) == 0 for bits in positive)
    assert all((bits & 0x7FFF_FFFF) != 0 for bits in positive)
    assert all(((bits >> 23) & 0xFF) != 0xFF for bits in positive)


def test_encoding_distributions_follow_the_shared_pipeline():
    from mac_experiment.widthselect.tf32_rne import round_dyadic_to_tf32
    from mac_experiment.arithmetic.fp32 import Fp32

    bits = sample_fp32_matrix("uniform", 11, "w", 32, 64, 1)
    tf32 = quantize_fp32_bits("tf32", "w", bits, 32, 64, group_axis=0)
    for raw, code in zip(bits, tf32.tensor.codes):
        assert round_dyadic_to_tf32(Fp32.from_bits(raw).exact_dyadic()).code == code


def test_prefix_and_extension_stability():
    rows, cols = 3, 32
    full = sample_fp32_matrix("outlier", 3, "x", rows, cols, 1)
    prefix = sample_fp32_matrix("outlier", 3, "x", rows, 16, 1)
    for row in range(rows):
        assert prefix[row * 16 : (row + 1) * 16] == full[row * cols : row * cols + 16]
    # requesting fewer columns never changes existing elements
    assert sample_fp32_matrix("outlier", 3, "x", rows, 16, 1) == prefix


def test_new_stream_namespace_differs_from_historical():
    seed = 11
    historical_rng = mx_input_stream_rng(seed, "x", "gaussian", 0, 0, "value")
    historical = [fp32_bits_from_float(LEGACY_GENERATORS["gaussian"](historical_rng)) for _ in range(8)]
    regenerated = sample_fp32_matrix("gaussian", seed, "x", 1, 8, 0)
    assert regenerated != historical
    assert len(set(regenerated) & set(historical)) < 8


def test_mx_quantisation_matches_historical_quantiser_bit_for_bit():
    bits = sample_fp32_matrix("laplace", 5, "x", 4, 64, 0)
    ours = quantize_fp32_bits("mx_fp8", "x", bits, 4, 64, group_axis=1)
    legacy = quantize_fp32_bits_to_mx("x", bits, 4, 64, group_axis=1)
    assert ours.tensor.codes == legacy.tensor.element_codes
    assert ours.tensor.scale_codes == legacy.tensor.scale_codes
    assert ours.events == {key: int(value) for key, value in legacy.events.items()}


def test_shared_fp32_bits_produce_both_formats():
    bits = sample_fp32_matrix("uniform", 9, "w", 32, 16, 2)
    mx = quantize_fp32_bits("mx_fp8", "w", bits, 32, 16, group_axis=0)
    tf32 = quantize_fp32_bits("tf32", "w", bits, 32, 16, group_axis=0)
    assert mx.tensor.rows == tf32.tensor.rows == 32
    assert mx.tensor.cols == tf32.tensor.cols == 16
    assert len(tf32.tensor.codes) == len(bits)
    # TF32 codes are the RNE of the same FP32 bits (re-derived through decode)
    from mac_experiment.arithmetic.fp32 import Fp32
    from mac_experiment.widthselect.tf32_rne import round_dyadic_to_tf32

    for raw, code in zip(bits, tf32.tensor.codes):
        expected = round_dyadic_to_tf32(Fp32.from_bits(raw).exact_dyadic())
        assert expected.code == code


def test_target_tensor_roundtrip_and_fingerprints():
    bits = sample_fp32_matrix("student_t3", 2, "x", 2, 32, 0)
    mx = quantize_fp32_bits("mx_fp8", "x", bits, 2, 32, group_axis=1)
    payload = mx.tensor.to_dict()
    from mac_experiment.widthselect.tensors import TargetTensor

    restored = TargetTensor.from_dict(payload)
    assert restored.codes == mx.tensor.codes
    assert restored.scale_codes == mx.tensor.scale_codes
    assert restored.code_fingerprint() == mx.tensor.code_fingerprint()
    assert payload["scale_fingerprint"].startswith("sha256:")
    tf32 = quantize_fp32_bits("tf32", "x", bits, 2, 32, group_axis=1)
    tf_payload = tf32.tensor.to_dict()
    assert "scale_codes" not in tf_payload
    assert "signed_zero_elements" in tf32.diagnostics
