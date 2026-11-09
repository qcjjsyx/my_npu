import math

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.metrics.compare import (
    STATUS_NO_VALID,
    STATUS_OK,
    STATUS_ZERO_L1,
    STATUS_ZERO_REFERENCE,
    ComparisonAccumulator,
    StageAccumulator,
)
from mac_experiment.metrics.quantiles import linear_quantile
from mac_experiment.results.schema import dumps_strict, loads_strict


def test_linear_quantile_definition():
    values = [0.0, 1.0, 2.0, 3.0]
    assert linear_quantile(values, 0.99) == pytest.approx(2.97)
    assert linear_quantile(values, 0.5) == pytest.approx(1.5)
    assert linear_quantile([], 0.99) is None
    assert linear_quantile([5.0], 0.99) == 5.0


def test_exact_match_and_zero_denominators():
    acc = ComparisonAccumulator("total")
    acc.add(Dyadic.zero(), Dyadic.zero(), Dyadic.zero())
    result = acc.finalize()
    assert result["metrics"]["match_rate"]["value"] == 1.0
    assert result["metrics"]["relative_l2"]["value"] == 0.0
    assert result["metrics"]["relative_l2"]["status"] == STATUS_OK
    assert result["metrics"]["p99_eta_l1"]["value"] == 0.0


def test_zero_reference_with_nonzero_error_is_undefined():
    acc = ComparisonAccumulator("total")
    acc.add(Dyadic(1, -5), Dyadic.zero(), Dyadic(1, -5))
    result = acc.finalize()
    assert result["metrics"]["relative_l2"]["value"] is None
    assert result["metrics"]["relative_l2"]["status"] == STATUS_ZERO_REFERENCE
    assert result["counts"]["n_zero_reference"] == 1


def test_zero_l1_with_nonzero_error_is_undefined_not_faked():
    acc = ComparisonAccumulator("total")
    acc.add(Dyadic(1, -5), Dyadic(1, -5), Dyadic.zero())
    result = acc.finalize()
    assert result["metrics"]["p99_eta_l1"]["value"] is None
    assert result["metrics"]["p99_eta_l1"]["status"] == STATUS_ZERO_L1
    assert result["metrics"]["relative_l2"]["value"] == pytest.approx(1.0)


def test_invalid_sample_marks_no_valid_samples():
    acc = ComparisonAccumulator("total")
    acc.add_invalid()
    result = acc.finalize()
    assert result["status"] == STATUS_NO_VALID
    assert result["counts"]["n_invalid"] == 1
    assert result["coverage"]["invalid_rate"] == 1.0
    assert result["metrics"]["relative_l2"]["value"] is None
    assert result["metrics"]["relative_l2"]["status"] == STATUS_NO_VALID
    assert result["finite_subset"] is None


def test_metrics_use_exact_errors_before_float_conversion():
    acc = ComparisonAccumulator("window_effect")
    tiny = Dyadic(1, -200)
    acc.add(tiny, Dyadic(1, 0), Dyadic(1, 0))
    result = acc.finalize()
    assert result["metrics"]["match_rate"]["value"] == 0.0
    assert result["metrics"]["p99_abs_error"]["value"] == tiny.to_float()
    assert result["reference_rms"]["value"] == pytest.approx(1.0)


def test_reference_rms_saved_separately():
    acc = ComparisonAccumulator("fp32_baseline")
    acc.add(Dyadic.zero(), Dyadic(3, 0), Dyadic(3, 0))
    acc.add(Dyadic.zero(), Dyadic(4, 0), Dyadic(4, 0))
    result = acc.finalize()
    expected = math.sqrt((9 + 16) / 2)
    assert result["reference_rms"]["value"] == pytest.approx(expected)


def test_stage_accumulator_mean_and_rms():
    stage = StageAccumulator("E_align")
    stage.add(Dyadic(1, 0))
    stage.add(Dyadic(-1, 0))
    result = stage.finalize()
    assert result["mean"] == 0.0
    assert result["rms"] == pytest.approx(1.0)


def test_strict_json_rejects_non_finite():
    with pytest.raises(ValueError):
        dumps_strict({"value": float("nan")})
    with pytest.raises(ValueError):
        loads_strict('{"value": NaN}')
    with pytest.raises(ValueError):
        loads_strict('{"value": Infinity}')
    assert loads_strict(dumps_strict({"value": None})) == {"value": None}


def test_fp32_roundtrip_dict():
    value = Fp32.from_bits(0xBF800000)
    assert Fp32.from_dict(value.to_dict()) == value
    assert value.to_dict()["hex"] == "0xBF800000"


def test_stage_accumulator_merge_matches_direct_accumulation():
    """merge_from must use StageAccumulator's own fields (total/total_sq)."""
    values = [Dyadic(3, -2), Dyadic(-1, -3), Dyadic(7, -1), Dyadic(0, 0)]
    direct = StageAccumulator("E_align")
    for value in values:
        direct.add(value)
    direct.add_invalid()

    left = StageAccumulator("E_align")
    for value in values[:2]:
        left.add(value)
    left.add_invalid()
    right = StageAccumulator("E_align")
    for value in values[2:]:
        right.add(value)
    left.merge_from(right)

    assert left.finalize() == direct.finalize()
    with pytest.raises(ValueError):
        StageAccumulator("other").merge_from(left)
