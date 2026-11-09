"""Metric correctness tests: bit match, ULP buckets, aux errors, merging."""

from __future__ import annotations

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.widthselect.metrics import (
    MetricsCollector,
    OutputComparison,
    ScalarErrorAccumulator,
    bucket_index,
    nearest_rank,
)


def _fp32_bits(value: float) -> int:
    from mac_experiment.arithmetic.fp32 import Fp32

    return Fp32.from_float(value).bits


def test_bit_match_distinguishes_signed_zero_but_ulp_merges_it():
    comparison = OutputComparison()
    comparison.add(_fp32_bits(0.0), 0x8000_0000, Dyadic.zero(), Dyadic(1, 0))
    result = comparison.finalize()
    assert result["counts"]["n_total"] == 1
    assert result["counts"]["n_bit_match"] == 0
    assert result["counts"]["n_finite_mismatch"] == 1
    assert result["counts"]["signed_zero_mismatch"] == 1
    assert result["conditional_ulp"]["p50"] == 0
    assert result["distance_buckets"]["buckets"]["0"]["count"] == 1


def test_ulp_buckets_and_conditional_quantiles():
    comparison = OutputComparison()
    # distance 0 (+0 vs -0), 1, 2, 3, 7
    comparison.add(_fp32_bits(0.0), 0x8000_0000, Dyadic.zero(), Dyadic(1, 0))
    comparison.add(_fp32_bits(1.0), _fp32_bits(1.0) + 1, Dyadic(1, 0), Dyadic(1, 0))
    comparison.add(_fp32_bits(1.0), _fp32_bits(1.0) + 2, Dyadic(1, 0), Dyadic(1, 0))
    comparison.add(_fp32_bits(1.0), _fp32_bits(1.0) + 3, Dyadic(1, 0), Dyadic(1, 0))
    comparison.add(_fp32_bits(1.0), _fp32_bits(1.0) + 7, Dyadic(1, 0), Dyadic(1, 0))
    # one exact match
    comparison.add(_fp32_bits(1.0), _fp32_bits(1.0), Dyadic(1, 0), Dyadic(1, 0))
    result = comparison.finalize()
    assert result["counts"]["n_total"] == 6
    assert result["counts"]["n_bit_match"] == 1
    assert result["counts"]["n_finite_pairs"] == 6
    assert result["counts"]["n_finite_mismatch"] == 5
    buckets = result["distance_buckets"]["buckets"]
    assert buckets["0"]["count"] == 2
    assert buckets["1"]["count"] == 1
    assert buckets["2"]["count"] == 1
    assert buckets["3-4"]["count"] == 1
    assert buckets[">=5"]["count"] == 1
    # distances sorted: 0,1,2,3,7 -> p50 = 2, p90 = 7, p99 = 7, max = 7
    conditional = result["conditional_ulp"]
    assert (conditional["p50"], conditional["p90"], conditional["p99"], conditional["max"]) == (2, 7, 7, 7)


def test_conditional_quantiles_null_when_no_mismatch():
    comparison = OutputComparison()
    comparison.add(_fp32_bits(1.5), _fp32_bits(1.5), Dyadic(3, -1), Dyadic(2, 0))
    result = comparison.finalize()
    assert result["bit_match_rate"] == 1.0
    assert result["conditional_ulp"]["status"] == "no_mismatch_samples"
    assert result["conditional_ulp"]["p50"] is None
    assert result["conditional_ulp"]["max"] is None


def test_nonfinite_candidate_counted_and_excluded_from_aux():
    comparison = OutputComparison()
    inf_bits = 0x7F80_0000
    comparison.add(_fp32_bits(3.0), inf_bits, Dyadic(3, 0), Dyadic(4, 0))
    result = comparison.finalize()
    assert result["counts"]["n_total"] == 1
    assert result["counts"]["n_finite_pairs"] == 0
    assert result["counts"]["n_nonfinite_candidate"] == 1
    assert result["events"]["reference_finite_candidate_nonfinite"] == 1
    assert result["aux"]["abs_error"]["status"] == "no_finite_candidate"
    assert result["bit_match_rate"] == 0.0


def test_aux_errors_are_exact():
    # S = 3, candidate = 2 -> |err| = 1; L1 = 10 -> eta = 1/10; rel L2 = 1/3
    comparison = OutputComparison()
    comparison.add(_fp32_bits(3.0), _fp32_bits(2.0), Dyadic(3, 0), Dyadic(10, 0))
    result = comparison.finalize()
    assert result["aux"]["abs_error"]["p50"] == 1.0
    assert result["aux"]["abs_error"]["mean"] == 1.0
    assert result["aux"]["eta_l1"]["p50"] == 0.1
    assert abs(result["aux"]["relative_l2"]["value"] - 1 / 3) < 1e-15
    assert result["aux"]["relative_l2"]["status"] == "ok"


def test_eta_and_relative_l2_statuses():
    comparison = OutputComparison()
    comparison.add(_fp32_bits(0.0), _fp32_bits(0.0), Dyadic.zero(), Dyadic.zero())
    result = comparison.finalize()
    assert result["aux"]["eta_l1"]["n_zero_l1"] == 1
    assert result["aux"]["eta_l1"]["status"] == "no_eta_samples"
    assert result["aux"]["relative_l2"]["status"] == "undefined_zero_reference"


def test_cell_is_exact_merge_of_pairs():
    collector = MetricsCollector()
    cell = ("tf32", "gaussian|gaussian", 21, 32)
    for pair_index in range(2):
        pair = cell + (f"pair{pair_index}",)
        collector.add(
            cell,
            pair,
            [_fp32_bits(3.0), _fp32_bits(1.0)],
            [_fp32_bits(2.0), _fp32_bits(1.0)],
            [Dyadic(3, 0), Dyadic(1, 0)],
            [Dyadic(10, 0), Dyadic(4, 0)],
            1,
        )
    result = collector.finalize()
    assert set(result["cells"]) == {cell}
    assert len(result["by_pair"]) == 2
    cell_result = result["cells"][cell]
    assert cell_result["counts"]["n_total"] == 4
    assert cell_result["counts"]["n_bit_match"] == 2
    # exact sums over both pairs: err^2 = 2, S^2 = 20 -> sqrt(2/20)
    assert abs(cell_result["aux"]["relative_l2"]["value"] - (2 / 20) ** 0.5) < 1e-15


def test_scalar_error_accumulator():
    accumulator = ScalarErrorAccumulator()
    accumulator.add(Dyadic(1, 0), Dyadic(3, 0), Dyadic(10, 0))
    accumulator.add(Dyadic(-2, 0), Dyadic(4, 0), Dyadic.zero())
    result = accumulator.finalize()
    assert result["abs_error"]["max"] == 2.0
    assert result["abs_error"]["mean"] == 1.5
    assert result["eta_l1"]["n_zero_l1"] == 1
    assert abs(result["relative_l2"]["value"] - (5 / 25) ** 0.5) < 1e-15


def test_bucket_and_nearest_rank_helpers():
    assert [bucket_index(value) for value in (0, 1, 2, 3, 4, 5, 100)] == [0, 1, 2, 3, 3, 4, 4]
    assert nearest_rank([1, 2, 3, 4], 0.5) == 2
    assert nearest_rank([1, 2, 3, 4], 0.9) == 4
    assert nearest_rank([], 0.5) is None
