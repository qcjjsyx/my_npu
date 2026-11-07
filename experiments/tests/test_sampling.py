import math

import pytest

from mac_experiment.formats import get_format
from mac_experiment.sampling.pmf import build_magnitude_pmf
from mac_experiment.sampling.sampler import DISTRIBUTIONS, Sampler
from mac_experiment.sampling.streams import stream_rng


def pmf_for(fmt_name, dist_name):
    fmt = get_format(fmt_name)
    spec = DISTRIBUTIONS[dist_name]
    return build_magnitude_pmf(fmt, spec.magnitude_kind, spec.shape)


@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "tf32"])
@pytest.mark.parametrize(
    "dist_name",
    ["positive", "uniform", "exponent_gaussian", "laplace", "student_t3", "outlier"],
)
def test_pmf_nonnegative_normalised_and_legal(fmt_name, dist_name):
    fmt = get_format(fmt_name)
    pmf = pmf_for(fmt_name, dist_name)
    assert len(pmf.codes) == len(pmf.probs) == len(pmf.values)
    assert all(p >= 0.0 for p in pmf.probs)
    assert math.isclose(sum(pmf.probs), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    for code in pmf.codes:
        operand = fmt.decode(code)
        assert not operand.is_zero
        assert not operand.is_special
        assert operand.raw_exp >= 1
        if fmt_name == "tf32":
            assert operand.raw_exp <= 254
    assert pmf.metadata["retained_probability"] <= 1.0
    assert pmf.metadata["excluded_probability"] >= 0.0
    assert pmf.rms() > 0.0


def test_code_interval_pmf_uses_midpoint_boundaries_and_tail_mass():
    pmf = pmf_for("fp8_e4m3", "laplace")
    meta = pmf.metadata
    assert meta["boundary_rule"].startswith("interior boundaries arithmetic midpoints")
    assert 0.0 < meta["retained_probability"] <= 1.0
    assert meta["excluded_probability"] == pytest.approx(1.0 - meta["retained_probability"])
    quantiles = meta["magnitude_quantiles"]
    assert quantiles["p50"] <= quantiles["p90"] <= quantiles["p99"]


def test_exponent_gaussian_reports_truncation():
    pmf = pmf_for("fp8_e4m3", "exponent_gaussian")
    meta = pmf.metadata
    assert meta["params"]["center"] == 7
    assert meta["params"]["sigma"] == 1.0
    assert 0.0 <= meta["excluded_probability"] < 0.2
    tf32 = pmf_for("tf32", "exponent_gaussian").metadata
    assert tf32["params"]["center"] == 127
    assert tf32["params"]["sigma"] == 4.0


def test_all_distributions_sample_only_legal_codes():
    for fmt_name in ("fp8_e4m3", "tf32"):
        fmt = get_format(fmt_name)
        for dist_name in DISTRIBUTIONS:
            sampler = Sampler(fmt, dist_name)
            for block_index in range(20):
                draw = sampler.sample_block(11, 0, block_index)
                assert len(draw.pairs) == 16
                for code_a, code_b in draw.pairs:
                    assert fmt.is_legal_input(code_a)
                    assert fmt.is_legal_input(code_b)


def test_streams_are_deterministic_and_block_independent():
    rng_a = stream_rng(1, "fp8_e4m3", "laplace", 0, 3, "amplitude")
    rng_b = stream_rng(1, "fp8_e4m3", "laplace", 0, 3, "amplitude")
    assert [rng_a.random() for _ in range(5)] == [rng_b.random() for _ in range(5)]
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "laplace")
    draw_first = sampler.sample_block(5, 2, 4)
    sampler.sample_block(5, 2, 0)
    sampler.sample_block(5, 2, 9)
    sampler.sample_block(99, 7, 1)
    draw_again = sampler.sample_block(5, 2, 4)
    assert draw_first.pairs == draw_again.pairs


def test_prefix_stability_of_samples():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "uniform")
    draw = sampler.sample_block(3, 1, 2)
    fresh = Sampler(fmt, "uniform").sample_block(3, 1, 2)
    assert draw.pairs == fresh.pairs
    sampler.sample_block(3, 0, 0)
    sampler.sample_block(3, 5, 7)
    assert sampler.sample_block(3, 1, 2).pairs == draw.pairs


def test_cancellation_preserves_magnitudes_and_has_no_rejection():
    fmt = get_format("tf32")
    greedy = Sampler(fmt, "cancellation", "greedy")
    independent = Sampler(fmt, "cancellation", "independent")
    for block_index in range(10):
        draw_g = greedy.sample_block(1, 0, block_index)
        draw_i = independent.sample_block(1, 0, block_index)
        assert sorted(m.to_fraction() for m in draw_g.magnitudes) == sorted(
            m.to_fraction() for m in draw_i.magnitudes
        )
        assert len(draw_g.magnitudes) == 16
        assert all(m.sign() > 0 for m in draw_g.magnitudes)


def test_cancellation_signs_consistent_with_operand_signs():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "cancellation", "greedy")
    for block_index in range(10):
        draw = sampler.sample_block(2, 0, block_index)
        total = None
        for (code_a, code_b), magnitude, sign in zip(draw.pairs, draw.magnitudes, draw.product_signs):
            operand_a = fmt.decode(code_a)
            operand_b = fmt.decode(code_b)
            expected = 1 if operand_a.sign == operand_b.sign else -1
            assert sign == expected
            term = magnitude if sign > 0 else -magnitude
            total = term if total is None else total + term
        assert total is not None
        assert draw.rho is not None and 0.0 <= draw.rho <= 1.0


def test_cancellation_greedy_reduces_rho():
    fmt = get_format("fp8_e4m3")
    greedy = Sampler(fmt, "cancellation", "greedy")
    independent = Sampler(fmt, "cancellation", "independent")
    rho_greedy = [greedy.sample_block(1, 0, b).rho for b in range(30)]
    rho_independent = [independent.sample_block(1, 0, b).rho for b in range(30)]
    assert sum(rho_greedy) / len(rho_greedy) < sum(rho_independent) / len(rho_independent)


def test_independent_cancellation_sign_stream_matches_fresh_stream():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "cancellation", "independent")
    draw = sampler.sample_block(4, 0, 1)
    rng = stream_rng(4, fmt.name, "cancellation", 0, 1, "sign")
    expected = sorted(1 if rng.getrandbits(1) else -1 for _ in range(16))
    assert sorted(draw.product_signs) == expected


def test_sampling_frequency_is_plausible():
    fmt = get_format("fp8_e4m3")
    pmf = pmf_for("fp8_e4m3", "laplace")
    rng = stream_rng(9, fmt.name, "laplace", 0, 0, "amplitude")
    draws = [pmf.sample(rng) for _ in range(20000)]
    assert len(set(draws)) > 40
    empirical_rms = math.sqrt(sum(v * v for v in (fmt.operand_value(fmt.decode(c)).to_float() for c in draws)) / len(draws))
    assert 0.5 * pmf.rms() < empirical_rms < 2.0 * pmf.rms()
