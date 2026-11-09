"""Tests for the ``alignment_only`` comparison.

``alignment_only`` compares the exact windowed partial-sum accumulation
``Q_B = S_B + E_align`` against the exact dot product ``S_B``.  Neither side is
rounded to FP32.  Validity depends only on a complete exact ``E_align`` and is
independent of the FP32 candidate/reference states.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest
import yaml

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.events import EVENT_KEYS
from mac_experiment.execution.engine import SampleEngine
from mac_experiment.experiments import block_sweep
from mac_experiment.formats import get_format
from mac_experiment.metrics.aggregate import SummaryBuilder
from mac_experiment.metrics.compare import (
    STATUS_NO_VALID,
    STATUS_OK,
    STATUS_PARTIAL,
    ComparisonAccumulator,
)
from mac_experiment.reduction import BlockReducer, BlockReduction, ProductTerm
from mac_experiment.results.enhance import enhance_run
from mac_experiment.results.reader import iter_records, load_summary
from mac_experiment.results.schema import SCHEMA_VERSION, SUMMARY_VERSION, dumps_strict, loads_strict
from mac_experiment.sampling.sampler import Sampler
from mac_experiment.version import PACKAGE_VERSION

INF_BITS = 0x7F800000
ONE_BITS = 0x3F800000
ZERO_BITS = 0x00000000


def make_record(sample_index=0, x=24, a_bits=ZERO_BITS, r_bits=ZERO_BITS, e_align=None, s_total=None, l1=None):
    s_total = Dyadic.zero() if s_total is None else s_total
    l1 = Dyadic(1, 0) if l1 is None else l1
    e_align = Dyadic.zero() if e_align is None else e_align
    return {
        "record_type": "snapshot",
        "schema": SCHEMA_VERSION,
        "mode": "width_sweep",
        "sample_id": f"fp8_e4m3|laplace|seed1|{sample_index:05d}",
        "seed": 1,
        "sample_index": sample_index,
        "format": "fp8_e4m3",
        "distribution": "laplace",
        "input_fingerprint": "sha256:test",
        "W": x,
        "B": 1,
        "K": 16,
        "a_bits": a_bits,
        "r_bits": r_bits,
        "a_finite": a_bits != INF_BITS,
        "r_finite": r_bits != INF_BITS,
        "S": s_total.to_dict(),
        "L1": l1.to_dict(),
        "E_align": e_align.to_dict(),
        "E_acc": Dyadic.zero().to_dict(),
        "E_acc_complete": True,
        "E_acc_prefix": None,
        "E_acc_prefix_until_block": None,
        "events_cumulative": {key: 0 for key in EVENT_KEYS},
        "events_endpoint": {key: 0 for key in EVENT_KEYS},
        "rho": None,
        "rho_per_block": None,
        "status": "ok",
    }


def build_cell(records):
    builder = SummaryBuilder("width_sweep", "W")
    for record in records:
        builder.add(record)
    return builder.finalize({"run_id": "unit"})["cells"][0]


def alignment_only_cell(records):
    return build_cell(records)["comparisons"]["alignment_only"]


def _block_terms(fmt, draw):
    terms = []
    for code_a, code_b in draw.pairs:
        product = fmt.product(fmt.decode(code_a), fmt.decode(code_b))
        terms.append(ProductTerm(product.dyadic, product.c))
    return terms


def test_saved_e_align_equals_exact_signed_block_sum():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "uniform")
    reducer = BlockReducer(window_offset=fmt.window_offset)
    blocks = 4
    engine = SampleEngine(fmt, sampler, windows=(24,), snapshots=(blocks,), mode="width_sweep")
    outcome = engine.run(1, 0)
    record = outcome.records[0]

    expected_e_align = Dyadic.zero()
    expected_s_total = Dyadic.zero()
    for block_index in range(blocks):
        terms = _block_terms(fmt, sampler.sample_block(1, 0, block_index))
        s_b = Dyadic.zero()
        for term in terms:
            s_b = s_b + term.dyadic
        reduction = reducer.reduce(terms, 24)
        expected_e_align = expected_e_align + (reduction.aligned_sum - s_b)
        expected_s_total = expected_s_total + s_b

    assert Dyadic.from_dict(record["S"]) == expected_s_total
    assert Dyadic.from_dict(record["E_align"]) == expected_e_align
    assert alignment_only_cell([record])["metrics"]["mean_signed_error"]["value"] == pytest.approx(
        expected_e_align.to_float()
    )


def test_no_alignment_loss_gives_zero_errors_and_match():
    fmt = get_format("fp8_e4m3")
    engine = SampleEngine(fmt, Sampler(fmt, "laplace"), windows=(41,), snapshots=(2,), mode="width_sweep")
    outcome = engine.run(3, 0)
    payload = alignment_only_cell(outcome.records)
    assert payload["status"] == STATUS_OK
    assert payload["coverage"]["n_invalid"] == 0
    assert payload["metrics"]["relative_l2"]["value"] == 0.0
    assert payload["metrics"]["match_rate"]["value"] == 1.0
    assert payload["metrics"]["max_abs_error"]["value"] == 0.0
    assert payload["metrics"]["p99_abs_error"]["value"] == 0.0


def test_signed_accumulation_is_not_abs_sum():
    payload = alignment_only_cell(
        [make_record(0, e_align=Dyadic(2, 0)), make_record(1, e_align=Dyadic(-2, 0))]
    )
    assert payload["metrics"]["mean_signed_error"]["value"] == 0.0
    assert payload["metrics"]["rmse_abs_error"]["value"] == pytest.approx(2.0)
    assert payload["metrics"]["max_abs_error"]["value"] == pytest.approx(2.0)
    assert payload["metrics"]["match_rate"]["value"] == 0.0
    zero_payload = alignment_only_cell([make_record(0, e_align=Dyadic.zero())])
    assert zero_payload["metrics"]["match_rate"]["value"] == 1.0
    assert zero_payload["metrics"]["p99_abs_error"]["value"] == 0.0


def test_stored_e_align_is_signed_sum_not_abs_sum():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "uniform")
    reducer = BlockReducer(window_offset=fmt.window_offset)
    blocks = 4
    signed = None
    unsigned = None
    for sample_index in range(20):
        signed = Dyadic.zero()
        unsigned = Dyadic.zero()
        for block_index in range(blocks):
            terms = _block_terms(fmt, sampler.sample_block(1, sample_index, block_index))
            s_b = Dyadic.zero()
            for term in terms:
                s_b = s_b + term.dyadic
            reduction = reducer.reduce(terms, 24)
            delta = reduction.aligned_sum - s_b
            signed = signed + delta
            unsigned = unsigned + abs(delta)
        if abs(unsigned) > abs(signed):
            break
    assert abs(unsigned) > abs(signed), "no discriminating sample found"

    engine = SampleEngine(fmt, sampler, windows=(24,), snapshots=(blocks,), mode="width_sweep")
    record = engine.run(1, sample_index).records[0]
    assert Dyadic.from_dict(record["E_align"]) == signed
    payload = alignment_only_cell([record])
    assert payload["metrics"]["mean_signed_error"]["value"] == pytest.approx(signed.to_float())
    assert payload["metrics"]["max_abs_error"]["value"] == pytest.approx(float(abs(signed).to_fraction()))
    assert float(abs(signed).to_fraction()) < float(unsigned.to_fraction())


def test_metrics_match_independent_fraction_computation():
    errors = [Dyadic(1, -1), Dyadic(-1, -2), Dyadic(3, -2)]
    references = [Dyadic(1, 0), Dyadic(2, 0), Dyadic(1, 1)]
    l1_values = [Dyadic(1, 0), Dyadic(1, 0), Dyadic(2, 0)]
    accumulator = ComparisonAccumulator("alignment_only")
    for error, reference, l1 in zip(errors, references, l1_values):
        accumulator.add(error, reference, l1)
    result = accumulator.finalize()

    def linear_quantile(values, q):
        n = len(values)
        position = (n - 1) * q
        lo = int(math.floor(position))
        hi = min(lo + 1, n - 1)
        frac = position - lo
        return float(values[lo] + frac * (values[hi] - values[lo]))

    abs_errors = sorted(abs(error.to_fraction()) for error in errors)
    etas = sorted(abs(error.to_fraction()) / l1.to_fraction() for error, l1 in zip(errors, l1_values))
    assert result["metrics"]["p99_abs_error"]["value"] == pytest.approx(linear_quantile(abs_errors, 0.99))
    assert result["metrics"]["p99_eta_l1"]["value"] == pytest.approx(linear_quantile(etas, 0.99))
    expected_rel_l2 = math.sqrt(
        float(sum(error.to_fraction() ** 2 for error in errors) / sum(ref.to_fraction() ** 2 for ref in references))
    )
    assert result["metrics"]["relative_l2"]["value"] == pytest.approx(expected_rel_l2)
    expected_rmse = math.sqrt(float(sum(error.to_fraction() ** 2 for error in errors) / len(errors)))
    assert result["metrics"]["rmse_abs_error"]["value"] == pytest.approx(expected_rmse)
    expected_mean = float(sum(error.to_fraction() for error in errors) / len(errors))
    assert result["metrics"]["mean_signed_error"]["value"] == pytest.approx(expected_mean)
    expected_ref_rms = math.sqrt(float(sum(ref.to_fraction() ** 2 for ref in references) / len(references)))
    assert result["reference_rms"]["value"] == pytest.approx(expected_ref_rms)
    assert result["metrics"]["match_rate"]["value"] == 0.0


class OverflowingReducer:
    def __init__(self):
        self.calls = 0

    def reduce(self, terms, w):
        self.calls += 1
        aligned = Dyadic(1, 0) if self.calls == 1 else Dyadic(1, 200)
        return BlockReduction(
            exact_sum=Dyadic.zero(),
            aligned_sum=aligned,
            window_lsb=0,
            cmax=0,
            zeroed_terms=0,
            all_zero=False,
            alignment_exact=True,
        )


def test_alignment_only_survives_fp32_overflow():
    fmt = get_format("fp8_e4m3")
    engine = SampleEngine(fmt, Sampler(fmt, "laplace"), windows=(24,), snapshots=(1, 2), mode="width_sweep")
    engine.reducer = OverflowingReducer()
    outcome = engine.run(1, 0)
    cell = build_cell(list(outcome.records))
    assert cell["comparisons"]["total"]["status"] == STATUS_PARTIAL
    assert cell["comparisons"]["fp32_baseline"]["status"] == STATUS_OK
    payload = cell["comparisons"]["alignment_only"]
    assert payload["status"] == STATUS_OK
    assert payload["coverage"]["n_valid"] == 2
    assert payload["metrics"]["p99_abs_error"]["value"] is not None
    assert cell["stage"]["E_align"]["status"] == STATUS_OK


def test_missing_or_partial_e_align_is_invalid_not_zero():
    good = make_record(0, e_align=Dyadic(1, 0))
    missing = make_record(1)
    del missing["E_align"]
    nulled = make_record(2)
    nulled["E_align"] = None
    malformed = make_record(3)
    malformed["E_align"] = {"m": "not-an-int", "e": 0}
    incomplete = make_record(4, e_align=Dyadic(1, 0))
    incomplete["E_align_complete"] = False

    cell = build_cell([good, missing, nulled, malformed, incomplete])
    payload = cell["comparisons"]["alignment_only"]
    assert payload["status"] == STATUS_PARTIAL
    assert payload["coverage"] == {"n_total": 5, "n_valid": 1, "n_invalid": 4, "invalid_rate": 0.8}
    assert payload["metrics"]["relative_l2"]["value"] is None
    assert payload["metrics"]["match_rate"]["value"] is None
    assert payload["finite_subset"]["metrics"]["mean_signed_error"]["value"] == 1.0
    assert cell["stage"]["E_align"]["status"] == STATUS_PARTIAL
    assert cell["stage"]["E_align"]["coverage"]["n_valid"] == 1

    only_missing = build_cell([missing])["comparisons"]["alignment_only"]
    assert only_missing["status"] == STATUS_NO_VALID
    assert only_missing["metrics"]["relative_l2"]["value"] is None
    assert only_missing["metrics"]["match_rate"]["value"] is None
    assert only_missing["finite_subset"] is None


def test_existing_comparisons_unchanged_and_stage_equivalence(tmp_path):
    config = {
        "mode": "block_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": 1,
        "W": 24,
        "B_list": [1, 2],
        "output_root": str(tmp_path / "results"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = block_sweep.run_from_file(path, verbose=False)
    summary = load_summary(run_dir)
    records = list(iter_records(run_dir))

    from collections import defaultdict

    groups = defaultdict(list)
    for record in records:
        groups[(record["format"], record["distribution"], record["B"])].append(record)

    for cell in summary["cells"]:
        cell_records = groups[(cell["format"], cell["distribution"], cell["x"])]
        expected = {"total": [], "window_effect": [], "fp32_baseline": [], "alignment_only": []}
        for record in cell_records:
            a = Fp32.from_bits(record["a_bits"]).exact_dyadic()
            r = Fp32.from_bits(record["r_bits"]).exact_dyadic()
            s_total = Dyadic.from_dict(record["S"])
            expected["total"].append(a - s_total)
            expected["window_effect"].append(a - r)
            expected["fp32_baseline"].append(r - s_total)
            expected["alignment_only"].append(Dyadic.from_dict(record["E_align"]))
        for comparison, errors in expected.items():
            payload = cell["comparisons"][comparison]
            assert payload["status"] == STATUS_OK
            mean_signed = float(sum(error.to_fraction() for error in errors) / len(errors))
            rmse = math.sqrt(float(sum(error.to_fraction() ** 2 for error in errors) / len(errors)))
            assert payload["metrics"]["mean_signed_error"]["value"] == mean_signed
            assert payload["metrics"]["rmse_abs_error"]["value"] == rmse

        align_payload = cell["comparisons"]["alignment_only"]
        assert align_payload["metrics"]["mean_signed_error"]["value"] == cell["stage"]["E_align"]["mean"]
        assert align_payload["metrics"]["rmse_abs_error"]["value"] == cell["stage"]["E_align"]["rms"]


def _write_enhance_source(tmp_path):
    config = {
        "mode": "width_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": 2,
        "B": 2,
        "output_root": str(tmp_path / "results"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    from mac_experiment.experiments import width_sweep

    return width_sweep.run_from_file(path, verbose=False)


def test_enhance_json_csv_plots_and_source_untouched(tmp_path):
    run_dir = _write_enhance_source(tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in run_dir.iterdir()}
    out = enhance_run(run_dir, out_dir=tmp_path / "enhanced", write_plots=True)

    after = {p.name: p.stat().st_mtime_ns for p in run_dir.iterdir()}
    assert before == after

    summary = load_summary(out)
    assert summary["summary_version"] == SUMMARY_VERSION
    assert summary["input_schema_version"] == SCHEMA_VERSION
    for cell in summary["cells"]:
        assert "alignment_only" in cell["comparisons"]
        assert cell["comparison_statuses"]["alignment_only"] == STATUS_OK
    round_trip = loads_strict(dumps_strict(summary))
    assert "alignment_only" in round_trip["cells"][0]["comparisons"]

    csv_text = (out / "summary.csv").read_text(encoding="utf-8")
    assert ",alignment_only," in csv_text
    analysis_manifest = json.loads((out / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert analysis_manifest["analysis_kind"] == "alignment_only_enhancement"
    assert analysis_manifest["analysis_version"] == PACKAGE_VERSION
    assert analysis_manifest["source"]["schema_version"] == SCHEMA_VERSION
    assert analysis_manifest["source"]["files"]["samples.jsonl.gz"]["sha256"].startswith("sha256:")
    assert analysis_manifest["records_read"] == summary["n_records_read"]

    plots = list((out / "plots").glob("*.png"))
    assert plots and all(path.stat().st_size > 1000 for path in plots)

    with pytest.raises(FileExistsError):
        enhance_run(run_dir, out_dir=tmp_path / "enhanced", write_plots=False)
    with pytest.raises(ValueError):
        enhance_run(run_dir, out_dir=run_dir / "nested", write_plots=False)


def test_enhance_entry_does_not_load_sampling_or_engine(tmp_path):
    run_dir = _write_enhance_source(tmp_path)
    out = tmp_path / "offline"
    script = (
        "import sys\n"
        "from mac_experiment.cli import main\n"
        f"code = main(['enhance', '--run', {str(run_dir)!r}, '--out', {str(out)!r}, '--no-plots'])\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('mac_experiment.sampling') or name.startswith('mac_experiment.execution')\n"
        ")\n"
        "print('RETURN', code)\n"
        "print('LOADED', ','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert "RETURN 0" in result.stdout
    assert "LOADED \n" in result.stdout or "LOADED\n" in result.stdout, result.stdout
    assert (out / "summary.json").exists()
