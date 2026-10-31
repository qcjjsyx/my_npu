"""Regression tests: non-finite samples, status propagation, display.

Covers the whole-batch / finite-subset policy: invalid samples must never be
reported as normal accuracy, per-comparison validity is independent, and the
E_acc stage is undefined once the candidate accumulator overflowed.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pytest
import yaml

from mac_experiment.accumulation.accumulator import Fp32Accumulator
from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.execution.engine import EVENT_KEYS, SampleEngine
from mac_experiment.experiments import block_sweep
from mac_experiment.formats import get_format
from mac_experiment.metrics.aggregate import SummaryBuilder
from mac_experiment.metrics.compare import (
    STATUS_NO_VALID,
    STATUS_OK,
    STATUS_PARTIAL,
    ComparisonAccumulator,
    StageAccumulator,
)
from mac_experiment.plotting import plot_all
from mac_experiment.reduction import BlockReduction
from mac_experiment.results.reader import iter_records, load_summary
from mac_experiment.results.reanalyze import reanalyze
from mac_experiment.results.schema import SCHEMA_VERSION, SUMMARY_VERSION, dumps_strict, loads_strict
from mac_experiment.results.writer import JsonlSink, summary_to_csv_rows, write_json, write_summary_csv
from mac_experiment.sampling.sampler import Sampler
from mac_experiment.version import PACKAGE_VERSION, STREAM_VERSION

INF_BITS = 0x7F800000
ONE_BITS = 0x3F800000
ZERO_BITS = 0x00000000


def make_record(
    sample_index=0,
    x=24,
    a_bits=ZERO_BITS,
    r_bits=ZERO_BITS,
    s_total=None,
    l1=None,
    e_align=None,
    e_acc=None,
    e_acc_complete=True,
    prefix_until=None,
    status="ok",
):
    s_total = Dyadic.zero() if s_total is None else s_total
    l1 = Dyadic(1, 0) if l1 is None else l1
    e_align = Dyadic.zero() if e_align is None else e_align
    e_acc = Dyadic.zero() if e_acc is None else e_acc
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
        "E_acc": e_acc.to_dict() if e_acc_complete else None,
        "E_acc_complete": e_acc_complete,
        "E_acc_prefix": None if e_acc_complete else e_acc.to_dict(),
        "E_acc_prefix_until_block": None if e_acc_complete else prefix_until,
        "events_cumulative": {key: 0 for key in EVENT_KEYS},
        "events_endpoint": {key: 0 for key in EVENT_KEYS},
        "rho": None,
        "rho_per_block": None,
        "status": status,
    }


def build_cell(records):
    builder = SummaryBuilder("width_sweep", "W")
    for record in records:
        builder.add(record)
    return builder.finalize({"run_id": "unit"})["cells"][0]


def test_all_valid_zero_error_is_ok():
    cell = build_cell([make_record(0), make_record(1)])
    assert cell["status"] == STATUS_OK
    for comparison in ("total", "window_effect", "fp32_baseline"):
        payload = cell["comparisons"][comparison]
        assert payload["status"] == STATUS_OK
        assert payload["coverage"] == {
            "n_total": 2,
            "n_valid": 2,
            "n_invalid": 0,
            "invalid_rate": 0.0,
        }
        assert payload["metrics"]["relative_l2"]["value"] == 0.0
        assert payload["metrics"]["match_rate"]["value"] == 1.0
        assert payload["finite_subset"] is None


def test_partial_candidate_nonfinite_marks_partial_and_subset():
    valid = make_record(0)
    invalid = make_record(1, a_bits=INF_BITS, e_acc_complete=False, status="nonfinite_fp32")
    cell = build_cell([valid, invalid])
    total = cell["comparisons"]["total"]
    assert total["status"] == STATUS_PARTIAL
    assert total["metrics"]["match_rate"]["value"] is None
    assert total["metrics"]["relative_l2"]["value"] is None
    assert total["coverage"]["n_total"] == 2
    assert total["coverage"]["n_valid"] == 1
    assert total["coverage"]["n_invalid"] == 1
    assert total["coverage"]["invalid_rate"] == 0.5
    subset = total["finite_subset"]
    assert subset is not None
    assert "finite subset" in subset["note"]
    assert subset["coverage"]["n_valid"] == 1
    assert subset["metrics"]["match_rate"]["value"] == 1.0
    assert cell["comparisons"]["window_effect"]["status"] == STATUS_PARTIAL
    assert cell["status"] == "partial_comparisons_invalid"


def test_all_candidates_nonfinite_is_no_valid_samples():
    cell = build_cell(
        [
            make_record(0, a_bits=INF_BITS, e_acc_complete=False, status="nonfinite_fp32"),
            make_record(1, a_bits=INF_BITS, e_acc_complete=False, status="nonfinite_fp32"),
        ]
    )
    total = cell["comparisons"]["total"]
    assert total["status"] == STATUS_NO_VALID
    assert total["metrics"]["relative_l2"]["value"] is None
    assert total["metrics"]["relative_l2"]["status"] == STATUS_NO_VALID
    assert total["finite_subset"] is None
    assert total["coverage"]["invalid_rate"] == 1.0


def test_candidate_nonfinite_baseline_still_defined():
    valid = make_record(0)
    invalid = make_record(1, a_bits=INF_BITS, r_bits=ONE_BITS, s_total=Dyadic.zero(), e_acc_complete=False, status="nonfinite_fp32")
    cell = build_cell([valid, invalid])
    baseline = cell["comparisons"]["fp32_baseline"]
    assert baseline["status"] == STATUS_OK
    assert baseline["coverage"]["n_valid"] == 2
    assert baseline["metrics"]["rmse_abs_error"]["value"] == pytest.approx(math.sqrt(0.5))
    assert cell["comparisons"]["total"]["status"] == STATUS_PARTIAL
    assert cell["comparisons"]["window_effect"]["status"] == STATUS_PARTIAL


def test_reference_nonfinite_total_still_defined():
    valid = make_record(0)
    invalid = make_record(1, a_bits=ONE_BITS, r_bits=INF_BITS, s_total=Dyadic.zero(), status="nonfinite_fp32")
    cell = build_cell([valid, invalid])
    total = cell["comparisons"]["total"]
    assert total["status"] == STATUS_OK
    assert total["metrics"]["rmse_abs_error"]["value"] == pytest.approx(math.sqrt(0.5))
    assert cell["comparisons"]["window_effect"]["status"] == STATUS_PARTIAL
    assert cell["comparisons"]["fp32_baseline"]["status"] == STATUS_PARTIAL
    assert cell["comparisons"]["fp32_baseline"]["coverage"]["n_valid"] == 1
    assert cell["status"] == "partial_comparisons_invalid"


class OverflowingReducer:
    """First block returns a normal partial, second block overflows FP32."""

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


def test_real_fp32_overflow_marks_e_acc_undefined():
    max_finite = Fp32.from_bits(0x7F7FFFFF)
    accumulator = Fp32Accumulator()
    accumulator.update(max_finite.exact_dyadic())
    outcome = accumulator.update(max_finite.exact_dyadic())
    assert outcome.fp32.is_inf()
    assert outcome.delta is None
    assert outcome.overflowed

    fmt = get_format("fp8_e4m3")
    engine = SampleEngine(fmt, Sampler(fmt, "laplace"), windows=(24,), snapshots=(1, 2), mode="width_sweep")
    engine.reducer = OverflowingReducer()
    sample = engine.run(1, 0)
    records = {record["B"]: record for record in sample.records}
    first = records[1]
    assert first["a_finite"] and first["E_acc_complete"]
    assert first["E_acc"] is not None
    assert first["E_acc_prefix"] is None
    second = records[2]
    assert not second["a_finite"]
    assert not second["E_acc_complete"]
    assert second["E_acc"] is None
    assert second["E_acc_prefix"] == Dyadic.zero().to_dict()
    assert second["E_acc_prefix_until_block"] == 0
    assert second["E_align"] is not None
    assert sample.status == "nonfinite_fp32"

    cell = build_cell(list(sample.records))
    assert cell["comparisons"]["total"]["status"] == STATUS_PARTIAL
    assert cell["comparisons"]["window_effect"]["status"] == STATUS_PARTIAL
    assert cell["comparisons"]["fp32_baseline"]["status"] == STATUS_OK
    assert cell["stage"]["E_align"]["status"] == STATUS_OK
    assert cell["stage"]["E_acc"]["status"] == STATUS_PARTIAL
    assert cell["stage"]["E_acc"]["finite_subset"]["n"] == 1
    assert cell["counts"]["nonfinite"] == 1


def test_zero_reference_valid_vs_no_valid_distinction():
    valid = ComparisonAccumulator("total")
    valid.add(Dyadic.zero(), Dyadic.zero(), Dyadic.zero())
    result = valid.finalize()
    assert result["status"] == STATUS_OK
    assert result["metrics"]["relative_l2"]["value"] == 0.0

    empty = ComparisonAccumulator("total")
    empty.add_invalid()
    result = empty.finalize()
    assert result["status"] == STATUS_NO_VALID
    assert result["metrics"]["relative_l2"]["value"] is None
    assert result["metrics"]["relative_l2"]["status"] == STATUS_NO_VALID

    stage = StageAccumulator("E_acc")
    stage.add_invalid()
    stage_result = stage.finalize()
    assert stage_result["status"] == STATUS_NO_VALID
    assert stage_result["mean"] is None and stage_result["rms"] is None
    assert stage_result["finite_subset"] is None


def _write_synthetic_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "plots").mkdir()
    config = {
        "mode": "width_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": 2,
        "cancellation_sign_mode": "greedy",
        "record_block_rho": False,
        "B": 1,
        "W": None,
        "B_list": [],
        "width_scan": list(range(18, 42)),
        "schema_version": SCHEMA_VERSION,
        "stream_version": STREAM_VERSION,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "summary_version": SUMMARY_VERSION,
        "stream_version": STREAM_VERSION,
        "package_version": PACKAGE_VERSION,
        "run_id": "run",
        "mode": "width_sweep",
        "x_name": "W",
        "fixed": {"B": 1},
        "config": config,
        "config_hash": "test",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "source_digest": "sha256:test",
        "status": "completed",
    }
    write_json(run_dir / "manifest.json", manifest)
    records = [
        make_record(0),
        make_record(1, a_bits=INF_BITS, e_acc_complete=False, status="nonfinite_fp32"),
    ]
    with JsonlSink(run_dir / "samples.jsonl.gz") as sink:
        for record in records:
            sink.write(record)
    builder = SummaryBuilder("width_sweep", "W")
    for record in records:
        builder.add(record)
    summary = builder.finalize(
        {
            "run_id": "run",
            "mode": "width_sweep",
            "config": config,
            "config_hash": "test",
            "fixed": {"B": 1},
            "created_utc": "2026-01-01T00:00:00+00:00",
            "source_digest": "sha256:test",
            "schema_version": SCHEMA_VERSION,
            "summary_version": SUMMARY_VERSION,
        }
    )
    write_json(run_dir / "summary.json", summary)
    write_summary_csv(run_dir / "summary.csv", summary)
    return run_dir, summary


def test_strict_json_csv_reanalysis_and_plot(tmp_path):
    run_dir, summary = _write_synthetic_run(tmp_path)
    round_trip = loads_strict(dumps_strict(summary))
    assert round_trip["cells"][0]["comparisons"]["total"]["status"] == STATUS_PARTIAL
    assert round_trip["cells"][0]["comparisons"]["total"]["metrics"]["match_rate"]["value"] is None

    rows = summary_to_csv_rows(summary)
    whole = [row for row in rows if row["scope"] == "whole_batch" and row["comparison"] == "total"]
    subset = [row for row in rows if row["scope"] == "finite_subset" and row["comparison"] == "total"]
    assert whole and all(row["status"] == STATUS_PARTIAL for row in whole)
    assert all(row["value"] == "" for row in whole)
    assert all(row["n_invalid"] == 1 for row in whole)
    assert subset and any(row["metric"] == "match_rate" and row["value"] != "" for row in subset)
    assert any("finite subset" in row["note"] for row in subset)
    csv_text = (run_dir / "summary.csv").read_text(encoding="utf-8")
    assert "finite_subset" in csv_text and "invalid_rate" in csv_text

    result = reanalyze(run_dir)
    assert result["matches_original_summary"], result["differences"]

    written = plot_all(load_summary(run_dir), tmp_path / "plots")
    assert written and all(path.exists() and path.stat().st_size > 1000 for path in written)


def test_valid_metrics_values_unchanged():
    accumulator = ComparisonAccumulator("total")
    accumulator.add(Dyadic(1, -1), Dyadic(1, 0), Dyadic(1, 0))
    accumulator.add(Dyadic(-1, -2), Dyadic(1, 0), Dyadic(1, 0))
    result = accumulator.finalize()
    assert result["status"] == STATUS_OK
    assert result["metrics"]["mean_signed_error"]["value"] == pytest.approx(0.125)
    assert result["metrics"]["rmse_abs_error"]["value"] == pytest.approx(math.sqrt(0.15625))
    assert result["metrics"]["max_abs_error"]["value"] == pytest.approx(0.5)
    assert result["metrics"]["match_rate"]["value"] == 0.0
    assert result["metrics"]["relative_l2"]["value"] == pytest.approx(math.sqrt(0.3125 / 2))
    assert result["metrics"]["p99_abs_error"]["value"] == pytest.approx(0.4975)
    assert result["reference_rms"]["value"] == pytest.approx(1.0)
    assert result["coverage"] == {"n_total": 2, "n_valid": 2, "n_invalid": 0, "invalid_rate": 0.0}
    assert result["finite_subset"] is None


def test_valid_run_metrics_unchanged_regression(tmp_path):
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
    assert summary["schema_version"] == SCHEMA_VERSION
    for cell in summary["cells"]:
        assert cell["status"] == STATUS_OK
        for payload in cell["comparisons"].values():
            assert payload["status"] == STATUS_OK
            assert payload["coverage"]["n_invalid"] == 0
            assert payload["coverage"]["invalid_rate"] == 0.0
            assert payload["finite_subset"] is None

    groups = defaultdict(list)
    for record in iter_records(run_dir):
        groups[(record["format"], record["distribution"], record["B"])].append(record)
    for cell in summary["cells"]:
        records = groups[(cell["format"], cell["distribution"], cell["x"])]
        errors = [
            Fp32.from_bits(record["a_bits"]).exact_dyadic() - Dyadic.from_dict(record["S"])
            for record in records
        ]
        expected = float(sum(error.to_fraction() for error in errors) / len(errors))
        assert cell["comparisons"]["total"]["metrics"]["mean_signed_error"]["value"] == pytest.approx(expected)
