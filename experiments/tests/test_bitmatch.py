"""Tests for the offline bit-match / ULP-distance analysis."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from mac_experiment.arithmetic.bf16 import round_dyadic_to_bf16
from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.rounding import round_dyadic_to_fp32
from mac_experiment.experiments import block_sweep, width_sweep
from mac_experiment.metrics.bitmatch import BitMatchBuilder
from mac_experiment.results.bitmatch_analysis import analyze_bitmatch
from mac_experiment.results.reader import iter_records, load_summary
from mac_experiment.results.schema import BITMATCH_SUMMARY_VERSION, dumps_strict, loads_strict

ONE = 0x3F800000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
NAN = 0x7FC00000


def make_record(sample_index=0, x=1, s_total=None, r_bits=ONE, seed=1, record_type="snapshot"):
    s_total = Dyadic.zero() if s_total is None else s_total
    return {
        "record_type": record_type,
        "mode": "block_sweep",
        "sample_id": f"fp8_e4m3|laplace|seed{seed}|{sample_index:05d}",
        "seed": seed,
        "sample_index": sample_index,
        "format": "fp8_e4m3",
        "distribution": "laplace",
        "B": x,
        "W": 24,
        "S": s_total.to_dict(),
        "r_bits": r_bits,
        "status": "ok",
    }


def build_cell(records):
    builder = BitMatchBuilder("block_sweep", "B")
    for record in records:
        builder.add(record)
    return builder.finalize({"run_id": "unit"})["cells"][0]


def test_exact_match_and_zero_distance():
    cell = build_cell([make_record(s_total=Dyadic(1, 0), r_bits=ONE)])
    for pair in ("fp32", "bf16", "aux_double_rounding", "aux_accum_bf16"):
        payload = cell["pairs"][pair]
        assert payload["counts"]["n_total"] == 1
        assert payload["counts"]["bit_match"] == 1
        assert payload["bit_match_rate"] == 1.0
        assert payload["bit_mismatch_rate"] == 0.0
        assert payload["conditional_ulp"]["status"] == "no_mismatch_samples"
        assert payload["conditional_ulp"]["p50"] is None
        assert payload["distance_buckets"]["buckets"]["0"]["count"] == 1
        assert payload["distance_buckets"]["buckets"]["0"]["rate"] == 1.0
        assert payload["distance_buckets"]["buckets"]["1"]["count"] == 0


def test_fp32_mismatch_bf16_match():
    s_total = Dyadic((1 << 8) + 1, -8)
    assert round_dyadic_to_fp32(s_total)[0].bits == 0x3F808000
    assert round_dyadic_to_bf16(s_total)[0].bits == 0x3F80
    cell = build_cell([make_record(s_total=s_total, r_bits=ONE)])
    fp32 = cell["pairs"]["fp32"]
    assert fp32["counts"]["bit_mismatch"] == 1
    assert fp32["conditional_ulp"]["p50"] == 0x8000
    assert fp32["conditional_ulp"]["max"] == 0x8000
    bf16 = cell["pairs"]["bf16"]
    assert bf16["counts"]["bit_match"] == 1
    assert bf16["conditional_ulp"]["status"] == "no_mismatch_samples"


def test_signed_zero_mismatch_is_distance_zero():
    cell = build_cell([make_record(s_total=Dyadic.zero(), r_bits=NEG_ZERO)])
    for pair in ("fp32", "bf16"):
        payload = cell["pairs"][pair]
        assert payload["counts"]["bit_mismatch"] == 1
        assert payload["counts"]["signed_zero_mismatch"] == 1
        assert payload["counts"]["n_finite_mismatch"] == 1
        assert payload["conditional_ulp"]["p50"] == 0
        assert payload["conditional_ulp"]["max"] == 0
        assert payload["distance_buckets"]["buckets"]["0"]["count"] == 1


def test_nonfinite_event_classification():
    huge = Dyadic(1, 200)
    records = [
        make_record(0, s_total=Dyadic(1, 0), r_bits=POS_INF),
        make_record(1, s_total=huge, r_bits=ONE),
        make_record(2, s_total=huge, r_bits=POS_INF),
        make_record(3, s_total=huge, r_bits=NEG_INF),
        make_record(4, s_total=Dyadic(1, 0), r_bits=NAN),
    ]
    cell = build_cell(records)
    fp32 = cell["pairs"]["fp32"]
    assert fp32["counts"]["n_total"] == 5
    assert fp32["events"] == {
        "reference_finite_result_nonfinite": 1,
        "reference_nonfinite_result_finite": 1,
        "both_inf_same_sign": 1,
        "both_inf_opposite_sign": 1,
        "any_nan": 1,
    }
    assert fp32["counts"]["n_finite_pairs"] == 0
    assert fp32["distance_buckets"]["buckets"]["0"]["rate"] is None
    assert fp32["counts"]["bit_match"] == 1  # both-inf same sign record
    assert fp32["counts"]["bit_mismatch"] == 4


def test_conditional_quantiles_and_buckets():
    records = [make_record(0, s_total=Dyadic.zero(), r_bits=NEG_ZERO)]
    for offset in (1, 2, 3, 4, 10):
        records.append(make_record(offset, s_total=Dyadic(1, 0), r_bits=ONE + offset))
    cell = build_cell(records)
    payload = cell["pairs"]["fp32"]
    assert payload["counts"]["n_finite_pairs"] == 6
    assert payload["counts"]["n_finite_mismatch"] == 6
    conditional = payload["conditional_ulp"]
    assert conditional["status"] == "ok"
    assert conditional["n"] == 6
    assert conditional["p50"] == 2
    assert conditional["p90"] == 10
    assert conditional["p99"] == 10
    assert conditional["max"] == 10
    buckets = payload["distance_buckets"]["buckets"]
    assert [buckets[label]["count"] for label in ("0", "1", "2", "3-4", ">=5")] == [1, 1, 1, 2, 1]
    assert buckets["3-4"]["rate"] == pytest.approx(2 / 6)
    assert payload["counts"]["signed_zero_mismatch"] == 1


def test_grouping_by_x_and_seed():
    records = [
        make_record(0, x=1, seed=1),
        make_record(1, x=1, seed=2),
        make_record(0, x=2, seed=1),
    ]
    builder = BitMatchBuilder("block_sweep", "B")
    for record in records:
        builder.add(record)
    summary = builder.finalize({"run_id": "unit"})
    assert summary["x_values"] == [1, 2]
    by_x = {(cell["x"]): cell for cell in summary["cells"]}
    assert by_x[1]["n_records"] == 2
    assert by_x[2]["n_records"] == 1
    fp32_seed_counts = by_x[1]["pairs"]["fp32"]["by_seed"]
    assert fp32_seed_counts["1"]["n_total"] == 1
    assert fp32_seed_counts["2"]["n_total"] == 1


def _block_run(tmp_path, b_list=(1, 2), samples=1):
    config = {
        "mode": "block_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": samples,
        "W": 24,
        "B_list": list(b_list),
        "output_root": str(tmp_path / "results"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return block_sweep.run_from_file(path, verbose=False)


def test_real_run_counts_match_independent_recomputation(tmp_path):
    run_dir = _block_run(tmp_path)
    out = analyze_bitmatch(run_dir, out_dir=tmp_path / "bitmatch", write_plots=False)
    summary = load_summary(out)
    assert summary["summary_version"] == BITMATCH_SUMMARY_VERSION
    assert summary["n_snapshots_analyzed"] == summary["n_records_read"]

    groups = {}
    for record in iter_records(run_dir):
        if record.get("record_type") != "snapshot":
            continue
        groups.setdefault(record["B"], []).append(record)
    for cell in summary["cells"]:
        records = groups[cell["x"]]
        matches = 0
        for record in records:
            o32, _ = round_dyadic_to_fp32(Dyadic.from_dict(record["S"]))
            if o32.bits == record["r_bits"]:
                matches += 1
        assert cell["pairs"]["fp32"]["counts"]["n_total"] == len(records)
        assert cell["pairs"]["fp32"]["counts"]["bit_match"] == matches
        assert cell["pairs"]["bf16"]["counts"]["n_total"] == len(records)


def test_w_sweep_duplicates_are_not_pooled(tmp_path):
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
    run_dir = width_sweep.run_from_file(path, verbose=False)
    out = analyze_bitmatch(run_dir, out_dir=tmp_path / "bitmatch_w", write_plots=False)
    summary = load_summary(out)
    assert summary["x_name"] == "W"
    assert len(summary["cells"]) == 24
    for cell in summary["cells"]:
        assert cell["n_records"] == 2
        assert cell["pairs"]["fp32"]["counts"]["n_total"] == 2

    r_bits_by_sample = {}
    for record in iter_records(run_dir):
        if record.get("record_type") != "snapshot":
            continue
        r_bits_by_sample.setdefault(record["sample_id"], set()).add(record["r_bits"])
    assert r_bits_by_sample
    assert all(len(values) == 1 for values in r_bits_by_sample.values()), "r_B must not depend on W"


def test_json_csv_plots_source_untouched_and_isolation(tmp_path):
    run_dir = _block_run(tmp_path)
    before = {path.name: path.stat().st_mtime_ns for path in run_dir.iterdir()}
    out = analyze_bitmatch(run_dir, out_dir=tmp_path / "bitmatch_full", write_plots=True)
    after = {path.name: path.stat().st_mtime_ns for path in run_dir.iterdir()}
    assert before == after

    summary = load_summary(out)
    assert loads_strict(dumps_strict(summary)) == summary
    csv_text = (out / "summary.csv").read_text(encoding="utf-8")
    assert ",fp32," in csv_text and ",bf16," in csv_text
    assert "conditional_ulp_p99" in csv_text and "bucket_0_rate" in csv_text

    manifest = json.loads((out / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_kind"] == "bitmatch_ulp"
    assert manifest["source"]["files"]["samples.jsonl.gz"]["sha256"].startswith("sha256:")
    assert manifest["policy"]["subnormals"] == "kept (no FTZ)"
    assert "aux_double_rounding" in manifest["aux_pairs"]

    plots = sorted((out / "plots").glob("*.png"))
    assert len(plots) == 2 * 3  # (fp32, bf16) x 3 figure families
    assert len(list((out / "plots").glob("*.svg"))) == len(plots)
    assert all(path.stat().st_size > 1000 for path in plots)

    with pytest.raises(FileExistsError):
        analyze_bitmatch(run_dir, out_dir=tmp_path / "bitmatch_full", write_plots=False)
    with pytest.raises(ValueError):
        analyze_bitmatch(run_dir, out_dir=run_dir / "nested", write_plots=False)


def test_bitmatch_entry_does_not_load_sampling_or_engine(tmp_path):
    run_dir = _block_run(tmp_path)
    out = tmp_path / "offline"
    script = (
        "import sys\n"
        "from mac_experiment.cli import main\n"
        f"code = main(['bitmatch', '--run', {str(run_dir)!r}, '--out', {str(out)!r}, '--no-plots'])\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('mac_experiment.sampling') or name.startswith('mac_experiment.execution')\n"
        ")\n"
        "print('RETURN', code)\n"
        "print('LOADED', ','.join(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "RETURN 0" in result.stdout
    assert "LOADED \n" in result.stdout or "LOADED\n" in result.stdout, result.stdout
    assert (out / "summary.json").exists()
