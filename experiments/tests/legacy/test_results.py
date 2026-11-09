import json

import pytest
import yaml

from mac_experiment.experiments import block_sweep, width_sweep
from mac_experiment.experiments.config import DEFAULT_B_LIST, WIDTH_SCAN, load_config
from mac_experiment.results.reader import assert_config_matches, iter_records, load_manifest, load_summary
from mac_experiment.results.reanalyze import compare_summaries, reanalyze
from mac_experiment.results.schema import config_hash
from mac_experiment.results.writer import create_run_dir


def write_config(tmp_path, **overrides):
    config = {
        "mode": "width_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": 2,
        "B": 2,
        "output_root": str(tmp_path / "results"),
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_config_validation_errors(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, mode="block_sweep"))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, W=20))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, B=None))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, formats=["fp16"]))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, distributions=["gaussian"]))
    block = load_config(
        write_config(tmp_path, mode="block_sweep", B=None, W=24, B_list=None)
    )
    assert block.B_list == DEFAULT_B_LIST
    assert block.W == 24
    assert block.windows == [24]
    width = load_config(write_config(tmp_path))
    assert width.windows == WIDTH_SCAN
    assert width.snapshots == [2]


def test_tiny_width_run_reanalyze_and_manifest(tmp_path):
    config_path = write_config(tmp_path)
    run_dir = width_sweep.run_from_file(config_path, verbose=False)
    manifest = load_manifest(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["counts"]["samples"] == 2
    assert manifest["counts"]["records"] == 2 * len(WIDTH_SCAN)
    assert manifest["config_hash"] == config_hash(manifest["config"])
    summary = load_summary(run_dir)
    assert summary["x_values"] == WIDTH_SCAN
    assert len(summary["cells"]) == len(WIDTH_SCAN)
    records = list(iter_records(run_dir))
    assert len(records) == 2 * len(WIDTH_SCAN)
    result = reanalyze(run_dir)
    assert result["matches_original_summary"], result["differences"]
    assert result["n_records_read"] == len(records)


def test_reanalysis_detects_tampering(tmp_path):
    run_dir = width_sweep.run_from_file(write_config(tmp_path), verbose=False)
    summary = load_summary(run_dir)
    summary["cells"][0]["comparisons"]["total"]["metrics"]["match_rate"]["value"] = 0.123
    assert compare_summaries(summary, load_summary(run_dir)) != []


def test_run_dir_refuses_overwrite(tmp_path):
    run_dir = width_sweep.run_from_file(write_config(tmp_path), verbose=False)
    with pytest.raises(FileExistsError):
        create_run_dir(run_dir.parent, run_dir.name)


def test_manifest_config_mismatch_is_rejected(tmp_path):
    run_dir = width_sweep.run_from_file(write_config(tmp_path), verbose=False)
    manifest = load_manifest(run_dir)
    other = dict(manifest["config"])
    other["samples_per_seed"] = 99
    with pytest.raises(ValueError):
        assert_config_matches(manifest, other)


def test_block_run_prefix_stability_across_runs(tmp_path):
    short = block_sweep.run_from_file(
        write_config(tmp_path, mode="block_sweep", B=None, W=24, B_list=[1, 2]), verbose=False
    )
    long = block_sweep.run_from_file(
        write_config(tmp_path, mode="block_sweep", B=None, W=24, B_list=[1, 2, 4]), verbose=False
    )
    short_records = {(r["sample_id"], r["B"]): r for r in iter_records(short)}
    long_records = {(r["sample_id"], r["B"]): r for r in iter_records(long)}
    assert short_records
    for key, record in short_records.items():
        assert key in long_records
        assert long_records[key] == record


def test_summary_csv_written(tmp_path):
    run_dir = width_sweep.run_from_file(write_config(tmp_path), verbose=False)
    csv_path = run_dir / "summary.csv"
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "comparison" in header and "metric" in header


def test_records_are_strict_json(tmp_path):
    run_dir = width_sweep.run_from_file(write_config(tmp_path), verbose=False)
    for record in iter_records(run_dir):
        text = json.dumps(record, allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text
