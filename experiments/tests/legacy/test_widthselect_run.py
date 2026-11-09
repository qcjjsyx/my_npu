"""End-to-end run/analyze/replay tests and isolation guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mac_experiment.results.reader import load_manifest
from mac_experiment.results.schema import dumps_strict
from mac_experiment.widthselect import runner
from mac_experiment.widthselect.analysis import analyze_width_run
from mac_experiment.widthselect.config import WidthSelectConfig, load_widthselect_config
from mac_experiment.widthselect.io import iter_width_inputs, iter_width_samples
from mac_experiment.widthselect.replay import replay_width_run

SAMPLE_ID = "width|gaussian|gaussian|seed1|00000"


def _config(tmp_path: Path, **overrides) -> WidthSelectConfig:
    base = dict(
        mode="widthselect_align",
        formats=["mx_fp8", "tf32"],
        distribution_pairs=[
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"}
        ],
        seeds=[1],
        pairs_per_seed=1,
        M=1,
        N=32,
        K_list=[32, 64],
        W_align_list=[21, 25],
        output_root=str(tmp_path / "runs"),
        run_name="width_smoke",
    )
    base.update(overrides)
    return WidthSelectConfig(**base)


def test_end_to_end_run_analyze_replay(tmp_path):
    run_dir = runner.run(_config(tmp_path), verbose=False)
    manifest = load_manifest(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["schema_version"] == "mac-width-v1"
    assert manifest["source"]["kind"] == "fresh"
    assert manifest["counts"]["matrix_pairs"] == 1
    assert manifest["counts"]["base_snapshot_records"] == 4
    assert manifest["counts"]["width_snapshot_records"] == 8
    assert set(manifest["file_identity"]) == {"width_inputs.jsonl.gz", "width_samples.jsonl.gz"}
    assert manifest["file_identity"]["width_samples.jsonl.gz"]["sha256"].startswith("sha256:")
    assert manifest["file_identity"]["width_samples.jsonl.gz"]["size"] > 0

    inputs = list(iter_width_inputs(run_dir))
    assert len(inputs) == 1
    record = inputs[0]
    assert record["sample_id"] == SAMPLE_ID
    assert sorted(record["formats"]) == ["mx_fp8", "tf32"]
    assert len(record["x_fp32_bits"]) == 1 * 64
    assert len(record["w_fp32_bits"]) == 64 * 32
    # one FP32 pair shared by both target formats (payloads differ by format)
    assert record["formats"]["mx_fp8"]["x"] != record["formats"]["tf32"]["x"]

    samples = list(iter_width_samples(run_dir))
    assert len(samples) == 12
    base = [item for item in samples if item["record_type"] == "widthselect_base_snapshot"]
    width = [item for item in samples if item["record_type"] == "widthselect_align_snapshot"]
    assert len(base) == 4 and len(width) == 8
    assert {(item["K"], item["format"]) for item in base} == {
        (32, "mx_fp8"),
        (64, "mx_fp8"),
        (32, "tf32"),
        (64, "tf32"),
    }
    for item in width:
        assert len(item["T"]) == 32
        assert len(item["Y_bits"]) == 32
        assert set(item["events_cumulative"]) == {
            "products_total",
            "products_nonzero",
            "products_changed",
            "products_zeroed",
            "dot16_total",
            "dot16_all_zero",
            "dot16_with_changed_product",
            "dot16_qb_diff",
        }

    out_dir = analyze_width_run(run_dir, out_dir=tmp_path / "analysis")
    summary = yaml.safe_load((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "mac-width-analysis-v1"
    assert len(summary["cells"]) == 8
    assert len(summary["by_pair"]) == 8
    assert summary["recommendation"] is None
    for cell in summary["cells"]:
        counts = cell["metrics"]["counts"]
        assert counts["n_total"] == 32
        assert 0 <= counts["n_bit_match"] <= 32
        assert cell["t_minus_s"]["relative_l2"]["status"] in ("ok", "undefined_zero_reference")
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "analysis_manifest.json").exists()

    replay = replay_width_run(
        run_dir, sample_id=SAMPLE_ID, row=0, col=0, K=64, w_align=21, format="tf32"
    )
    assert replay["saved_snapshot_matches"] is True
    assert replay["run_kind"] == "widthselect_align"
    assert len(replay["blocks"]) == 4
    assert replay["blocks"][0]["terms"]
    assert isinstance(replay["blocks"][0]["terms"][0]["n"], int)
    assert replay["final_exact_T"] == replay["blocks"][-1]["exact_T_after_block"]


def test_plot_annotations():
    from mac_experiment.widthselect.plots import (
        _panel_totals,
        all_nonfinite_curves,
        hollow_points_note,
        series_points,
    )

    def cell(finite, mismatch, nonfinite):
        return {
            "metrics": {
                "counts": {
                    "n_finite_pairs": finite,
                    "n_finite_mismatch": mismatch,
                    "n_nonfinite_candidate": nonfinite,
                }
            }
        }

    cells = [cell(0, 0, 100), cell(20, 0, 80)]
    assert _panel_totals(cells) == (20, 0, 180)
    note = hollow_points_note(cells)
    assert "no mismatch samples (drawn at 0)" in note
    assert "finite pairs = 20" in note and "nonfinite = 180" in note

    # null metrics are drawn at 0 with a hollow marker mask (summary keeps null)
    ys, hollow = series_points([1.0, None, 0.0, None])
    assert ys == [1.0, 0.0, 0.0, 0.0]
    assert hollow == [False, True, False, True]

    by_key = {
        ("tf32", (27,), 2): cell(0, 0, 64),
        ("tf32", (27,), 4): cell(0, 0, 64),
        ("tf32", (30,), 2): cell(64, 1, 0),
        ("tf32", (30,), 4): cell(0, 0, 64),
    }
    labels = all_nonfinite_curves("acc", "tf32", [(27,), (30,)], [2, 4], by_key)
    assert labels == ["W_acc=27"]


def test_analysis_plots(tmp_path):
    pytest.importorskip("matplotlib")
    run_dir = runner.run(
        _config(
            tmp_path,
            run_name="width_plots",
            distribution_pairs=[
                {"activation_distribution": "gaussian", "weight_distribution": "gaussian"},
                {"activation_distribution": "laplace", "weight_distribution": "laplace"},
            ],
        ),
        verbose=False,
    )
    out_dir = analyze_width_run(run_dir, out_dir=tmp_path / "plots_analysis", write_plots=True)
    plot_dir = out_dir / "plots"
    names = sorted(path.name for path in plot_dir.glob("*.png"))
    # one figure per synthetic distribution, 3 metric rows x format columns
    assert names == ["gaussian_metrics.png", "laplace_metrics.png"]
    assert all((plot_dir / name).stat().st_size > 1000 for name in names)
    manifest = json.loads((out_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["plots"]) == names
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert sorted(summary["plots"]) == names


def test_run_is_deterministic_across_run_ids(tmp_path):
    first = runner.run(_config(tmp_path, run_name="first"), verbose=False)
    second = runner.run(_config(tmp_path, run_name="second"), verbose=False)
    first_inputs = list(iter_width_inputs(first))
    second_inputs = list(iter_width_inputs(second))
    assert first_inputs[0]["x_fp32_bits"] == second_inputs[0]["x_fp32_bits"]
    assert first_inputs[0]["w_fp32_bits"] == second_inputs[0]["w_fp32_bits"]

    def _t_by_key(run_dir):
        result = {}
        for record in iter_width_samples(run_dir):
            if record["record_type"] == "widthselect_align_snapshot":
                result[(record["format"], record["K"], record["w_align"])] = record["T"]
        return result

    assert _t_by_key(first) == _t_by_key(second)


def test_config_loader_defaults_and_rejections(tmp_path):
    smoke = Path(__file__).resolve().parents[2] / "configs" / "legacy" / "widthselect_smoke.yaml"
    config = load_widthselect_config(smoke)
    assert config.mode == "widthselect_align"
    assert config.formats == ["mx_fp8", "tf32"]
    assert len(config.distribution_pairs) == 6
    assert config.pass_criteria == {
        "min_bit_match_rate": 0.5,
        "max_ulp_p99": 4,
        "nonfinite_policy": "ignore",
    }

    def _write(payload: dict) -> Path:
        path = tmp_path / f"cfg_{len(list(tmp_path.glob('cfg_*.yaml')))}.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    base = {
        "mode": "widthselect_align",
        "M": 1,
        "N": 32,
        "K_list": [32],
        "W_align_list": [21],
    }
    with pytest.raises(ValueError, match="unknown configuration keys"):
        load_widthselect_config(_write({**base, "source_run": "results_mx/whatever"}))
    with pytest.raises(ValueError, match="unknown configuration keys"):
        load_widthselect_config(_write({**base, "B": 2}))
    with pytest.raises(ValueError, match="unknown input distribution"):
        load_widthselect_config(
            _write({**base, "distribution_pairs": [
                {"activation_distribution": "cauchy", "weight_distribution": "gaussian"}
            ]})
        )
    with pytest.raises(ValueError, match="custom distribution parameters"):
        load_widthselect_config(
            _write({**base, "distribution_pairs": [
                {
                    "activation_distribution": "gaussian",
                    "weight_distribution": "gaussian",
                    "activation_parameters": {"std": 2},
                }
            ]})
        )
    with pytest.raises(ValueError):
        load_widthselect_config(_write({**base, "K_list": [24]}))
    with pytest.raises(ValueError):
        load_widthselect_config(_write({**base, "pass_criteria": {"unknown": 1}}))

    acc = load_widthselect_config(
        _write(
            {
                "mode": "widthselect_acc",
                "M": 1,
                "N": 32,
                "K_list": [32],
                "W_acc_list": [30],
            }
        )
    )
    assert acc.W_acc_list == [30]


def test_formal_templates_load():
    configs_dir = Path(__file__).resolve().parents[2] / "configs" / "legacy"
    align = load_widthselect_config(configs_dir / "widthselect_align_formal_template.yaml")
    assert align.mode == "widthselect_align"
    assert align.W_align_list and all(1 <= value <= 60 for value in align.W_align_list)
    assert align.W_acc_list == [] and align.joint_pairs == []

    acc = load_widthselect_config(configs_dir / "widthselect_acc_formal_template.yaml")
    assert acc.mode == "widthselect_acc"
    assert acc.W_acc_list and all(8 <= value <= 60 for value in acc.W_acc_list)
    assert acc.W_align_list == [] and acc.joint_pairs == []

    joint = load_widthselect_config(configs_dir / "widthselect_joint_formal_template.yaml")
    assert joint.mode == "widthselect_joint"
    assert joint.joint_pairs and all(
        set(pair) == {"w_align", "w_acc"} for pair in joint.joint_pairs
    )
    assert joint.W_align_list == [] and joint.W_acc_list == []

    # one shared input identity across the three isolated templates
    assert (align.M, align.N, align.K_list, align.formats) == (
        acc.M,
        acc.N,
        acc.K_list,
        acc.formats,
    )
    assert (align.M, align.N, align.K_list, align.formats) == (
        joint.M,
        joint.N,
        joint.K_list,
        joint.formats,
    )


def test_acc_and_joint_modes_end_to_end(tmp_path):
    acc = WidthSelectConfig(
        mode="widthselect_acc",
        formats=["tf32"],
        distribution_pairs=[
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"}
        ],
        seeds=[1],
        pairs_per_seed=1,
        M=1,
        N=32,
        K_list=[32, 64],
        W_acc_list=[12, 16],
        output_root=str(tmp_path / "runs"),
        run_name="width_acc_smoke",
        pass_criteria={"min_bit_match_rate": 0.0, "nonfinite_policy": "ignore"},
    )
    acc_dir = runner.run(acc, verbose=False)
    acc_manifest = load_manifest(acc_dir)
    assert acc_manifest["run_kind"] == "widthselect_acc"
    assert acc_manifest["counts"]["width_snapshot_records"] == 4
    assert acc_manifest["policies"]["acc_format"]["status"] == "hwacc_v1_confirmed_implemented"
    acc_records = [
        item for item in iter_width_samples(acc_dir) if item["record_type"] == "widthselect_acc_snapshot"
    ]
    assert len(acc_records) == 4
    for item in acc_records:
        assert len(item["C_state"]) == 32 and len(item["C_exact"]) == 32
        assert all(set(state) == {"E_code", "M", "w_acc"} for state in item["C_state"])
        assert set(item["events_cumulative"]) == {
            "acc_updates",
            "acc_input_zero",
            "acc_input_absorbed",
            "acc_state_bits_dropped",
            "acc_input_bits_dropped",
            "acc_carry",
            "acc_cancellation",
            "acc_overflow",
            "acc_subnormal",
            "acc_marker_propagated",
        }
    analysis_dir = analyze_width_run(acc_dir, out_dir=tmp_path / "acc_analysis")
    summary = yaml.safe_load((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_kind"] == "widthselect_acc_analysis"
    assert len(summary["cells"]) == 4
    assert summary["recommendation"]["by_group"][0]["minimal_passing_candidate"] == 12
    for cell in summary["cells"]:
        assert cell["series_kind"] == "c_minus_s"
        assert cell["t_minus_s"] is None and cell["c_minus_s"] is not None
        assert cell["w_align"] is None and cell["w_acc"] in (12, 16)
    replay = replay_width_run(
        acc_dir, sample_id=SAMPLE_ID, row=0, col=0, K=64, w_acc=16, format="tf32"
    )
    assert replay["saved_snapshot_matches"] is True
    assert replay["run_kind"] == "widthselect_acc"
    assert len(replay["blocks"]) == 4
    first = replay["blocks"][0]
    assert "hw_add" in first and "state_after" in first and "incoming" in first

    joint = WidthSelectConfig(
        mode="widthselect_joint",
        formats=["tf32"],
        distribution_pairs=[
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"}
        ],
        seeds=[1],
        pairs_per_seed=1,
        M=1,
        N=32,
        K_list=[32],
        joint_pairs=[{"w_align": 21, "w_acc": 12}],
        output_root=str(tmp_path / "runs"),
        run_name="width_joint_smoke",
    )
    joint_dir = runner.run(joint, verbose=False)
    joint_manifest = load_manifest(joint_dir)
    assert joint_manifest["run_kind"] == "widthselect_joint"
    assert joint_manifest["counts"]["width_snapshot_records"] == 1
    joint_analysis = analyze_width_run(joint_dir, out_dir=tmp_path / "joint_analysis")
    joint_summary = yaml.safe_load((joint_analysis / "summary.json").read_text(encoding="utf-8"))
    assert joint_summary["analysis_kind"] == "widthselect_joint_analysis"
    cell = joint_summary["cells"][0]
    assert cell["w_align"] == 21 and cell["w_acc"] == 12
    joint_replay = replay_width_run(
        joint_dir, sample_id=SAMPLE_ID, row=0, col=0, K=32, w_align=21, w_acc=12, format="tf32"
    )
    assert joint_replay["saved_snapshot_matches"] is True
    assert joint_replay["blocks"][0]["alignment"] is not None


def test_acc_analysis_detects_corrupted_state(tmp_path):
    import gzip
    import json

    from mac_experiment.widthselect.io import WIDTH_SAMPLES_NAME

    acc = WidthSelectConfig(
        mode="widthselect_acc",
        formats=["tf32"],
        distribution_pairs=[
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"}
        ],
        seeds=[1],
        pairs_per_seed=1,
        M=1,
        N=32,
        K_list=[32],
        W_acc_list=[12],
        output_root=str(tmp_path / "runs"),
        run_name="width_acc_tamper",
    )
    run_dir = runner.run(acc, verbose=False)
    path = run_dir / WIDTH_SAMPLES_NAME
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.readlines()
    target = None
    for index, line in enumerate(lines):
        payload = json.loads(line)
        if payload.get("record_type") == "widthselect_acc_snapshot":
            target = index
            break
    assert target is not None
    payload = json.loads(lines[target])
    payload["C_exact"][0] = {"m": "12345", "e": 0}
    lines[target] = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(lines)
    with pytest.raises(ValueError, match="C_exact integrity failure"):
        analyze_width_run(run_dir, out_dir=tmp_path / "tamper_analysis")


def test_analysis_rejects_historical_schemas(tmp_path):
    fake = tmp_path / "historical"
    fake.mkdir()
    (fake / "manifest.json").write_text(
        dumps_strict({"schema_version": "mac-mx-align-v1", "run_id": "old"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="historical runs are never read"):
        analyze_width_run(fake, out_dir=tmp_path / "out")


def test_replay_rejects_unknown_sample_and_coordinate(tmp_path):
    run_dir = runner.run(_config(tmp_path), verbose=False)
    with pytest.raises(ValueError, match="no input record"):
        replay_width_run(
            run_dir, sample_id="width|nope|nope|seed1|00000", row=0, col=0, K=32, w_align=21, format="tf32"
        )
    with pytest.raises(ValueError, match="outside"):
        replay_width_run(run_dir, sample_id=SAMPLE_ID, row=5, col=0, K=32, w_align=21, format="tf32")
    with pytest.raises(ValueError, match="not a declared snapshot"):
        replay_width_run(run_dir, sample_id=SAMPLE_ID, row=0, col=0, K=48, w_align=21, format="tf32")
