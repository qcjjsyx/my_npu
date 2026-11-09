"""Batch runner tests: unit helpers plus a tiny three-experiment end-to-end."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_widthselect_all.py"

_spec = importlib.util.spec_from_file_location("run_widthselect_all", SCRIPT)
batch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(batch)


def test_progress_parsing_and_helpers():
    assert batch.parse_progress("[width-align] 3/6 matrix pairs, 2 format(s)") == (3, 6)
    assert batch.parse_progress("[width-acc] 12/40 matrix pairs") == (12, 40)
    assert batch.parse_progress("[width-align] completed: x") is None
    assert batch.format_duration(0) == "0:00"
    assert batch.format_duration(61.9) == "1:01"
    assert batch._median([3.0, 1.0, 2.0]) == 2.0
    assert batch._median([1.0, 2.0]) == 1.5
    assert batch._median([]) is None


def test_summarize_analysis_report_only_and_criteria():
    report_only = {
        "cells": [
            {
                "metrics": {
                    "bit_match_rate": 1.0,
                    "counts": {"n_nonfinite_candidate": 0},
                },
                "state_marker_count": 0,
            },
            {
                "metrics": {
                    "bit_match_rate": 0.5,
                    "counts": {"n_nonfinite_candidate": 3},
                },
                "state_marker_count": 2,
            },
        ],
        "recommendation": None,
    }
    lines = batch.summarize_analysis(report_only)
    assert any("bit_match min/med/max" in line for line in lines)
    assert any("nonfinite candidates=3" in line for line in lines)
    assert any("report-only" in line for line in lines)

    with_criteria = {
        "cells": [],
        "recommendation": {
            "criteria": {"min_bit_match_rate": 0.9},
            "by_group": [
                {
                    "format": "tf32",
                    "distribution_pair": "gaussian|gaussian",
                    "K": 128,
                    "tested": [{"candidate": {"w_align": 24, "w_acc": None}}],
                    "minimal_passing_candidate": 24,
                },
                {
                    "format": "tf32",
                    "distribution_pair": "laplace|laplace",
                    "K": 128,
                    "tested": [{"candidate": {"w_align": None, "w_acc": 30}}],
                    "passing_candidates": [{"w_align": 21, "w_acc": 30}],
                },
            ],
        },
    }
    lines = batch.summarize_analysis(with_criteria)
    assert any("2/2 groups" in line for line in lines)
    assert any("w_align=24" in line for line in lines)
    assert any("w_align=21,w_acc=30" in line for line in lines)


def _write_configs(root: Path) -> dict[str, Path]:
    shared = {
        "M": 1,
        "N": 32,
        "K_list": [32],
        "formats": ["tf32"],
        "distribution_pairs": [
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"}
        ],
        "seeds": [1],
        "pairs_per_seed": 1,
        "output_root": str(root / "runs"),
    }
    payloads = {
        "align": {
            **shared,
            "mode": "widthselect_align",
            "W_align_list": [21],
            "run_name": "batch_align",
        },
        "acc": {
            **shared,
            "mode": "widthselect_acc",
            "W_acc_list": [16],
            "run_name": "batch_acc",
        },
        "joint": {
            **shared,
            "mode": "widthselect_joint",
            "joint_pairs": [{"w_align": 21, "w_acc": 16}],
            "run_name": "batch_joint",
        },
    }
    paths = {}
    for name, payload in payloads.items():
        path = root / f"batch_{name}.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        paths[name] = path
    return paths


def _run_script(configs: dict[str, Path], *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plain",
            "--align-config",
            str(configs["align"]),
            "--acc-config",
            str(configs["acc"]),
            "--joint-config",
            str(configs["joint"]),
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_batch_script_end_to_end(tmp_path):
    configs = _write_configs(tmp_path)
    result = _run_script(configs)
    assert result.returncode == 0, result.stderr
    assert "[1/3] widthselect_align" in result.stdout
    assert "[2/3] widthselect_acc" in result.stdout
    assert "[3/3] widthselect_joint" in result.stdout
    assert "=== summary: 3/3 succeeded" in result.stdout
    assert "1/1 pairs" in result.stderr
    assert result.stderr.count("done in") >= 6
    for name in ("batch_align", "batch_acc", "batch_joint"):
        assert (tmp_path / "runs" / name / "manifest.json").exists()
    analysis_dirs = sorted((tmp_path / "analysis").iterdir())
    assert len(analysis_dirs) == 3
    assert all((path / "summary.json").exists() for path in analysis_dirs)

    # a repeat without --overwrite must fail on the existing run directories
    repeated = _run_script(configs)
    assert repeated.returncode == 1
    assert "already exists" in repeated.stderr

    # --overwrite removes the configured runs (guarded by schema) and succeeds
    overwritten = _run_script(configs, "--overwrite")
    assert overwritten.returncode == 0, overwritten.stderr
    assert "overwrite" in overwritten.stdout
