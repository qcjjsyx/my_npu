"""Shared run driver: sample loop, manifest, records, summary.

The W/B sweep modules only choose the scan axes; the numerical core lives in
the execution engine and is never duplicated here.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..execution.engine import SampleEngine
from ..formats import get_format
from ..metrics.aggregate import SummaryBuilder
from ..results.reader import assert_config_matches, load_manifest
from ..results.schema import SCHEMA_VERSION, SUMMARY_VERSION, config_hash
from ..results.schema import source_digest as _source_digest
from ..results.writer import (
    JsonlSink,
    SAMPLES_NAME,
    SUMMARY_CSV_NAME,
    SUMMARY_NAME,
    create_run_dir,
    read_json,
    write_json,
    write_summary_csv,
)
from ..sampling.sampler import Sampler
from ..version import (
    ACCUMULATION_POLICY,
    ALIGNMENT_POLICY,
    INPUT_FTZ,
    OUTPUT_SUBNORMAL_POLICY,
    PACKAGE_VERSION,
    STREAM_VERSION,
)
from .config import RunConfig

def source_digest() -> str:
    """Re-exported lightweight implementation (``results.schema.source_digest``)."""
    return _source_digest()


def execute(config: RunConfig, verbose: bool = True) -> Path:
    formats = {name: get_format(name) for name in config.formats}
    samplers = {
        (fmt_name, dist): Sampler(formats[fmt_name], dist, config.cancellation_sign_mode)
        for fmt_name in config.formats
        for dist in config.distributions
    }
    if config.mode == "width_sweep":
        x_name = "W"
    else:
        x_name = "B"
    windows = config.windows
    snapshots = config.snapshots

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = source_digest()
    config_payload = config.effective()
    chash = config_hash(config_payload)
    run_id = config.run_name or f"{config.mode}_{stamp}_{chash[:8]}"
    run_dir = create_run_dir(config.output_root, run_id)
    if verbose:
        print(f"[run] {run_id} -> {run_dir}", file=sys.stderr)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "summary_version": SUMMARY_VERSION,
        "stream_version": STREAM_VERSION,
        "package_version": PACKAGE_VERSION,
        "source_digest": digest,
        "created_utc": created,
        "run_id": run_id,
        "mode": config.mode,
        "x_name": x_name,
        "fixed": config.fixed,
        "windows": list(windows),
        "snapshots": list(snapshots),
        "config": config_payload,
        "config_hash": chash,
        "config_source": config.describe(),
        "policies": {
            "alignment": ALIGNMENT_POLICY,
            "accumulation": ACCUMULATION_POLICY,
            "input_ftz": INPUT_FTZ,
            "output_subnormal": OUTPUT_SUBNORMAL_POLICY,
            "window_container_bits": "W + 7 signed; overflow raises and marks the cell",
        },
        "formats": {name: fmt.metadata() for name, fmt in formats.items()},
        "distributions": {
            f"{fmt_name}|{dist}": sampler.metadata() for (fmt_name, dist), sampler in samplers.items()
        },
        "expected": {
            "samples_per_cell": len(config.seeds) * config.samples_per_seed,
            "cells": len(config.formats) * len(config.distributions) * len(config.x_values),
            "records": (
                len(config.formats)
                * len(config.distributions)
                * len(config.seeds)
                * config.samples_per_seed
                * (len(windows) if config.mode == "width_sweep" else len(snapshots))
            ),
        },
        "counts": {},
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    builder = SummaryBuilder(mode=config.mode, x_name=x_name)
    n_records = 0
    n_aborts = 0
    n_samples = 0
    n_samples_aborted = 0
    n_samples_nonfinite = 0
    total_samples = len(config.formats) * len(config.distributions) * len(config.seeds) * config.samples_per_seed
    done = 0
    with JsonlSink(run_dir / SAMPLES_NAME) as sink:
        for fmt_name in config.formats:
            for dist in config.distributions:
                engine = SampleEngine(
                    formats[fmt_name],
                    samplers[(fmt_name, dist)],
                    windows,
                    snapshots,
                    config.mode,
                    record_block_rho=config.record_block_rho,
                )
                for seed in config.seeds:
                    for sample_index in range(config.samples_per_seed):
                        outcome = engine.run(seed, sample_index)
                        for record in outcome.records:
                            sink.write(record)
                            builder.add(record)
                            n_records += 1
                            if record["record_type"] == "abort":
                                n_aborts += 1
                        n_samples += 1
                        if outcome.status == "aborted_window_overflow":
                            n_samples_aborted += 1
                        elif outcome.status == "nonfinite_fp32":
                            n_samples_nonfinite += 1
                        done += 1
                        if verbose and (done % 20 == 0 or done == total_samples):
                            print(f"[run] {done}/{total_samples} samples", file=sys.stderr)

    summary_base = {
        "schema_version": SCHEMA_VERSION,
        "summary_version": SUMMARY_VERSION,
        "run_id": run_id,
        "mode": config.mode,
        "config": config_payload,
        "config_hash": chash,
        "fixed": config.fixed,
        "created_utc": created,
        "source_digest": digest,
    }
    summary = builder.finalize(summary_base)
    write_json(run_dir / SUMMARY_NAME, summary)
    write_summary_csv(run_dir / SUMMARY_CSV_NAME, summary)

    manifest["counts"] = {
        "samples": n_samples,
        "samples_aborted": n_samples_aborted,
        "samples_nonfinite": n_samples_nonfinite,
        "records": n_records,
        "abort_records": n_aborts,
    }
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    if verbose:
        print(f"[run] completed: {run_dir}", file=sys.stderr)
    return run_dir


def load_completed_run(run_dir: str | Path, expected_config: dict | None = None) -> dict:
    manifest = load_manifest(run_dir)
    if manifest.get("status") != "completed":
        raise ValueError(f"run {run_dir} is not completed (status={manifest.get('status')!r})")
    if expected_config is not None:
        assert_config_matches(manifest, expected_config)
    return manifest


def read_summary_file(run_dir: str | Path) -> dict:
    return read_json(Path(run_dir) / SUMMARY_NAME)
