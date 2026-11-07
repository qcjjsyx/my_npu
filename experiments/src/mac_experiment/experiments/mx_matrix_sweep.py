"""MX matrix sweep driver: paired-prefix K scan over one input matrix pair.

Current input pipeline: numeric-domain sampling -> FP32 matrices -> G32 MXFP8
input quantisation -> exact reference / dot16 FP32 accumulation.  Execution
only.  The run directory stores the largest input matrices once per pair
(``inputs.jsonl.gz``: FP32 bits + MX element/scale codes + per-path input
quantisation events/diagnostics) and one exact/FP32 snapshot record per
(pair, K) (``samples.jsonl.gz``).  All output quantisation and metrics are
computed offline by ``results/mx_analysis.py`` so that scale-policy, quantiser
or plot changes never require re-running the MAC.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from ..formats.e8m0 import metadata as e8m0_metadata
from ..matrix.engine import MxMatrixEngine
from ..matrix.inputs import metadata as input_metadata
from ..matrix.quantize import metadata as quantize_metadata
from ..matrix.tensor import element_metadata
from ..results.mx_io import MX_INPUTS_NAME, MX_SAMPLES_NAME
from ..results.schema import mx_config_hash
from ..results.writer import JsonlSink, create_run_dir, write_json
from ..version import (
    MX_ACCUMULATION_POLICY,
    MX_ELEMENT_SEMANTICS,
    MX_INPUT_QUANTIZATION_POLICY,
    MX_INPUT_SAMPLING_VERSION,
    MX_INPUT_STREAM_VERSION,
    MX_OUTPUT_SCALE_POLICY,
    MX_SCALE_SEMANTICS,
    MX_SCHEMA_VERSION,
    PACKAGE_VERSION,
)
from .common import source_digest
from .config import MX_GROUP_LENGTH, MX_MAC_PARALLELISM, MX_MODE, RunConfig


def run(config: RunConfig, verbose: bool = True) -> Path:
    if config.mode != MX_MODE:
        raise ValueError("mx_matrix_sweep driver requires mode=mx_matrix_sweep")
    return execute(config, verbose=verbose)


def run_from_file(path, verbose: bool = True) -> Path:
    from .config import load_config

    return run(load_config(path), verbose=verbose)


def _engine_key(pair: dict) -> tuple:
    return (
        pair["activation_distribution"],
        pair["weight_distribution"],
    )


def execute(config: RunConfig, verbose: bool = True) -> Path:
    engines: dict[tuple, MxMatrixEngine] = {}
    for pair in config.distribution_pairs:
        engines[_engine_key(pair)] = MxMatrixEngine(
            config.M,
            config.N,
            config.K_list,
            pair["activation_distribution"],
            pair["weight_distribution"],
        )

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = source_digest()
    config_payload = config.effective()
    chash = mx_config_hash(config_payload)
    run_id = config.run_name or f"mx_matrix_sweep_{stamp}_{chash[:8]}"
    run_dir = create_run_dir(config.output_root, run_id)
    if verbose:
        print(f"[mx-run] {run_id} -> {run_dir}", file=sys.stderr)

    n_combination_pairs = len(config.distribution_pairs)
    n_pairs = n_combination_pairs * len(config.seeds) * config.pairs_per_seed
    manifest = {
        "schema_version": MX_SCHEMA_VERSION,
        "run_kind": "mx_matrix_sweep",
        "package_version": PACKAGE_VERSION,
        "input_sampling_version": MX_INPUT_SAMPLING_VERSION,
        "input_stream_version": MX_INPUT_STREAM_VERSION,
        "input_quantization_policy": MX_INPUT_QUANTIZATION_POLICY,
        "source_digest": digest,
        "created_utc": created,
        "run_id": run_id,
        "mode": MX_MODE,
        "x_name": "K",
        "fixed": config.fixed,
        "K_list": list(config.K_list),
        "B_list": [k // MX_MAC_PARALLELISM for k in config.K_list],
        "config": config_payload,
        "config_hash": chash,
        "config_source": config.describe(),
        "policies": {
            "input_sampling": MX_INPUT_SAMPLING_VERSION,
            "input_stream_version": MX_INPUT_STREAM_VERSION,
            "input_pipeline": (
                "numeric-domain sample -> explicit FP32 RNE conversion (bits recorded) -> "
                "G32 MXFP8 input quantization"
            ),
            "input_quantization": MX_INPUT_QUANTIZATION_POLICY,
            "element_semantics": MX_ELEMENT_SEMANTICS,
            "element_rounding": "RNE ties-to-even, subnormals kept, no FTZ",
            "scale_semantics": MX_SCALE_SEMANTICS,
            "scale_application": "exact dyadic exponent shift (never a host float multiply)",
            "accumulation": MX_ACCUMULATION_POLICY,
            "reduction": "dot16 blocks summed exactly; one FP32 RNE per dot16 update; gradual underflow",
            "scale_grouping": (
                "input scale groups of 32 along K; the two dot16 blocks of a group share the group "
                "scale and update FP32 separately (never merged into a dot32)"
            ),
            "finite_window": "none (no W-window loss is modelled; the window reducer is not used)",
            "output_scale_policy": MX_OUTPUT_SCALE_POLICY,
            "output_grouping": "output scale groups of 32 along N",
            "nonfinite_input": (
                "non-finite FP32 draws raise; non-finite saved inputs are rejected before evaluation"
            ),
            "scope_note": (
                "measures the behaviour of MXFP8-quantized inputs under dot16+FP32 accumulation and "
                "MX output quantization; it does not measure input quantization loss relative to the "
                "raw FP32 matrix product"
            ),
        },
        "element_format": element_metadata(),
        "scale_format": e8m0_metadata(),
        "group_length": MX_GROUP_LENGTH,
        "mac_parallelism": MX_MAC_PARALLELISM,
        "input_sampling": input_metadata(),
        "distribution_pairs": [dict(pair) for pair in config.distribution_pairs],
        "output_quantization": quantize_metadata(),
        "sampling": {
            f"{pair['activation_distribution']}|{pair['weight_distribution']}": engines[
                _engine_key(pair)
            ].sampling_metadata()
            for pair in config.distribution_pairs
        },
        "expected": {
            "matrix_pairs": n_pairs,
            "snapshot_records": n_pairs * len(config.K_list),
            "output_elements_per_snapshot": config.M * config.N,
        },
        "counts": {},
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    n_records = 0
    n_pair_done = 0
    pair_status: dict[str, int] = {}
    with JsonlSink(run_dir / MX_INPUTS_NAME) as input_sink:
        with JsonlSink(run_dir / MX_SAMPLES_NAME) as sample_sink:
            for pair in config.distribution_pairs:
                engine = engines[_engine_key(pair)]
                for seed in config.seeds:
                    for pair_index in range(config.pairs_per_seed):
                        sample = engine.build_sample(seed, pair_index)
                        input_sink.write(engine.input_record(sample))
                        result = engine.evaluate(sample)
                        for snapshot in result.snapshots:
                            record = {
                                "record_type": "matrix_snapshot",
                                "schema": MX_SCHEMA_VERSION,
                                "mode": MX_MODE,
                                "sample_id": sample.sample_id,
                                "seed": sample.seed,
                                "pair_index": sample.pair_index,
                                "activation_distribution": sample.activation_distribution,
                                "weight_distribution": sample.weight_distribution,
                                **snapshot,
                            }
                            sample_sink.write(record)
                            n_records += 1
                        pair_status[result.status] = pair_status.get(result.status, 0) + 1
                        n_pair_done += 1
                        if verbose:
                            print(
                                f"[mx-run] {n_pair_done}/{n_pairs} matrix pairs ({result.status})",
                                file=sys.stderr,
                            )

    manifest["counts"] = {
        "matrix_pairs": n_pairs,
        "snapshot_records": n_records,
        "pair_status": pair_status,
    }
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    if verbose:
        print(f"[mx-run] completed: {run_dir}", file=sys.stderr)
    return run_dir
