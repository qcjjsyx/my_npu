"""Run driver for the W_align / W_acc bit-width selection experiment.

* ``widthselect_align``: fresh inputs, exact ``T(W_align)`` per dot16.
* ``widthselect_acc``: fresh inputs, exact ``s_b`` through ``HWAdd_{W_acc}``.
* ``widthselect_joint``: ``C_{b+1} = HWAdd_{W_acc}(C_b, q_b(W_align))`` for the
  explicitly configured combinations only (never a default 2-D sweep).

No historical input, result or analysis is ever read; inputs are regenerated
from the new stream namespace, and the run directory/schema/records are new.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from ..results.reader import load_manifest
from ..results.schema import source_digest, widthselect_config_hash
from ..results.writer import JsonlSink, MANIFEST_NAME, create_run_dir, write_json
from ..version import (
    PACKAGE_VERSION,
    WIDTHSELECT_ACC_FORMAT_STATUS,
    WIDTHSELECT_ACC_MODE,
    WIDTHSELECT_ACC_POLICY,
    WIDTHSELECT_ALIGN_MODE,
    WIDTHSELECT_ALIGN_POLICY,
    WIDTHSELECT_CONTAINER_EXTRA_BITS,
    WIDTHSELECT_INPUT_SAMPLING_VERSION,
    WIDTHSELECT_INPUT_STREAM_VERSION,
    WIDTHSELECT_JOINT_MODE,
    WIDTHSELECT_SCHEMA_VERSION,
    WIDTHSELECT_TF32_POLICY,
)
from .acc_format import metadata as acc_metadata
from .config import MAC_PARALLELISM, WidthSelectConfig, load_widthselect_config
from .distributions import (
    WIDTH_DISTRIBUTIONS,
    distribution_metadata,
    sample_fp32_matrix,
)
from .io import WIDTH_INPUTS_NAME, WIDTH_SAMPLES_NAME, file_identity
from .reference import evaluate_pair_acc, evaluate_pair_align, evaluate_pair_joint
from .tensors import metadata as tensor_metadata, quantize_fp32_bits

RUN_PREFIX = {
    WIDTHSELECT_ALIGN_MODE: "width_align",
    WIDTHSELECT_ACC_MODE: "width_acc",
    WIDTHSELECT_JOINT_MODE: "width_joint",
}


def run(config: WidthSelectConfig, verbose: bool = True) -> Path:
    if config.mode not in RUN_PREFIX:
        raise ValueError(f"unsupported mode {config.mode!r}")
    return _execute(config, verbose=verbose)


def run_from_file(path, verbose: bool = True) -> Path:
    return run(load_widthselect_config(path), verbose=verbose)


def _sample_id(act: str, weight: str, seed: int, pair_index: int) -> str:
    return f"width|{act}|{weight}|seed{seed}|{pair_index:05d}"


def _candidate_count(config: WidthSelectConfig) -> int:
    if config.mode == WIDTHSELECT_ALIGN_MODE:
        return len(config.W_align_list)
    if config.mode == WIDTHSELECT_ACC_MODE:
        return len(config.W_acc_list)
    return len(config.joint_pairs)


def _expected_counts(config: WidthSelectConfig) -> dict:
    n_pairs = config.n_pairs()
    n_k = len(config.K_list)
    n_formats = len(config.formats)
    return {
        "matrix_pairs": n_pairs,
        "base_snapshot_records": n_pairs * n_k * n_formats,
        "width_snapshot_records": n_pairs * n_k * n_formats * _candidate_count(config),
        "output_elements_per_snapshot": config.M * config.N,
        "formats": list(config.formats),
        "W_align_values": list(config.W_align_list),
        "W_acc_values": list(config.W_acc_list),
        "joint_pairs": [dict(pair) for pair in config.joint_pairs],
    }


def _policies(config: WidthSelectConfig) -> dict:
    if config.mode == WIDTHSELECT_ALIGN_MODE:
        candidate = {
            "kind": WIDTHSELECT_ALIGN_MODE,
            "policy": WIDTHSELECT_ALIGN_POLICY,
            "formula_mx_fp8": (
                "n_i = RNE(P_i * 2**(W_align - 6 - (Cmax - C_i))); N_b = sum n_i; "
                "q_b = N_b * 2**(Cmax - 14 - W_align + h)"
            ),
            "formula_tf32": (
                "n_i = RNE(P_i * 2**(W_align - 20 - (Cmax - C_i))); N_b = sum n_i; "
                "q_b = N_b * 2**(Cmax - 254 - W_align)"
            ),
            "container_extra_bits": WIDTHSELECT_CONTAINER_EXTRA_BITS,
            "state": "T = exact sum of q_b over dot16 blocks; Y_align = RN32(T) only at snapshots",
        }
    elif config.mode == WIDTHSELECT_ACC_MODE:
        candidate = {
            "kind": WIDTHSELECT_ACC_MODE,
            "policy": WIDTHSELECT_ACC_POLICY,
            "update": "C_(b+1) = HWAdd_{W_acc}(C_b, s_b) with the exact dot16 partial s_b",
            "state": "two's complement W_acc+1 bit significand + 8 bit exponent (W_acc+9 logical bits)",
            "final": "Y_acc = RN32(decode(C_B)) only at snapshots; no per-update FP32",
            "g32_note": "one G32 group still holds two independent dot16 blocks",
        }
    else:
        candidate = {
            "kind": WIDTHSELECT_JOINT_MODE,
            "policy": WIDTHSELECT_ACC_POLICY,
            "update": "C_(b+1) = HWAdd_{W_acc}(C_b, q_b(W_align))",
            "state": "two's complement W_acc+1 bit significand + 8 bit exponent (W_acc+9 logical bits)",
            "final": "Y_joint = RN32(decode(C_B)) only at snapshots",
            "scope_note": "isolated qualifications do not imply a joint qualification; reported separately",
        }
    return {
        "input_pipeline": (
            "distribution sample -> explicit integer RNE to FP32 bits -> target-format "
            "quantization; both formats share the same freshly generated FP32 matrix pair"
        ),
        "input_sampling_version": WIDTHSELECT_INPUT_SAMPLING_VERSION,
        "input_stream_namespace": f"mac-width-fp32-stream-v{WIDTHSELECT_INPUT_STREAM_VERSION}",
        "input_stream_note": (
            "keys include seed, matrix, distribution label, pair index, row and column; "
            "paired-prefix stable and independent of Kmax/pair count"
        ),
        "tf32": {
            "policy": WIDTHSELECT_TF32_POLICY,
            "rounding": "RNE ties-to-even on the exact dyadic significand (11 significant bits)",
            "subnormal": "raw_exp=0, frac>0 -> frac * 2**-136",
            "overflow": (
                "input quantization saturates to +/-max finite and counts element_overflow_saturated; "
                "the generic conversion keeps the +/-Inf encoding"
            ),
        },
        "mx_fp8": {
            "element_semantics": "ocp_e4m3_no_ftz",
            "scale_semantics": "ocp_e8m0_bias127_reserved_255",
            "input_quantization": "minimal binary power covering the exact group maximum",
            "group_length": 32,
            "x_group_axis": 1,
            "w_group_axis": 0,
        },
        "candidate": candidate,
        "reference": {
            "s_b": "exact dyadic sum of the 16 decoded target-format products",
            "S": "exact dyadic sum of s_b (never FP32)",
            "L1": "exact dyadic sum of |decoded product|",
            "O": "RN32(S), the primary reference",
            "scope_note": (
                "the metrics exclude the raw-FP32-to-target-format input quantization loss"
            ),
        },
        "acc_format": acc_metadata(),
        "metrics": {
            "primary": "FP32 full 32-bit encoding match rate and ULP distance buckets",
            "ulp": "ordered finite FP32 encodings; +0/-0 merged for distance, distinguished for bit match",
            "conditional_ulp_denominator": "finite pairs with bit mismatch; null when empty",
            "nonfinite": "counted separately and never silently removed from denominators",
            "aux": "absolute error, etaL1 (=|err|/L1), relative L2 with explicit denominators",
            "independent_unit": "matrix pair; no element-level Bernoulli confidence bounds",
        },
        "distribution_note": "six fixed parameter definitions; custom parameters rejected",
    }


def _execute(config: WidthSelectConfig, verbose: bool = True) -> Path:
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = source_digest()
    config_payload = config.effective()
    chash = widthselect_config_hash(config_payload)
    run_id = config.run_name or f"{RUN_PREFIX[config.mode]}_{stamp}_{chash[:8]}"
    run_dir = create_run_dir(config.output_root, run_id)
    if verbose:
        print(f"[{RUN_PREFIX[config.mode]}] {run_id} -> {run_dir}", file=sys.stderr)

    kmax = max(config.K_list)
    manifest = {
        "schema_version": WIDTHSELECT_SCHEMA_VERSION,
        "run_kind": config.mode,
        "package_version": PACKAGE_VERSION,
        "source_digest": digest,
        "created_utc": created,
        "run_id": run_id,
        "mode": config.mode,
        "x_name": config.x_name,
        "B_rule": "B = K // 16",
        "formats": list(config.formats),
        "distribution_pairs": [dict(pair) for pair in config.distribution_pairs],
        "seeds": list(config.seeds),
        "pairs_per_seed": config.pairs_per_seed,
        "fixed": {"M": config.M, "N": config.N, "mac_parallelism": MAC_PARALLELISM},
        "K_list": list(config.K_list),
        "B_list": config.B_list,
        "Kmax": kmax,
        "W_align_list": list(config.W_align_list),
        "W_acc_list": list(config.W_acc_list),
        "joint_pairs": [dict(pair) for pair in config.joint_pairs],
        "pass_criteria": config.pass_criteria,
        "config": config_payload,
        "config_hash": chash,
        "config_source": config.describe(),
        "source": {"kind": "fresh", "note": "no historical input reuse; inputs regenerated in this run"},
        "policies": _policies(config),
        "input_metadata": {
            "distributions": {name: distribution_metadata(name) for name in WIDTH_DISTRIBUTIONS},
            "target_formats": tensor_metadata(),
        },
        "expected": _expected_counts(config),
        "counts": {},
        "status": "running",
    }
    write_json(run_dir / MANIFEST_NAME, manifest)

    n_inputs = 0
    n_base = 0
    n_width = 0
    n_pair_done = 0
    pair_status: dict[str, int] = {}
    n_pairs = config.n_pairs()

    def common(sample_id: str, seed: int, pair_index: int, act: str, weight: str, fmt: str) -> dict:
        return {
            "schema": WIDTHSELECT_SCHEMA_VERSION,
            "sample_id": sample_id,
            "seed": seed,
            "pair_index": pair_index,
            "activation_distribution": act,
            "weight_distribution": weight,
            "format": fmt,
            "M": config.M,
            "N": config.N,
            "Kmax": kmax,
            "K_list": list(config.K_list),
            "B_rule": "B = K // 16",
            "input_sampling_version": WIDTHSELECT_INPUT_SAMPLING_VERSION,
            "input_stream_version": WIDTHSELECT_INPUT_STREAM_VERSION,
            "tf32_policy": WIDTHSELECT_TF32_POLICY,
            "w_align_policy": WIDTHSELECT_ALIGN_POLICY,
            "acc_policy": WIDTHSELECT_ACC_POLICY,
            "acc_format_status": WIDTHSELECT_ACC_FORMAT_STATUS,
        }

    try:
        with JsonlSink(run_dir / WIDTH_INPUTS_NAME) as input_sink:
            with JsonlSink(run_dir / WIDTH_SAMPLES_NAME) as sample_sink:
                for pair in config.distribution_pairs:
                    act = pair["activation_distribution"]
                    weight = pair["weight_distribution"]
                    for seed in config.seeds:
                        for pair_index in range(config.pairs_per_seed):
                            sample_id = _sample_id(act, weight, seed, pair_index)
                            x_bits = sample_fp32_matrix(act, seed, "x", config.M, kmax, pair_index)
                            w_bits = sample_fp32_matrix(weight, seed, "w", kmax, config.N, pair_index)
                            quantized = {
                                fmt: {
                                    "x": quantize_fp32_bits(fmt, "x", x_bits, config.M, kmax, group_axis=1),
                                    "w": quantize_fp32_bits(fmt, "w", w_bits, kmax, config.N, group_axis=0),
                                }
                                for fmt in config.formats
                            }
                            input_sink.write(
                                {
                                    "record_type": "widthselect_input_pair",
                                    "schema": WIDTHSELECT_SCHEMA_VERSION,
                                    "sample_id": sample_id,
                                    "seed": seed,
                                    "pair_index": pair_index,
                                    "activation_distribution": act,
                                    "weight_distribution": weight,
                                    "M": config.M,
                                    "N": config.N,
                                    "Kmax": kmax,
                                    "K_list": list(config.K_list),
                                    "B_list": config.B_list,
                                    "mac_parallelism": MAC_PARALLELISM,
                                    "input_sampling_version": WIDTHSELECT_INPUT_SAMPLING_VERSION,
                                    "input_stream_version": WIDTHSELECT_INPUT_STREAM_VERSION,
                                    "activation_sampler": distribution_metadata(act),
                                    "weight_sampler": distribution_metadata(weight),
                                    "x_fp32_bits": list(x_bits),
                                    "w_fp32_bits": list(w_bits),
                                    "formats": {
                                        fmt: {
                                            "x": payload["x"].tensor.to_dict(),
                                            "w": payload["w"].tensor.to_dict(),
                                            "x_input_quantization_events": payload["x"].events,
                                            "w_input_quantization_events": payload["w"].events,
                                            "x_input_diagnostics": payload["x"].diagnostics,
                                            "w_input_diagnostics": payload["w"].diagnostics,
                                        }
                                        for fmt, payload in quantized.items()
                                    },
                                }
                            )
                            n_inputs += 1
                            for fmt in config.formats:
                                x = quantized[fmt]["x"].tensor
                                w = quantized[fmt]["w"].tensor
                                if config.mode == WIDTHSELECT_ALIGN_MODE:
                                    result = evaluate_pair_align(x, w, config.K_list, config.W_align_list)
                                    base_snapshots = result.base_snapshots
                                    candidate_snapshots = result.width_snapshots
                                    status = result.status
                                elif config.mode == WIDTHSELECT_ACC_MODE:
                                    result = evaluate_pair_acc(x, w, config.K_list, config.W_acc_list)
                                    base_snapshots = result.base_snapshots
                                    candidate_snapshots = result.acc_snapshots
                                    status = result.status
                                else:
                                    result = evaluate_pair_joint(x, w, config.K_list, config.joint_pairs)
                                    base_snapshots = result.base_snapshots
                                    candidate_snapshots = result.joint_snapshots
                                    status = result.status
                                for snapshot in base_snapshots:
                                    sample_sink.write({**snapshot, **common(sample_id, seed, pair_index, act, weight, fmt)})
                                    n_base += 1
                                for snapshot in candidate_snapshots:
                                    sample_sink.write({**snapshot, **common(sample_id, seed, pair_index, act, weight, fmt)})
                                    n_width += 1
                                pair_status[status] = pair_status.get(status, 0) + 1
                            n_pair_done += 1
                            if verbose:
                                print(
                                    f"[{RUN_PREFIX[config.mode]}] {n_pair_done}/{n_pairs} matrix pairs, "
                                    f"{len(config.formats)} format(s)",
                                    file=sys.stderr,
                                )
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest["counts"] = {
            "matrix_pairs": n_pair_done,
            "input_records": n_inputs,
            "base_snapshot_records": n_base,
            "width_snapshot_records": n_width,
            "pair_status": pair_status,
        }
        write_json(run_dir / MANIFEST_NAME, manifest)
        raise

    expected = _expected_counts(config)
    if n_inputs != expected["matrix_pairs"]:
        raise AssertionError(f"input count {n_inputs} != expected {expected['matrix_pairs']}")
    if n_base != expected["base_snapshot_records"]:
        raise AssertionError(f"base snapshot count {n_base} != expected {expected['base_snapshot_records']}")
    if n_width != expected["width_snapshot_records"]:
        raise AssertionError(f"width snapshot count {n_width} != expected {expected['width_snapshot_records']}")

    manifest["counts"] = {
        "matrix_pairs": n_inputs,
        "input_records": n_inputs,
        "base_snapshot_records": n_base,
        "width_snapshot_records": n_width,
        "pair_status": pair_status,
    }
    manifest["file_identity"] = {
        name: file_identity(run_dir / name) for name in (WIDTH_INPUTS_NAME, WIDTH_SAMPLES_NAME)
    }
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir / MANIFEST_NAME, manifest)
    if verbose:
        print(f"[{RUN_PREFIX[config.mode]}] completed: {run_dir}", file=sys.stderr)
    return run_dir


def read_manifest(run_dir) -> dict:
    """Manifest reader with a schema guard (no historical schemas accepted)."""
    manifest = load_manifest(run_dir)
    if manifest.get("schema_version") != WIDTHSELECT_SCHEMA_VERSION:
        raise ValueError(
            f"expected schema {WIDTHSELECT_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}; "
            "this entry point never reads historical run directories"
        )
    return manifest
