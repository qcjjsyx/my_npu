"""Fixed-W_align K/B sweep driver.

Two input modes:

* ``source_run`` set: reuse the saved MXFP8 element/scale codes of an existing
  ``mx_matrix_sweep`` run (verified schema, identity, dimensions, grouping,
  counts and uniqueness; K_list must be a subset of the source list), slice
  them to the requested prefix and recompute the aligned candidate trajectory.
  The rebuilt no-window ``S``/``L1``/``R`` are compared exactly against the
  source snapshots; any mismatch aborts with the location.
* no ``source_run``: generate fresh numeric-domain FP32 inputs and G32 MX
  quantisation through the existing ``MxMatrixEngine``.

Only execution happens here.  Output quantisation, metrics, alignment
diagnostics and plots are computed offline by
``results/mx_alignment_analysis.py``; source runs stay read-only.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..arithmetic.dyadic import Dyadic
from ..formats.e8m0 import metadata as e8m0_metadata
from ..matrix.aligned_engine import evaluate_mx_pair_aligned
from ..matrix.engine import MxMatrixEngine
from ..matrix.inputs import bits_are_finite
from ..matrix.tensor import MxTensor, element_metadata
from ..results.mx_io import MX_INPUTS_NAME, MX_SAMPLES_NAME, iter_mx_inputs, iter_mx_samples
from ..results.reader import load_manifest, require_schema, run_path
from ..results.schema import mx_align_config_hash
from ..results.writer import JsonlSink, create_run_dir, write_json
from ..version import (
    MX_ACCUMULATION_POLICY,
    MX_ALIGN_CONTAINER_EXTRA_BITS,
    MX_ALIGN_POLICY,
    MX_ALIGN_SCHEMA_VERSION,
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
from .config import MX_GROUP_LENGTH, MX_MAC_PARALLELISM, MX_ALIGN_MODE, RunConfig


def run(config: RunConfig, verbose: bool = True) -> Path:
    if config.mode != MX_ALIGN_MODE:
        raise ValueError("mx_alignment_sweep driver requires mode=mx_alignment_sweep")
    return execute(config, verbose=verbose)


def run_from_file(path, verbose: bool = True) -> Path:
    from .config import load_config

    return run(load_config(path), verbose=verbose)


def _slice_tensor(tensor: MxTensor, kmax: int) -> MxTensor:
    """Prefix slice to the requested K (paired-prefix by construction)."""
    if tensor.group_axis == 1:
        if kmax > tensor.cols:
            raise ValueError(f"cannot slice {tensor.name}: K={kmax} > {tensor.cols}")
        cols = kmax
        grid_cols = tensor.cols // tensor.group_length
        keep_cols = cols // tensor.group_length
        codes: list[int] = []
        for row in range(tensor.rows):
            codes.extend(tensor.element_codes[row * tensor.cols : row * tensor.cols + cols])
        scales: list[int] = []
        for row in range(tensor.rows):
            scales.extend(tensor.scale_codes[row * grid_cols : row * grid_cols + keep_cols])
        return MxTensor(tensor.name, (tensor.rows, cols), 1, codes, scales)
    if tensor.group_axis == 0:
        if kmax > tensor.rows:
            raise ValueError(f"cannot slice {tensor.name}: K={kmax} > {tensor.rows}")
        rows = kmax
        cols = tensor.cols
        codes = tensor.element_codes[: rows * cols]
        scales = tensor.scale_codes[: (rows // tensor.group_length) * cols]
        return MxTensor(tensor.name, (rows, cols), 0, codes, scales)
    raise ValueError("group_axis must be 0 or 1")


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return {"sha256": "sha256:" + digest.hexdigest(), "size": path.stat().st_size}


def _fingerprints(tensor: MxTensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "group_axis": tensor.group_axis,
        "group_length": tensor.group_length,
        "element_fingerprint": tensor.element_fingerprint(),
        "scale_fingerprint": tensor.scale_fingerprint(),
    }


def _check_tensor(name: str, tensor: MxTensor, M: int, N: int, Kmax: int, role: str) -> None:
    if role == "x":
        if tensor.rows != M or tensor.cols != Kmax or tensor.group_axis != 1:
            raise ValueError(
                f"source X tensor shape/axis mismatch: {tensor.shape}, axis {tensor.group_axis}"
            )
    else:
        if tensor.rows != Kmax or tensor.cols != N or tensor.group_axis != 0:
            raise ValueError(
                f"source W tensor shape/axis mismatch: {tensor.shape}, axis {tensor.group_axis}"
            )


def _compare_saved_snapshot(sample_id: str, K: int, saved: dict, source_record: dict) -> None:
    """Compare the trajectory that will be written against the source snapshot.

    The check runs on the new engine result itself (not on a separate legacy
    recomputation), so a fault in the saved S/L1/R cannot pass unnoticed.
    """
    s_source = source_record["S"]
    l1_source = source_record["L1"]
    r_source = source_record["r_bits"]
    if not (len(s_source) == len(l1_source) == len(r_source) == len(saved["S"])):
        raise ValueError(f"snapshot size mismatch for {sample_id} K={K}")
    for index in range(len(s_source)):
        if Dyadic.from_dict(saved["S"][index]) != Dyadic.from_dict(s_source[index]):
            raise ValueError(
                f"S mismatch for {sample_id} K={K} element {index} (field 'S')"
            )
        if Dyadic.from_dict(saved["L1"][index]) != Dyadic.from_dict(l1_source[index]):
            raise ValueError(
                f"L1 mismatch for {sample_id} K={K} element {index} (field 'L1')"
            )
        if int(saved["r_bits"][index]) != int(r_source[index]):
            raise ValueError(
                f"R bits mismatch for {sample_id} K={K} element {index} (field 'r_bits')"
            )


class _Peekable:
    def __init__(self, iterator) -> None:
        self._iterator = iterator
        self._buffer = None

    def next(self):
        if self._buffer is not None:
            value, self._buffer = self._buffer, None
            return value
        return next(self._iterator, None)

    def peek(self):
        if self._buffer is None:
            self._buffer = next(self._iterator, None)
        return self._buffer


def _source_identity(config: RunConfig):
    source = run_path(config.source_run)
    manifest = load_manifest(source)
    require_schema(manifest, (MX_SCHEMA_VERSION,))
    if manifest.get("status") != "completed":
        raise ValueError(
            f"source run {manifest.get('run_id')!r} status is {manifest.get('status')!r}, not 'completed'; "
            "refusing to reuse a partial run"
        )
    if manifest.get("run_kind") != "mx_matrix_sweep":
        raise ValueError(f"source run kind {manifest.get('run_kind')!r} is not mx_matrix_sweep")
    if manifest.get("input_sampling_version") != MX_INPUT_SAMPLING_VERSION:
        raise ValueError(
            f"source run {manifest.get('run_id')!r} does not carry the current input sampling "
            f"version {MX_INPUT_SAMPLING_VERSION!r}"
        )
    expected_pairs = [dict(pair) for pair in config.distribution_pairs]
    source_pairs = [dict(pair) for pair in manifest.get("distribution_pairs") or []]
    if source_pairs != expected_pairs:
        raise ValueError(
            "config distribution_pairs must equal the source run's list and order; "
            f"source={source_pairs} config={expected_pairs}"
        )
    source_seeds = [int(seed) for seed in manifest["config"]["seeds"]]
    if source_seeds != list(config.seeds):
        raise ValueError(
            f"config seeds must equal the source seeds: source={source_seeds} config={config.seeds}"
        )
    source_ppseed = int(manifest["config"]["pairs_per_seed"])
    if source_ppseed != config.pairs_per_seed:
        raise ValueError(
            f"config pairs_per_seed ({config.pairs_per_seed}) must equal the source value ({source_ppseed})"
        )
    source_klist = [int(k) for k in manifest["K_list"]]
    if not set(config.K_list).issubset(set(source_klist)):
        raise ValueError(
            f"config K_list {config.K_list} must be a subset of source K_list {source_klist}"
        )
    expected_source_pairs = len(expected_pairs) * len(source_seeds) * source_ppseed
    expected_source_snapshots = expected_source_pairs * len(source_klist)
    counts = manifest.get("counts") or {}
    if int(counts.get("matrix_pairs", -1)) != expected_source_pairs:
        raise ValueError(
            "source manifest counts.matrix_pairs does not match its identity set: "
            f"{counts.get('matrix_pairs')} != {expected_source_pairs}"
        )
    if int(counts.get("snapshot_records", -1)) != expected_source_snapshots:
        raise ValueError(
            "source manifest counts.snapshot_records does not match its identity set: "
            f"{counts.get('snapshot_records')} != {expected_source_snapshots}"
        )
    for name in ("manifest.json", MX_INPUTS_NAME, MX_SAMPLES_NAME):
        if not (source / name).exists():
            raise ValueError(f"source run {source} is missing {name}")
    public = {
        "kind": "reused",
        "source_run_dir": str(source.resolve()),
        "source_run_id": manifest["run_id"],
        "source_schema_version": manifest["schema_version"],
        "source_config_hash": manifest.get("config_hash"),
        "source_K_list": source_klist,
        "source_M": manifest["fixed"]["M"],
        "source_N": manifest["fixed"]["N"],
        "source_file_identity": {
            name: _file_identity(source / name)
            for name in ("manifest.json", MX_INPUTS_NAME, MX_SAMPLES_NAME)
        },
    }
    return public, source, source_klist


def execute(config: RunConfig, verbose: bool = True) -> Path:
    w_align = config.W_align
    if w_align is None:
        raise ValueError("mx_alignment_sweep requires W_align")
    eval_kmax = max(config.K_list)
    M, N = config.M, config.N
    identity = None
    source = None
    source_klist = None
    if config.source_run:
        identity, source, source_klist = _source_identity(config)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = source_digest()
    config_payload = config.effective()
    chash = mx_align_config_hash(config_payload)
    run_id = config.run_name or f"mx_align_{stamp}_{chash[:8]}"
    run_dir = create_run_dir(config.output_root, run_id)
    if verbose:
        print(f"[mx-align] {run_id} -> {run_dir}", file=sys.stderr)

    n_pairs = len(config.distribution_pairs) * len(config.seeds) * config.pairs_per_seed
    manifest = {
        "schema_version": MX_ALIGN_SCHEMA_VERSION,
        "run_kind": "mx_alignment_sweep",
        "package_version": PACKAGE_VERSION,
        "source_digest": digest,
        "created_utc": created,
        "run_id": run_id,
        "mode": MX_ALIGN_MODE,
        "x_name": "K",
        "w_align": w_align,
        "container_bits": w_align + MX_ALIGN_CONTAINER_EXTRA_BITS,
        "align_policy": MX_ALIGN_POLICY,
        "input_sampling_version": MX_INPUT_SAMPLING_VERSION,
        "input_stream_version": MX_INPUT_STREAM_VERSION,
        "input_quantization_policy": MX_INPUT_QUANTIZATION_POLICY,
        "fixed": config.fixed,
        "K_list": list(config.K_list),
        "B_list": [k // MX_MAC_PARALLELISM for k in config.K_list],
        "config": config_payload,
        "config_hash": chash,
        "config_source": config.describe(),
        "source": identity,
        "policies": {
            "alignment": MX_ALIGN_POLICY,
            "alignment_formula": (
                "n_i = RNE(P_i * 2**(w_align - 6 - (Cmax - C_i))); N_b = sum n_i; "
                "q_b = N_b * 2**(Cmax - 14 - w_align + h)"
            ),
            "element_semantics": MX_ELEMENT_SEMANTICS,
            "element_rounding": "RNE ties-to-even, subnormals kept, no FTZ",
            "scale_semantics": MX_SCALE_SEMANTICS,
            "accumulation": MX_ACCUMULATION_POLICY,
            "candidate_accumulation": "a_(b+1) = RN32(exact(a_b) + q_b); q_b never pre-rounded to FP32",
            "reference": "S/L1 from decoded MX inputs; T = exact sum of q_b; A = FP32 accumulation of q_b",
            "finite_window": "none (legacy no-window path retained; its semantics are unchanged)",
            "output_scale_policy": MX_OUTPUT_SCALE_POLICY,
            "scope_note": (
                "measures fixed-W_align dot16 alignment error and its interaction with FP32 "
                "accumulation and MX output quantization on MXFP8 inputs"
            ),
        },
        "element_format": element_metadata(),
        "scale_format": e8m0_metadata(),
        "group_length": MX_GROUP_LENGTH,
        "mac_parallelism": MX_MAC_PARALLELISM,
        "expected": {
            "matrix_pairs": n_pairs,
            "snapshot_records": n_pairs * len(config.K_list),
            "output_elements_per_snapshot": M * N,
            "output_groups_per_snapshot": M * N // MX_GROUP_LENGTH,
        },
        "counts": {},
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    n_records = 0
    n_pair_done = 0
    pair_status: dict[str, int] = {}

    def snapshot_payload(snapshot: dict, sample_id: str, seed: int, pair_index: int, act: str, weight: str) -> dict:
        return {
            **snapshot,
            "sample_id": sample_id,
            "seed": seed,
            "pair_index": pair_index,
            "activation_distribution": act,
            "weight_distribution": weight,
            "schema": MX_ALIGN_SCHEMA_VERSION,
        }

    with JsonlSink(run_dir / MX_INPUTS_NAME) as input_sink:
        with JsonlSink(run_dir / MX_SAMPLES_NAME) as sample_sink:
            if config.source_run:
                input_iter = _Peekable(iter_mx_inputs(source))
                sample_iter = _Peekable(iter_mx_samples(source))
                for pair in config.distribution_pairs:
                    act = pair["activation_distribution"]
                    weight = pair["weight_distribution"]
                    for seed in config.seeds:
                        for pair_index in range(config.pairs_per_seed):
                            sample_id = f"mx|{act}|{weight}|seed{seed}|{pair_index:05d}"
                            record = input_iter.next()
                            if record is None or record.get("sample_id") != sample_id:
                                raise ValueError(
                                    f"source input records are not in the expected order at {sample_id}: "
                                    f"got {None if record is None else record.get('sample_id')}"
                                )
                            if int(record["pair_index"]) != pair_index or int(record["seed"]) != seed:
                                raise ValueError(f"source input identity mismatch for {sample_id}")
                            if record.get("activation_distribution") != act or record.get(
                                "weight_distribution"
                            ) != weight:
                                raise ValueError(f"source input distribution mismatch for {sample_id}")
                            x = MxTensor.from_dict(record["x"])
                            w = MxTensor.from_dict(record["w"])
                            if int(record["M"]) != M or int(record["N"]) != N:
                                raise ValueError(f"source input dimensions mismatch for {sample_id}")
                            _check_tensor("x", x, M, N, max(source_klist), "x")
                            _check_tensor("w", w, M, N, max(source_klist), "w")
                            if not bits_are_finite(record.get("x_fp32_bits") or []):
                                raise ValueError(f"source X FP32 bits non-finite/malformed for {sample_id}")
                            if not bits_are_finite(record.get("w_fp32_bits") or []):
                                raise ValueError(f"source W FP32 bits non-finite/malformed for {sample_id}")
                            if x.element_fingerprint() != record["x"].get("element_fingerprint"):
                                raise ValueError(f"source X element fingerprint mismatch for {sample_id}")
                            if w.element_fingerprint() != record["w"].get("element_fingerprint"):
                                raise ValueError(f"source W element fingerprint mismatch for {sample_id}")
                            x_eval = _slice_tensor(x, eval_kmax) if eval_kmax != x.cols else x
                            w_eval = _slice_tensor(w, eval_kmax) if eval_kmax != w.rows else w

                            # gather all source snapshots of this sample (contiguous in file order)
                            source_snapshots: dict[int, dict] = {}
                            while True:
                                candidate = sample_iter.peek()
                                if candidate is None or candidate.get("sample_id") != sample_id:
                                    break
                                candidate = sample_iter.next()
                                source_K = int(candidate["K"])
                                if source_K in source_snapshots:
                                    raise ValueError(
                                        f"duplicate source snapshot for {sample_id} K={source_K}; "
                                        "refusing to silently overwrite"
                                    )
                                source_snapshots[source_K] = candidate
                            # The source run declares its own complete K list; every
                            # reused pair must actually provide that full set, not
                            # only the K values requested by this alignment config.
                            gathered = sorted(source_snapshots)
                            if gathered != source_klist:
                                missing = sorted(set(source_klist) - set(gathered))
                                extra = sorted(set(gathered) - set(source_klist))
                                raise ValueError(
                                    f"source snapshots incomplete for {sample_id}: "
                                    f"missing K={missing} unexpected K={extra}; "
                                    "the source run is not the complete run declared by its manifest"
                                )
                            result = evaluate_mx_pair_aligned(
                                x_eval, w_eval, config.K_list, w_align=w_align
                            )
                            for k_index, K in enumerate(config.K_list):
                                source_record = source_snapshots.get(int(K))
                                if source_record is None:
                                    raise ValueError(f"source snapshot missing for {sample_id} K={K}")
                                _compare_saved_snapshot(
                                    sample_id, K, result.snapshots[k_index], source_record
                                )
                                sample_sink.write(
                                    snapshot_payload(
                                        result.snapshots[k_index], sample_id, seed, pair_index, act, weight
                                    )
                                )
                                n_records += 1
                            input_sink.write(
                                {
                                    "record_type": "align_input_pair",
                                    "schema": MX_ALIGN_SCHEMA_VERSION,
                                    "sample_id": sample_id,
                                    "seed": seed,
                                    "pair_index": pair_index,
                                    "activation_distribution": act,
                                    "weight_distribution": weight,
                                    "M": M,
                                    "N": N,
                                    "Kmax": eval_kmax,
                                    "K_list": list(config.K_list),
                                    "B_list": [k // MX_MAC_PARALLELISM for k in config.K_list],
                                    "group_length": MX_GROUP_LENGTH,
                                    "mac_parallelism": MX_MAC_PARALLELISM,
                                    "w_align": w_align,
                                    "source": {
                                        "kind": "reused",
                                        "source_run_id": identity["source_run_id"],
                                        "source_run_dir": identity["source_run_dir"],
                                        "source_config_hash": identity["source_config_hash"],
                                        "source_sample_id": record.get("sample_id"),
                                        "source_file_identity": identity["source_file_identity"],
                                        "prefix_Kmax": eval_kmax,
                                        "source_Kmax": max(source_klist),
                                    },
                                    "codes_available": False,
                                    "codes_in": "source_run",
                                    "x": _fingerprints(x_eval),
                                    "w": _fingerprints(w_eval),
                                    "x_input_quantization_events": record.get(
                                        "x_input_quantization_events", {}
                                    ),
                                    "w_input_quantization_events": record.get(
                                        "w_input_quantization_events", {}
                                    ),
                                    "x_input_diagnostics": record.get("x_input_diagnostics", {}),
                                    "w_input_diagnostics": record.get("w_input_diagnostics", {}),
                                }
                            )
                            pair_status[result.status] = pair_status.get(result.status, 0) + 1
                            n_pair_done += 1
                            if verbose:
                                print(
                                    f"[mx-align] {n_pair_done}/{n_pairs} matrix pairs ({result.status})",
                                    file=sys.stderr,
                                )
                leftover_input = input_iter.next()
                if leftover_input is not None:
                    raise ValueError(
                        "unexpected extra source input record: "
                        f"{leftover_input.get('sample_id')!r}"
                    )
                leftover_sample = sample_iter.next()
                if leftover_sample is not None:
                    raise ValueError(
                        "unexpected extra source snapshot record: "
                        f"{leftover_sample.get('sample_id')!r} K={leftover_sample.get('K')}"
                    )
            else:
                engines: dict[tuple, MxMatrixEngine] = {}
                for pair in config.distribution_pairs:
                    key = (pair["activation_distribution"], pair["weight_distribution"])
                    engines[key] = MxMatrixEngine(M, N, config.K_list, key[0], key[1])
                for pair in config.distribution_pairs:
                    act = pair["activation_distribution"]
                    weight = pair["weight_distribution"]
                    engine = engines[(act, weight)]
                    for seed in config.seeds:
                        for pair_index in range(config.pairs_per_seed):
                            sample = engine.build_sample(seed, pair_index)
                            base_record = engine.input_record(sample)
                            input_sink.write(
                                {
                                    "record_type": "align_input_pair",
                                    "schema": MX_ALIGN_SCHEMA_VERSION,
                                    "sample_id": sample.sample_id,
                                    "seed": sample.seed,
                                    "pair_index": sample.pair_index,
                                    "activation_distribution": act,
                                    "weight_distribution": weight,
                                    "M": M,
                                    "N": N,
                                    "Kmax": engine.kmax,
                                    "K_list": list(config.K_list),
                                    "B_list": [k // MX_MAC_PARALLELISM for k in config.K_list],
                                    "group_length": MX_GROUP_LENGTH,
                                    "mac_parallelism": MX_MAC_PARALLELISM,
                                    "w_align": w_align,
                                    "source": {"kind": "generated"},
                                    "codes_available": True,
                                    "codes_in": "this_run",
                                    "x_fp32_bits": base_record["x_fp32_bits"],
                                    "w_fp32_bits": base_record["w_fp32_bits"],
                                    "x": base_record["x"],
                                    "w": base_record["w"],
                                    "x_input_quantization_events": base_record[
                                        "x_input_quantization_events"
                                    ],
                                    "w_input_quantization_events": base_record[
                                        "w_input_quantization_events"
                                    ],
                                    "x_input_diagnostics": base_record["x_input_diagnostics"],
                                    "w_input_diagnostics": base_record["w_input_diagnostics"],
                                }
                            )
                            result = evaluate_mx_pair_aligned(
                                sample.x, sample.w, config.K_list, w_align=w_align
                            )
                            for snapshot in result.snapshots:
                                sample_sink.write(
                                    snapshot_payload(
                                        snapshot, sample.sample_id, sample.seed, sample.pair_index, act, weight
                                    )
                                )
                                n_records += 1
                            pair_status[result.status] = pair_status.get(result.status, 0) + 1
                            n_pair_done += 1
                            if verbose:
                                print(
                                    f"[mx-align] {n_pair_done}/{n_pairs} matrix pairs ({result.status})",
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
        print(f"[mx-align] completed: {run_dir}", file=sys.stderr)
    return run_dir
