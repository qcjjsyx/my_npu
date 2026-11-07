"""Offline analysis for the width-select experiment (schema ``mac-width-v1``).

Supports all three run kinds:

* ``widthselect_align``: candidate ``T(W_align)`` / ``Y_align``;
* ``widthselect_acc``: candidate ``C_B(W_acc)`` / ``Y_acc``;
* ``widthselect_joint``: candidate joint states / ``Y_joint``.

Read-only over the source run; never imports the execution engine or the
sampler.  All source files are fingerprinted and every expected
``(sample_id, K, format, candidate)`` cell must be present exactly once;
missing, duplicate or unknown records abort before any output is written.

Per cell the summary reports the primary FP32 bit-match / ULP statistics against
``O = RN32(S)``, the auxiliary ``decode(Y)-S`` errors, the exact trajectory
series (``T-S`` for experiment 1, ``decode(C_B)-S`` for the accumulator runs)
and, when ``pass_criteria`` is configured, the minimal passing candidate within
the tested range.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.rounding import round_dyadic_to_fp32
from ..results.reader import load_manifest, run_path
from ..results.schema import source_digest
from ..results.writer import write_json
from ..version import (
    WIDTHSELECT_ACC_MODE,
    WIDTHSELECT_ALIGN_MODE,
    WIDTHSELECT_ANALYSIS_VERSION,
    WIDTHSELECT_JOINT_MODE,
    WIDTHSELECT_SCHEMA_VERSION,
)
from .acc_format import HwAccState, decode_state
from .config import MAC_PARALLELISM
from .io import (
    WIDTH_INPUTS_NAME,
    WIDTH_SAMPLES_NAME,
    file_identity,
    iter_width_inputs,
    iter_width_samples,
)
from .metrics import MetricsCollector, ScalarErrorCollector

ANALYSIS_KIND = {
    WIDTHSELECT_ALIGN_MODE: "widthselect_align_analysis",
    WIDTHSELECT_ACC_MODE: "widthselect_acc_analysis",
    WIDTHSELECT_JOINT_MODE: "widthselect_joint_analysis",
}

CANDIDATE_RECORD_TYPES = {
    WIDTHSELECT_ALIGN_MODE: "widthselect_align_snapshot",
    WIDTHSELECT_ACC_MODE: "widthselect_acc_snapshot",
    WIDTHSELECT_JOINT_MODE: "widthselect_joint_snapshot",
}

NOTES = {
    "scope": "measures the MAC numeric path (alignment / accumulation) on decoded target-format inputs",
    "reference": "O = RN32(S); the metrics exclude the raw-FP32-to-target-format quantization loss",
    "independent_unit": (
        "matrix pair; elements and K prefixes are correlated, so no element-level Bernoulli "
        "confidence bounds are reported"
    ),
    "monotonicity": (
        "no monotonicity of the match rate in W_align/W_acc is assumed; recommendations only state "
        "the minimal passing candidate within the tested widths/distributions/K and the configured thresholds"
    ),
    "nonfinite": "non-finite results are counted and never silently dropped from the primary denominator",
}


def _recompute_bits(value: Dyadic) -> int:
    bits, _ = round_dyadic_to_fp32(value)
    return bits.bits


def _candidate_key(config: dict, record: dict) -> tuple:
    mode = config["mode"]
    if mode == WIDTHSELECT_ALIGN_MODE:
        return ("align", int(record["w_align"]))
    if mode == WIDTHSELECT_ACC_MODE:
        return ("acc", int(record["w_acc"]))
    return ("joint", int(record["w_align"]), int(record["w_acc"]))


def _candidate_fields(key: tuple) -> dict:
    if key[0] == "align":
        return {"w_align": key[1], "w_acc": None}
    if key[0] == "acc":
        return {"w_align": None, "w_acc": key[1]}
    return {"w_align": key[1], "w_acc": key[2]}


def _expected_candidate_keys(config: dict) -> list[tuple]:
    mode = config["mode"]
    if mode == WIDTHSELECT_ALIGN_MODE:
        return [("align", int(value)) for value in config["W_align_list"]]
    if mode == WIDTHSELECT_ACC_MODE:
        return [("acc", int(value)) for value in config["W_acc_list"]]
    return [
        ("joint", int(pair["w_align"]), int(pair["w_acc"])) for pair in config["joint_pairs"]
    ]


def _expected_identities(config: dict) -> tuple[set, set, set, dict]:
    pairs = config["distribution_pairs"]
    seeds = config["seeds"]
    pairs_per_seed = config["pairs_per_seed"]
    k_list = config["K_list"]
    formats = config["formats"]
    candidates = _expected_candidate_keys(config)
    expected_inputs: set = set()
    expected_base: set = set()
    expected_candidates: set = set()
    sample_meta: dict = {}
    for pair in pairs:
        act = pair["activation_distribution"]
        weight = pair["weight_distribution"]
        for seed in seeds:
            for pair_index in range(pairs_per_seed):
                sample_id = f"width|{act}|{weight}|seed{seed}|{pair_index:05d}"
                expected_inputs.add(sample_id)
                sample_meta[sample_id] = {
                    "activation_distribution": act,
                    "weight_distribution": weight,
                    "seed": seed,
                    "pair_index": pair_index,
                }
                for k in k_list:
                    for fmt in formats:
                        expected_base.add((sample_id, int(k), fmt))
                        for candidate in candidates:
                            expected_candidates.add((sample_id, int(k), fmt, candidate))
    return expected_inputs, expected_base, expected_candidates, sample_meta


def _check_tensor_payload(sample_id: str, fmt: str, role: str, payload: dict, rows: int, cols: int) -> None:
    from .tensors import TargetTensor

    if not payload:
        raise ValueError(f"input record {sample_id} is missing the {fmt}/{role} tensor")
    tensor = TargetTensor.from_dict(payload)
    if tensor.rows != rows or tensor.cols != cols:
        raise ValueError(
            f"input record {sample_id} {fmt}/{role} shape mismatch: {tensor.shape} != ({rows}, {cols})"
        )
    if tensor.code_fingerprint() != payload.get("code_fingerprint"):
        raise ValueError(f"input record {sample_id} {fmt}/{role} code fingerprint mismatch")
    if fmt == "mx_fp8" and tensor.scale_fingerprint() != payload.get("scale_fingerprint"):
        raise ValueError(f"input record {sample_id} {fmt}/{role} scale fingerprint mismatch")


def _load_inputs(source: Path, config: dict, expected_inputs: set, verbose: bool) -> tuple[dict, dict]:
    M, N = int(config["M"]), int(config["N"])
    kmax = max(int(k) for k in config["K_list"])
    records: dict = {}
    events_by_format: dict = {
        fmt: {"x": {}, "w": {}} for fmt in config["formats"]
    }

    def _accumulate(fmt: str, role: str, events: dict) -> None:
        bucket = events_by_format[fmt][role]
        for key, value in (events or {}).items():
            bucket[key] = bucket.get(key, 0) + int(value)

    for record in iter_width_inputs(source):
        if record.get("schema") != WIDTHSELECT_SCHEMA_VERSION:
            raise ValueError(f"input record schema {record.get('schema')!r} is not {WIDTHSELECT_SCHEMA_VERSION!r}")
        if record.get("record_type") != "widthselect_input_pair":
            raise ValueError(f"unknown input record type {record.get('record_type')!r}")
        sample_id = record.get("sample_id")
        if sample_id in records:
            raise ValueError(f"duplicate input record for {sample_id}")
        if int(record.get("M")) != M or int(record.get("N")) != N or int(record.get("Kmax")) != kmax:
            raise ValueError(f"input record {sample_id} dimensions/Kmax mismatch")
        for role, length in (("x", M * kmax), ("w", kmax * N)):
            bits = record.get(f"{role}_fp32_bits")
            if not isinstance(bits, list) or len(bits) != length:
                raise ValueError(f"input record {sample_id} {role}_fp32_bits malformed")
            for raw in bits:
                if ((int(raw) >> 23) & 0xFF) == 0xFF:
                    raise ValueError(f"input record {sample_id} {role}_fp32_bits contains a non-finite encoding")
        formats_payload = record.get("formats") or {}
        if sorted(formats_payload) != sorted(config["formats"]):
            raise ValueError(f"input record {sample_id} formats {sorted(formats_payload)} != {sorted(config['formats'])}")
        for fmt, payload in formats_payload.items():
            _check_tensor_payload(sample_id, fmt, "x", payload.get("x"), M, kmax)
            _check_tensor_payload(sample_id, fmt, "w", payload.get("w"), kmax, N)
            _accumulate(fmt, "x", payload.get("x_input_quantization_events"))
            _accumulate(fmt, "w", payload.get("w_input_quantization_events"))
        records[sample_id] = record
    missing = sorted(expected_inputs - set(records))
    extra = sorted(set(records) - expected_inputs)
    if missing or extra:
        raise ValueError(f"input record set mismatch: missing={missing[:5]} extra={extra[:5]}")
    if verbose:
        print(f"[width-analyze] verified {len(records)} input pairs")
    return records, events_by_format


def _check_lengths(sample_id: str, K: int, fmt: str, record: dict, grid: int, fields: tuple[str, ...]) -> None:
    for field in fields:
        values = record.get(field)
        if not isinstance(values, list) or len(values) != grid:
            raise ValueError(f"{sample_id} K={K} format={fmt}: {field} malformed")


def _load_snapshots(
    source: Path,
    config: dict,
    expected_base: set,
    expected_candidates: set,
    verbose: bool,
) -> tuple[dict, dict]:
    mode = config["mode"]
    M, N = int(config["M"]), int(config["N"])
    grid = M * N
    k_set = {int(k) for k in config["K_list"]}
    candidate_type = CANDIDATE_RECORD_TYPES[mode]
    candidate_values = set(_expected_candidate_keys(config))
    base: dict = {}
    candidates: dict = {}
    for record in iter_width_samples(source):
        if record.get("schema") != WIDTHSELECT_SCHEMA_VERSION:
            raise ValueError(f"sample record schema {record.get('schema')!r} is not {WIDTHSELECT_SCHEMA_VERSION!r}")
        sample_id = record.get("sample_id")
        K = int(record.get("K"))
        if K not in k_set or int(record.get("B")) != K // MAC_PARALLELISM:
            raise ValueError(f"sample record {sample_id} K/B inconsistent: K={K} B={record.get('B')}")
        if int(record.get("M")) != M or int(record.get("N")) != N:
            raise ValueError(f"sample record {sample_id} K={K} dimensions mismatch")
        fmt = record.get("format")
        if fmt not in config["formats"]:
            raise ValueError(f"sample record {sample_id} K={K} unknown format {fmt!r}")
        record_type = record.get("record_type")
        if record_type == "widthselect_base_snapshot":
            key = (sample_id, K, fmt)
            if key in base:
                raise ValueError(f"duplicate base snapshot for {sample_id} K={K} format={fmt}")
            _check_lengths(sample_id, K, fmt, record, grid, ("S", "L1", "O_bits"))
            base[key] = record
        elif record_type == candidate_type:
            candidate = _candidate_key(config, record)
            if candidate not in candidate_values:
                raise ValueError(
                    f"sample record {sample_id} K={K} unexpected candidate {candidate}: not declared by the config"
                )
            key = (sample_id, K, fmt, candidate)
            if key in candidates:
                raise ValueError(f"duplicate candidate snapshot for {sample_id} K={K} format={fmt} {candidate}")
            if mode == WIDTHSELECT_ALIGN_MODE:
                _check_lengths(sample_id, K, fmt, record, grid, ("T", "Y_bits"))
            else:
                _check_lengths(sample_id, K, fmt, record, grid, ("C_state", "C_exact", "Y_bits"))
            candidates[key] = record
        else:
            raise ValueError(f"unknown sample record type {record_type!r}")
    if set(base) != expected_base:
        missing = sorted(expected_base - set(base))[:5]
        extra = sorted(set(base) - expected_base)[:5]
        raise ValueError(f"base snapshot set mismatch: missing={missing} extra={extra}")
    if set(candidates) != expected_candidates:
        missing = sorted(expected_candidates - set(candidates))[:5]
        extra = sorted(set(candidates) - expected_candidates)[:5]
        raise ValueError(f"candidate snapshot set mismatch: missing={missing} extra={extra}")
    if verbose:
        print(
            f"[width-analyze] verified {len(base)} base and {len(candidates)} candidate snapshots"
        )
    return base, candidates


def _verify_candidate(
    mode: str,
    sample_id: str,
    K: int,
    fmt: str,
    candidate: tuple,
    base_record: dict,
    record: dict,
) -> tuple[list[Dyadic | None], int]:
    """Recompute ``Y_bits`` and return ``(trajectory values, marker count)``.

    For ``align`` the trajectory is the exact ``T``; for the accumulator runs
    it is the exact ``decode(C_B)`` value per element (``None`` where the state
    is an overflow marker, which has no finite value and is counted separately).
    """
    grid = len(base_record["O_bits"])
    if mode == WIDTHSELECT_ALIGN_MODE:
        values = [Dyadic.from_dict(item) for item in record["T"]]
        for index in range(grid):
            if _recompute_bits(values[index]) != int(record["Y_bits"][index]):
                raise ValueError(
                    f"Y_bits integrity failure for {sample_id} K={K} format={fmt} candidate={candidate} "
                    f"element {index}"
                )
        for index in range(grid):
            if _recompute_bits(Dyadic.from_dict(base_record["S"][index])) != int(base_record["O_bits"][index]):
                raise ValueError(
                    f"O_bits integrity failure for {sample_id} K={K} format={fmt} element {index}"
                )
        return values, 0

    values: list[Dyadic | None] = [None] * grid
    markers = 0
    for index in range(grid):
        state = HwAccState.from_dict(record["C_state"][index])
        if state.is_overflow_marker:
            markers += 1
            if record["C_exact"][index] is not None:
                raise ValueError(f"overflow marker must not carry a decoded value at element {index}")
            expected_bits = 0x7F80_0000 | (0x8000_0000 if state.M < 0 else 0)
            if int(record["Y_bits"][index]) != expected_bits:
                raise ValueError(f"marker output bits mismatch at element {index}")
            continue
        decoded = decode_state(state)
        if Dyadic.from_dict(record["C_exact"][index]) != decoded:
            raise ValueError(
                f"C_exact integrity failure for {sample_id} K={K} format={fmt} candidate={candidate} "
                f"element {index}"
            )
        if _recompute_bits(decoded) != int(record["Y_bits"][index]):
            raise ValueError(
                f"Y_bits integrity failure for {sample_id} K={K} format={fmt} candidate={candidate} "
                f"element {index}"
            )
        values[index] = decoded
    for index in range(grid):
        if _recompute_bits(Dyadic.from_dict(base_record["S"][index])) != int(base_record["O_bits"][index]):
            raise ValueError(f"O_bits integrity failure for {sample_id} K={K} format={fmt} element {index}")
    return values, markers


def _pass_reasons(cell: dict, criteria: dict) -> list[str]:
    reasons: list[str] = []
    rate = cell["metrics"]["bit_match_rate"]
    minimum = criteria.get("min_bit_match_rate")
    if minimum is not None and (rate is None or rate < minimum):
        reasons.append("bit_match_rate_below_min")
    conditional = cell["metrics"]["conditional_ulp"]
    for key, criterion in (("p99", criteria.get("max_ulp_p99")), ("max", criteria.get("max_ulp_max"))):
        if criterion is None:
            continue
        value = conditional.get(key)
        if value is not None and value > criterion:
            reasons.append(f"ulp_{key}_above_max")
    if criteria.get("nonfinite_policy") == "reject":
        if cell["metrics"]["counts"]["n_nonfinite_candidate"] > 0:
            reasons.append("nonfinite_candidate_present")
        if cell.get("state_marker_count", 0) > 0:
            reasons.append("acc_overflow_marker_present")
    return reasons


def _recommendation(cells: list[dict], criteria: dict, mode: str) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for cell in cells:
        groups.setdefault((cell["format"], cell["distribution_pair"], cell["K"]), []).append(cell)
    by_group = []
    for key in sorted(groups):
        members = groups[key]
        if mode == WIDTHSELECT_JOINT_MODE:
            members = sorted(members, key=lambda item: (item["w_align"], item["w_acc"]))
        elif mode == WIDTHSELECT_ACC_MODE:
            members = sorted(members, key=lambda item: item["w_acc"])
        else:
            members = sorted(members, key=lambda item: item["w_align"])
        tested = []
        passing = []
        for cell in members:
            reasons = _pass_reasons(cell, criteria)
            tested.append(
                {
                    "candidate": {"w_align": cell["w_align"], "w_acc": cell["w_acc"]},
                    "passes": not reasons,
                    "reasons": reasons,
                }
            )
            if not reasons:
                passing.append(cell)
        group = {
            "format": key[0],
            "distribution_pair": key[1],
            "K": key[2],
            "B": key[2] // MAC_PARALLELISM,
            "tested": tested,
        }
        if mode == WIDTHSELECT_JOINT_MODE:
            group["passing_candidates"] = [
                {"w_align": cell["w_align"], "w_acc": cell["w_acc"]} for cell in passing
            ]
        else:
            label = "w_acc" if mode == WIDTHSELECT_ACC_MODE else "w_align"
            group["minimal_passing_candidate"] = min(cell[label] for cell in passing) if passing else None
        by_group.append(group)
    return {
        "criteria": criteria,
        "by_group": by_group,
        "note": (
            "minimal passing candidate within the tested widths, distributions and K only; "
            "not a hardware guarantee and not an extrapolation"
        ),
    }


def _csv_rows(summary: dict) -> list[dict]:
    rows = []
    for cell in summary["cells"]:
        metrics = cell["metrics"]
        conditional = metrics["conditional_ulp"]
        aux = metrics["aux"]
        series = cell["t_minus_s"] or cell["c_minus_s"] or {}
        rows.append(
            {
                "format": cell["format"],
                "activation_distribution": cell["activation_distribution"],
                "weight_distribution": cell["weight_distribution"],
                "w_align": "" if cell["w_align"] is None else cell["w_align"],
                "w_acc": "" if cell["w_acc"] is None else cell["w_acc"],
                "K": cell["K"],
                "B": cell["B"],
                "n_total": metrics["counts"]["n_total"],
                "n_bit_match": metrics["counts"]["n_bit_match"],
                "bit_match_rate": "" if metrics["bit_match_rate"] is None else repr(metrics["bit_match_rate"]),
                "n_finite_mismatch": metrics["counts"]["n_finite_mismatch"],
                "signed_zero_mismatch": metrics["counts"]["signed_zero_mismatch"],
                "n_nonfinite_candidate": metrics["counts"]["n_nonfinite_candidate"],
                "state_marker_count": cell["state_marker_count"],
                "ulp_p50": conditional["p50"],
                "ulp_p90": conditional["p90"],
                "ulp_p99": conditional["p99"],
                "ulp_max": conditional["max"],
                "bucket_0": metrics["distance_buckets"]["buckets"]["0"]["count"],
                "bucket_1": metrics["distance_buckets"]["buckets"]["1"]["count"],
                "bucket_2": metrics["distance_buckets"]["buckets"]["2"]["count"],
                "bucket_3_4": metrics["distance_buckets"]["buckets"]["3-4"]["count"],
                "bucket_ge5": metrics["distance_buckets"]["buckets"][">=5"]["count"],
                "abs_p50": aux["abs_error"]["p50"],
                "abs_p99": aux["abs_error"]["p99"],
                "abs_max": aux["abs_error"]["max"],
                "eta_p99": aux["eta_l1"]["p99"],
                "relative_l2": aux["relative_l2"]["value"],
                "series_kind": cell["series_kind"],
                "series_p50": series.get("abs_error", {}).get("p50"),
                "series_p99": series.get("abs_error", {}).get("p99"),
                "series_max": series.get("abs_error", {}).get("max"),
                "series_relative_l2": series.get("relative_l2", {}).get("value"),
            }
        )
    return rows


CSV_FIELDS = [
    "format",
    "activation_distribution",
    "weight_distribution",
    "w_align",
    "w_acc",
    "K",
    "B",
    "n_total",
    "n_bit_match",
    "bit_match_rate",
    "n_finite_mismatch",
    "signed_zero_mismatch",
    "n_nonfinite_candidate",
    "state_marker_count",
    "ulp_p50",
    "ulp_p90",
    "ulp_p99",
    "ulp_max",
    "bucket_0",
    "bucket_1",
    "bucket_2",
    "bucket_3_4",
    "bucket_ge5",
    "abs_p50",
    "abs_p99",
    "abs_max",
    "eta_p99",
    "relative_l2",
    "series_kind",
    "series_p50",
    "series_p99",
    "series_max",
    "series_relative_l2",
]


def analyze_width_run(run_dir, out_dir=None, write_plots: bool = False) -> Path:
    source = run_path(run_dir)
    manifest = load_manifest(source)
    if manifest.get("schema_version") != WIDTHSELECT_SCHEMA_VERSION:
        raise ValueError(
            f"width-select analysis expects schema {WIDTHSELECT_SCHEMA_VERSION!r}, got "
            f"{manifest.get('schema_version')!r}; historical runs are never read by this analyzer"
        )
    mode = manifest.get("run_kind")
    if mode not in (WIDTHSELECT_ALIGN_MODE, WIDTHSELECT_ACC_MODE, WIDTHSELECT_JOINT_MODE):
        raise ValueError(f"unsupported run kind {mode!r}")
    if manifest.get("status") != "completed":
        raise ValueError(f"source run status is {manifest.get('status')!r}, not 'completed'")

    config = manifest["config"]
    expected_inputs, expected_base, expected_candidates, sample_meta = _expected_identities(config)
    counts = manifest.get("counts") or {}
    if int(counts.get("matrix_pairs", -1)) != len(expected_inputs):
        raise ValueError("manifest counts.matrix_pairs is inconsistent with its identity set")
    if int(counts.get("base_snapshot_records", -1)) != len(expected_base):
        raise ValueError("manifest counts.base_snapshot_records is inconsistent with its identity set")
    if int(counts.get("width_snapshot_records", -1)) != len(expected_candidates):
        raise ValueError("manifest counts.width_snapshot_records is inconsistent with its identity set")

    _, input_events = _load_inputs(source, config, expected_inputs, verbose=True)
    base, candidates = _load_snapshots(
        source, config, expected_base, expected_candidates, verbose=True
    )

    metrics = MetricsCollector()
    t_minus = ScalarErrorCollector()
    c_minus = ScalarErrorCollector()
    marker_counts: dict[tuple, int] = {}
    for key in sorted(candidates):
        sample_id, K, fmt, candidate = key
        base_record = base[(sample_id, K, fmt)]
        record = candidates[key]
        S = [Dyadic.from_dict(item) for item in base_record["S"]]
        L1 = [Dyadic.from_dict(item) for item in base_record["L1"]]
        values, markers = _verify_candidate(mode, sample_id, K, fmt, candidate, base_record, record)
        meta = sample_meta[sample_id]
        pair_label = f"{meta['activation_distribution']}|{meta['weight_distribution']}"
        cell_key = (fmt, pair_label, candidate, K)
        pair_key = cell_key + (sample_id,)
        metrics.add(
            cell_key, pair_key, base_record["O_bits"], record["Y_bits"], S, L1, meta["seed"]
        )
        if mode == WIDTHSELECT_ALIGN_MODE:
            t_minus.add(cell_key, pair_key, [values[i] - S[i] for i in range(len(S))], S, L1)
        else:
            marker_counts[cell_key] = marker_counts.get(cell_key, 0) + markers
            finite = [index for index in range(len(values)) if values[index] is not None]
            c_minus.add(
                cell_key,
                pair_key,
                [values[index] - S[index] for index in finite],
                [S[index] for index in finite],
                [L1[index] for index in finite],
            )

    metrics_final = metrics.finalize()
    t_final = t_minus.finalize()
    c_final = c_minus.finalize()

    event_lookup = {}
    for key, record in candidates.items():
        sample_id, K, fmt, candidate = key
        meta = sample_meta[sample_id]
        pair_label = f"{meta['activation_distribution']}|{meta['weight_distribution']}"
        event_lookup[(fmt, pair_label, candidate, K)] = record["events_cumulative"]

    def _cell_payload(key: tuple) -> dict:
        fmt, pair_label, candidate, K = key
        activation, weight = pair_label.split("|")
        fields = _candidate_fields(candidate)
        return {
            "format": fmt,
            "distribution_pair": pair_label,
            "activation_distribution": activation,
            "weight_distribution": weight,
            "w_align": fields["w_align"],
            "w_acc": fields["w_acc"],
            "K": K,
            "B": K // MAC_PARALLELISM,
            "state_marker_count": marker_counts.get(key, 0),
            "series_kind": "t_minus_s" if mode == WIDTHSELECT_ALIGN_MODE else "c_minus_s",
            "metrics": metrics_final["cells"][key],
            "t_minus_s": t_final["cells"].get(key),
            "c_minus_s": c_final["cells"].get(key),
            "events_cumulative": event_lookup.get(key),
        }

    cells = [_cell_payload(key) for key in sorted(metrics_final["cells"])]
    by_pair = []
    for key in sorted(metrics_final["by_pair"]):
        fmt, pair_label, candidate, K, sample_id = key
        meta = sample_meta[sample_id]
        fields = _candidate_fields(candidate)
        by_pair.append(
            {
                "format": fmt,
                "distribution_pair": pair_label,
                "w_align": fields["w_align"],
                "w_acc": fields["w_acc"],
                "K": K,
                "B": K // MAC_PARALLELISM,
                "sample_id": sample_id,
                "seed": meta["seed"],
                "pair_index": meta["pair_index"],
                "series_kind": "t_minus_s" if mode == WIDTHSELECT_ALIGN_MODE else "c_minus_s",
                "metrics": metrics_final["by_pair"][key],
                "t_minus_s": t_final["by_pair"].get(key),
                "c_minus_s": c_final["by_pair"].get(key),
            }
        )

    criteria = manifest.get("pass_criteria")
    recommendation = _recommendation(cells, criteria, mode) if criteria else None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if out_dir is None:
        out_dir = Path("results/numerical/widthselect/analysis") / f"{manifest['run_id']}_analysis_{stamp}"
    out_path = Path(out_dir)
    resolved_source = source.resolve()
    if resolved_source == out_path.resolve() or resolved_source in out_path.resolve().parents:
        raise ValueError("analysis output directory must be outside the source run directory")
    out_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": WIDTHSELECT_ANALYSIS_VERSION,
        "analysis_kind": ANALYSIS_KIND[mode],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": {
            "run_id": manifest["run_id"],
            "run_kind": mode,
            "run_dir": str(source.resolve()),
            "schema_version": manifest["schema_version"],
            "config_hash": manifest.get("config_hash"),
            "numeric_source_digest": manifest.get("source_digest"),
            "config": config,
            "policies": manifest.get("policies"),
        },
        "analyzer_source_digest": source_digest(),
        "criteria": criteria,
        "input_quantization_events": input_events,
        "x_name": "K",
        "B_rule": "B = K // 16",
        "formats": list(config["formats"]),
        "W_align_list": list(config["W_align_list"]),
        "W_acc_list": list(config["W_acc_list"]),
        "joint_pairs": [dict(pair) for pair in config["joint_pairs"]],
        "K_list": list(config["K_list"]),
        "B_list": [int(k) // MAC_PARALLELISM for k in config["K_list"]],
        "cells": cells,
        "by_pair": by_pair,
        "recommendation": recommendation,
        "notes": NOTES,
    }
    plot_paths: list[Path] = []
    if write_plots:
        from .plots import render_plots

        plot_paths = render_plots(summary, out_path / "plots")
        summary["plots"] = [path.name for path in plot_paths]
    write_json(out_path / "summary.json", summary)
    with (out_path / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(summary))
    write_json(
        out_path / "analysis_manifest.json",
        {
            "analysis_kind": ANALYSIS_KIND[mode],
            "analysis_schema_version": WIDTHSELECT_ANALYSIS_VERSION,
            "source_run_dir": str(source.resolve()),
            "source_run_id": manifest["run_id"],
            "source_run_kind": mode,
            "source_files": {
                name: file_identity(source / name)
                for name in ("manifest.json", WIDTH_INPUTS_NAME, WIDTH_SAMPLES_NAME)
            },
            "numeric_run_source_digest": manifest.get("source_digest"),
            "analyzer_source_digest": source_digest(),
            "plots": [path.name for path in plot_paths],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    print(f"[width-analyze] wrote {out_path}")
    return out_path
