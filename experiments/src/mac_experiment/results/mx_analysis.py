"""Offline output-quantisation analysis for MX matrix runs.

Reads only the saved input matrices and exact/FP32 snapshot records, then

* quantises exact ``S`` (independent scale) and FP32 ``R`` (independent and
  shared/rescale comparisons);
* accumulates scale/group/element/decoded match statistics, per-group mismatch
  histograms, scale exponent differences, conditional E4M3 ULP distances,
  decoded absolute errors and relative L2;
* adds the existing error metrics for ``R - S`` (accumulation error) and
  ``decode(Q_R) - S`` (accumulation + output quantisation).

The sampler and the numerical execution engine are never imported; source runs
are never modified.  Output goes to an independent directory so that changing
the scale policy, quantiser or plots never requires re-running the MAC.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..formats.e4m3_element import is_nan_code
from ..matrix.quantize import (
    GROUP_LENGTH,
    decoded_group_values,
    group_index_lists,
    quantize_matrix_independent,
    quantize_matrix_shared,
)
from ..matrix.inputs import bits_are_finite
from ..matrix.tensor import MxTensor
from ..metrics.compare import ComparisonAccumulator
from ..metrics.mx_quant import main_accumulator, shared_accumulator
from ..version import (
    MX_ACCUMULATION_POLICY,
    MX_INPUT_QUANTIZATION_POLICY,
    MX_INPUT_SAMPLING_VERSION,
    MX_INPUT_STREAM_VERSION,
    MX_OUTPUT_SCALE_POLICY,
    MX_SCHEMA_VERSION,
    PACKAGE_VERSION,
)
from .enhance import _file_identity
from .mx_io import iter_mx_inputs, iter_mx_samples
from .reader import load_manifest, require_schema, run_path
from .schema import MX_ANALYSIS_VERSION, MX_SUMMARY_VERSION
from .writer import write_json

SOURCE_FILES = ("manifest.json", "inputs.jsonl.gz", "samples.jsonl.gz")

LEGACY_INPUT_ERROR = (
    "this run uses the retired v1 MX input scheme (independent E4M3 element + E8M0 scale "
    "sampling); the current analyzer only processes mac-mx-exp-v2 runs with "
    "input_sampling_version='%s'. Historical v1 runs stay read-only and are excluded from "
    "default result discovery; re-run with the current input pipeline for new analyses."
    % MX_INPUT_SAMPLING_VERSION
)

POLICY = {
    "input_pipeline": (
        "numeric-domain sample -> explicit FP32 RNE conversion (bits recorded) -> G32 MXFP8 input quantization"
    ),
    "input_sampling_version": MX_INPUT_SAMPLING_VERSION,
    "input_stream_version": MX_INPUT_STREAM_VERSION,
    "input_quantization": MX_INPUT_QUANTIZATION_POLICY,
    "scope_note": (
        "measures the behaviour of MXFP8-quantized inputs under dot16+FP32 accumulation and MX output "
        "quantization; it does not measure input quantization loss relative to the raw FP32 matrix product"
    ),
    "reference": "S and L1 use the exact decoded MX input values, never the raw FP32 matrices",
    "output_group": "32 elements along N (row-major groups)",
    "output_scale_policy": MX_OUTPUT_SCALE_POLICY,
    "element_rounding": "RNE ties-to-even, subnormals kept, saturation to +/-448 (E4M3 has no infinity)",
    "exactness": "all maxima, thresholds, scale choices and roundings computed from exact dyadics",
    "zero_sign": "S canonical +0; FP32 R signed zeros preserved through quantisation",
    "accumulation": MX_ACCUMULATION_POLICY,
    "comparisons": {
        "main_independent_scale": "Q_S and Q_R each select their own group scale",
        "aux_shared_reference_scale": "Q_S and Q_R both quantised with the Q_S scale (never reselected)",
        "accum_error_r_vs_s": "existing error metrics with error = R - S, reference = S, eta denominator = input-product L1",
        "total_error_decode_qr_vs_s": "error = decode(Q_R) - S, reference = S",
    },
    "ulp": {
        "format": "E4M3 ordered finite codes",
        "condition_main": "finite elements in valid groups with equal group scales",
        "condition_shared": "all finite elements in valid groups",
        "quantiles": "nearest_rank over mismatching elements; null with status no_mismatch_samples when none",
        "signed_zero": "+0/-0 merged in distance, distinguished by code match",
    },
    "error_quantiles": "linear interpolation at (N-1)q (existing framework definition)",
    "validity": "invalid groups are counted with reasons and excluded from finite metrics; whole-batch metrics are null if any group is invalid",
    "mismatch_definitions": {
        "representation": "measure (scale_code, element_code) pairs; a group with different scales has 32/32 representation mismatches",
        "code_only": "element codes only, explicitly named code_only_*; must not be used as the primary mismatch definition",
        "decoded_value": "exact decoded dyadic values, independent of representation",
    },
    "independence": (
        "the independent repeat unit is the input matrix pair; K prefixes and the elements of one "
        "matrix are correlated, so no simple binomial confidence bound is reported"
    ),
}

NOTES = [
    "offline analysis of saved MX records; the sampler and the numerical execution engine were not imported or executed",
    "source run files were not modified",
    "the exact reference S is computed from the decoded MX inputs (not from the raw FP32 matrices)",
    "K/B scan uses paired prefixes of one maximum-K input matrix pair; inputs are stored once",
    "per-K snapshots of the same pair share inputs and scales, so they are not independent samples",
    "numeric-domain input distributions are experiment definitions and do not claim to represent real LLM tensors",
    "input diagnostics (RMS, max magnitude, zero fraction, scale histogram) are not error metrics of this experiment",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class MxCell:
    activation_distribution: str
    weight_distribution: str
    K: int
    B: int
    M: int = 0
    N: int = 0
    main: object = field(default_factory=main_accumulator)
    aux: object = field(default_factory=shared_accumulator)
    accum_error: ComparisonAccumulator = field(default_factory=lambda: ComparisonAccumulator("accum_error_r_vs_s"))
    total_error: ComparisonAccumulator = field(
        default_factory=lambda: ComparisonAccumulator("total_error_decode_qr_vs_s")
    )
    quantization_events: Counter = field(default_factory=Counter)
    n_snapshots: int = 0
    invalid_input_snapshots: int = 0
    invalid_input_reasons: Counter = field(default_factory=Counter)

    def add(self, record: dict, pair_valid: bool, pair_reason: str | None) -> None:
        self.n_snapshots += 1
        if not pair_valid:
            self.invalid_input_snapshots += 1
            self.invalid_input_reasons[pair_reason or "invalid_input"] += 1
            return
        M = int(record["M"])
        N = int(record["N"])
        self.M = M
        self.N = N
        sample_id = record["sample_id"]
        seed = int(record["seed"])
        values_s = [Dyadic.from_dict(payload) for payload in record["S"]]
        l1 = [Dyadic.from_dict(payload) for payload in record["L1"]]
        r_values: list[Dyadic | None] = []
        r_zero_signs: list[int] = []
        for bits in record["r_bits"]:
            state = Fp32.from_bits(int(bits))
            if state.is_finite():
                r_values.append(state.exact_dyadic())
                r_zero_signs.append(state.sign if state.is_zero() else 0)
            else:
                r_values.append(None)
                r_zero_signs.append(0)

        q_s = quantize_matrix_independent(values_s, None, M, N)
        q_r = quantize_matrix_independent(r_values, r_zero_signs, M, N)
        reference_scales = list(q_s.scale_codes)
        aux_s = quantize_matrix_shared(values_s, None, M, N, reference_scales)
        aux_r = quantize_matrix_shared(r_values, r_zero_signs, M, N, reference_scales)

        positions = group_index_lists(M, N, GROUP_LENGTH, 1)
        for group_index, indices in enumerate(positions):
            group_s = q_s.groups[group_index]
            group_r = q_r.groups[group_index]
            if group_s.invalid or group_r.invalid:
                reason = group_s.invalid_reason or group_r.invalid_reason or "invalid"
                self.main.add_invalid_group(sample_id, seed, len(indices), reason)
            else:
                self.main.add_group(
                    sample_id,
                    seed,
                    group_s.scale_code,
                    group_r.scale_code,
                    group_s.element_codes,
                    group_r.element_codes,
                    decoded_group_values(group_s),
                    decoded_group_values(group_r),
                )
            shared_s = aux_s.groups[group_index]
            shared_r = aux_r.groups[group_index]
            if shared_s.invalid or shared_r.invalid:
                reason = shared_s.invalid_reason or shared_r.invalid_reason or "invalid"
                self.aux.add_invalid_group(sample_id, seed, len(indices), reason)
            else:
                self.aux.add_group(
                    sample_id,
                    seed,
                    shared_s.scale_code,
                    shared_r.scale_code,
                    shared_s.element_codes,
                    shared_r.element_codes,
                    decoded_group_values(shared_s),
                    decoded_group_values(shared_r),
                )

        for index, value in enumerate(values_s):
            if r_values[index] is None:
                self.accum_error.add_invalid()
            else:
                self.accum_error.add(r_values[index] - value, value, l1[index])

        for group_index, indices in enumerate(positions):
            group_r = q_r.groups[group_index]
            if group_r.invalid:
                for _ in indices:
                    self.total_error.add_invalid()
                continue
            decoded = decoded_group_values(group_r)
            for offset, index in enumerate(indices):
                self.total_error.add(decoded[offset] - values_s[index], values_s[index], l1[index])

        for quant in (q_s, q_r, aux_s, aux_r):
            self.quantization_events.update(quant.events)

    def finalize(self) -> dict:
        return {
            "activation_distribution": self.activation_distribution,
            "weight_distribution": self.weight_distribution,
            "x": self.K,
            "K": self.K,
            "B": self.B,
            "M": self.M,
            "N": self.N,
            "comparisons": {
                "main_independent_scale": self.main.finalize(),
                "aux_shared_reference_scale": self.aux.finalize(),
                "accum_error_r_vs_s": self.accum_error.finalize(),
                "total_error_decode_qr_vs_s": self.total_error.finalize(),
            },
            "quantization_events": {
                key: int(value) for key, value in sorted(self.quantization_events.items())
            },
            "counts": {
                "snapshots": self.n_snapshots,
                "invalid_input_snapshots": self.invalid_input_snapshots,
            },
            "invalid_input_reasons": dict(self.invalid_input_reasons),
        }


class MxAnalysisBuilder:
    def __init__(self) -> None:
        self.cells: dict[tuple[str, str, int], MxCell] = {}

    def add(self, record: dict, pair_valid: bool, pair_reason: str | None = None) -> None:
        if record.get("record_type") != "matrix_snapshot":
            return
        activation = record["activation_distribution"]
        weight = record["weight_distribution"]
        K = int(record["K"])
        key = (activation, weight, K)
        cell = self.cells.get(key)
        if cell is None:
            cell = MxCell(
                activation_distribution=activation,
                weight_distribution=weight,
                K=K,
                B=int(record["B"]),
            )
            self.cells[key] = cell
        cell.add(record, pair_valid, pair_reason)

    def finalize(self, base: dict) -> dict:
        summary = dict(base)
        summary["x_name"] = "K"
        summary["x_values"] = sorted({key[2] for key in self.cells})
        summary["K_to_B"] = {str(key[2]): key[2] // 16 for key in self.cells}
        summary["cells"] = [self.cells[key].finalize() for key in sorted(self.cells)]
        return summary


def _validate_input_pair(record: dict) -> tuple[bool, str | None, dict]:
    reasons: list[str] = []
    for name in ("x", "w"):
        payload = record.get(name)
        if not isinstance(payload, dict):
            reasons.append(f"{name}:missing")
            continue
        element_codes = payload.get("element_codes")
        scale_codes = payload.get("scale_codes")
        if element_codes is None or scale_codes is None:
            reasons.append(f"{name}:missing_codes")
            continue
        if any(is_nan_code(int(code)) for code in element_codes):
            reasons.append(f"{name}:nan_element_code")
        if any(int(code) == 0xFF for code in scale_codes):
            reasons.append(f"{name}:reserved_scale_code")
        try:
            MxTensor.from_dict(payload)
        except ValueError as exc:
            reasons.append(f"{name}:invalid_tensor:{exc}")
    for name in ("x_fp32_bits", "w_fp32_bits"):
        bits = record.get(name)
        if not isinstance(bits, list) or not bits:
            reasons.append(f"{name}:missing")
            continue
        if not bits_are_finite(bits):
            reasons.append(f"{name}:nonfinite_or_malformed")
    meta = {
        "activation_distribution": record.get("activation_distribution"),
        "weight_distribution": record.get("weight_distribution"),
        "input_sampling_version": record.get("input_sampling_version"),
        "input_stream_version": record.get("input_stream_version"),
        "input_quantization_policy": record.get("input_quantization_policy"),
        "input_quantization_events": {
            "x": record.get("x_input_quantization_events") or {},
            "w": record.get("w_input_quantization_events") or {},
        },
        "input_diagnostics": {
            "x": record.get("x_input_diagnostics") or {},
            "w": record.get("w_input_diagnostics") or {},
        },
    }
    if reasons:
        return False, ";".join(reasons), meta
    return True, None, meta


def _aggregate_input_diagnostics(pair_meta: dict) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for meta in pair_meta.values():
        if not meta.get("input_sampling_version"):
            continue
        key = (meta.get("activation_distribution"), meta.get("weight_distribution"))
        entry = grouped.setdefault(
            key,
            {
                "activation_distribution": key[0],
                "weight_distribution": key[1],
                "matrix_pairs": 0,
                "input_quantization_events": {"x": Counter(), "w": Counter()},
                "fp32_rms": [],
                "mx_decoded_rms": [],
                "fp32_max_abs": [],
                "mx_max_decoded_abs": [],
                "mx_zero_fraction": [],
                "scale_exponent_histogram": Counter(),
            },
        )
        entry["matrix_pairs"] += 1
        events = meta.get("input_quantization_events") or {}
        for path in ("x", "w"):
            entry["input_quantization_events"][path].update(events.get(path) or {})
        diagnostics = meta.get("input_diagnostics") or {}
        for path in ("x", "w"):
            data = diagnostics.get(path) or {}
            if data.get("fp32_rms") is not None:
                entry["fp32_rms"].append(data["fp32_rms"])
            if data.get("mx_decoded_rms") is not None:
                entry["mx_decoded_rms"].append(data["mx_decoded_rms"])
            if data.get("fp32_max_abs") is not None:
                entry["fp32_max_abs"].append(data["fp32_max_abs"])
            if data.get("mx_max_decoded_abs") is not None:
                entry["mx_max_decoded_abs"].append(data["mx_max_decoded_abs"])
            if data.get("mx_zero_fraction") is not None:
                entry["mx_zero_fraction"].append(data["mx_zero_fraction"])
            for exponent, count in (data.get("scale_exponent_histogram") or {}).items():
                entry["scale_exponent_histogram"][exponent] += int(count)
    result = []
    for key in sorted(grouped):
        entry = grouped[key]

        def mean(values: list[float]):
            return sum(values) / len(values) if values else None

        def maximum(values: list[float]):
            return max(values) if values else None

        result.append(
            {
                "activation_distribution": entry["activation_distribution"],
                "weight_distribution": entry["weight_distribution"],
                "matrix_pairs": entry["matrix_pairs"],
                "definition": "input diagnostics only; not error metrics of this experiment",
                "input_quantization_events": {
                    path: {k: int(v) for k, v in sorted(entry["input_quantization_events"][path].items())}
                    for path in ("x", "w")
                },
                "fp32_rms_mean": mean(entry["fp32_rms"]),
                "mx_decoded_rms_mean": mean(entry["mx_decoded_rms"]),
                "fp32_max_abs_max": maximum(entry["fp32_max_abs"]),
                "mx_max_decoded_abs_max": maximum(entry["mx_max_decoded_abs"]),
                "mx_zero_fraction_mean": mean(entry["mx_zero_fraction"]),
                "scale_exponent_histogram": {
                    key: int(entry["scale_exponent_histogram"][key])
                    for key in sorted(entry["scale_exponent_histogram"], key=lambda item: int(item))
                },
            }
        )
    return result


def analyze_mx_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    write_plots: bool = True,
) -> Path:
    source = run_path(run_dir)
    manifest = load_manifest(source)
    require_schema(manifest, (MX_SCHEMA_VERSION,))
    if manifest.get("input_sampling_version") != MX_INPUT_SAMPLING_VERSION:
        raise ValueError(LEGACY_INPUT_ERROR)

    if out_dir is None:
        out = Path("results/numerical/mxfp8/analysis") / f"{manifest['run_id']}_quant_{_timestamp()}"
    else:
        out = Path(out_dir)
    source_resolved = source.resolve()
    out_resolved = out.resolve()
    if out_resolved == source_resolved or source_resolved in out_resolved.parents:
        raise ValueError(f"output directory must not be inside the source run: {out}")
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True)

    pair_meta: dict[str, dict] = {}
    pair_status: dict[str, tuple[bool, str | None, dict]] = {}
    n_input_records = 0
    for record in iter_mx_inputs(source):
        verdict = _validate_input_pair(record)
        pair_status[record["sample_id"]] = verdict
        pair_meta[record["sample_id"]] = verdict[2]
        n_input_records += 1

    builder = MxAnalysisBuilder()
    n_records = 0
    n_snapshots = 0
    for record in iter_mx_samples(source):
        n_records += 1
        if record.get("record_type") != "matrix_snapshot":
            continue
        n_snapshots += 1
        valid, reason, _ = pair_status.get(record["sample_id"], (False, "missing_input_record", {}))
        builder.add(record, valid, reason)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {
        "schema_version": MX_SCHEMA_VERSION,
        "summary_version": MX_SUMMARY_VERSION,
        "analysis_version": MX_ANALYSIS_VERSION,
        "run_id": f"{manifest['run_id']}+mx_quant",
        "source_run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "fixed": manifest["fixed"],
        "config": manifest["config"],
        "config_hash": manifest.get("config_hash"),
        "created_utc": created,
        "source_created_utc": manifest.get("created_utc"),
        "source_digest": manifest.get("source_digest"),
        "input_sampling_version": manifest.get("input_sampling_version"),
        "input_stream_version": manifest.get("input_stream_version"),
        "input_quantization_policy": manifest.get("input_quantization_policy"),
        "distribution_pairs": manifest.get("distribution_pairs"),
        "n_input_records_read": n_input_records,
        "n_records_read": n_records,
        "n_snapshots_analyzed": n_snapshots,
        "input_diagnostics": _aggregate_input_diagnostics(pair_meta),
        "policy": POLICY,
        "notes": NOTES,
    }
    summary = builder.finalize(base)
    write_json(out / "summary.json", summary)
    write_mx_csv(out / "summary.csv", summary)

    source_files = {}
    for name in SOURCE_FILES:
        path = source / name
        if path.exists():
            source_files[name] = _file_identity(path)
    analysis_manifest = {
        "analysis_version": PACKAGE_VERSION,
        "analysis_kind": "mx_output_quantization",
        "summary_version": MX_SUMMARY_VERSION,
        "schema_version": MX_SCHEMA_VERSION,
        "created_utc": created,
        "source": {
            "run_dir": str(source_resolved),
            "run_id": manifest["run_id"],
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
            "config_hash": manifest.get("config_hash"),
            "input_sampling_version": manifest.get("input_sampling_version"),
            "input_stream_version": manifest.get("input_stream_version"),
            "input_quantization_policy": manifest.get("input_quantization_policy"),
            "files": source_files,
        },
        "input_records_read": n_input_records,
        "records_read": n_records,
        "snapshots_analyzed": n_snapshots,
        "policy": POLICY,
        "notes": NOTES,
    }
    write_json(out / "analysis_manifest.json", analysis_manifest)

    if write_plots:
        from ..plotting.mx_plots import plot_mx_summary

        plot_mx_summary(summary, out / "plots")
    return out


def _coverage_text(coverage) -> str:
    if not isinstance(coverage, dict):
        return ""
    return " ".join(f"{key}={value}" for key, value in sorted(coverage.items()))


def write_mx_csv(path: str | Path, summary: dict) -> None:
    import csv

    fieldnames = [
        "activation_distribution",
        "weight_distribution",
        "x_name",
        "K",
        "B",
        "comparison",
        "metric",
        "value",
        "status",
        "denominator",
        "note",
    ]
    rows: list[dict] = []
    for cell in summary["cells"]:
        base = {
            "activation_distribution": cell["activation_distribution"],
            "weight_distribution": cell["weight_distribution"],
            "x_name": summary["x_name"],
            "K": cell["K"],
            "B": cell["B"],
        }

        def emit(comparison: str, metric: str, value, status: str, denominator="", note: str = "") -> None:
            if value is None:
                text = ""
            elif isinstance(value, float):
                text = repr(value)
            else:
                text = str(value)
            rows.append(
                {
                    **base,
                    "comparison": comparison,
                    "metric": metric,
                    "value": text,
                    "status": status,
                    "denominator": denominator,
                    "note": note,
                }
            )

        for comparison, payload in cell["comparisons"].items():
            emit(comparison, "_status", payload.get("status"), "ok", _coverage_text(payload.get("coverage")))
            for scope_key, scope_label, scope_payload in (
                ("metrics", "whole_batch", payload),
                ("finite_subset", "finite_subset", payload.get("finite_subset")),
            ):
                if not scope_payload or not isinstance(scope_payload.get("metrics"), dict):
                    continue
                for metric, entry in scope_payload["metrics"].items():
                    emit(
                        comparison,
                        metric,
                        entry["value"],
                        entry["status"],
                        f"scope:{scope_label}",
                        scope_payload.get("note", "") if scope_key == "finite_subset" else "",
                    )
                if scope_key != "metrics":
                    continue
                distributions = scope_payload.get("distributions")
                if isinstance(distributions, dict) and distributions.get("ulp") is not None:
                    ulp = distributions["ulp"]
                    for key in ("p50", "p90", "p99", "max"):
                        emit(
                            comparison,
                            f"ulp_{key}",
                            ulp[key],
                            ulp["status"],
                            f"n_mismatch={ulp['n_mismatch']}",
                            ulp["condition"],
                        )
                    diff = distributions["scale_exponent_diff"]
                    for key in ("negative_rate", "zero_rate", "positive_rate"):
                        emit(
                            comparison,
                            f"scale_exponent_diff_{key}",
                            diff[key],
                            "ok",
                            f"groups={diff['denominator']}",
                        )
                    mismatch = distributions["group_representation_mismatch_elements"]
                    emit(
                        comparison,
                        "group_representation_mismatch_mean",
                        mismatch["mean"],
                        "ok" if mismatch["mean"] is not None else "no_valid_groups",
                        f"groups={mismatch['denominator']}",
                        mismatch["definition"],
                    )
                    code_only = distributions["group_code_only_mismatch_elements"]
                    emit(
                        comparison,
                        "group_code_only_mismatch_mean",
                        code_only["mean"],
                        "ok" if code_only["mean"] is not None else "no_valid_groups",
                        f"groups={code_only['denominator']}",
                        code_only["definition"],
                    )
                counts = scope_payload.get("counts")
                if isinstance(counts, dict):
                    for key, value in counts.items():
                        emit(comparison, f"count_{key}", value, "ok")
        events = cell.get("quantization_events", {})
        for key, value in events.items():
            emit("quantization", f"event_{key}", value, "ok")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
