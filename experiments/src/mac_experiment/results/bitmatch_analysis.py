"""Offline bit-match / ULP-distance analysis of saved runs.

Reads only saved sample records through the streaming reader.  It never
imports or invokes the sampler or the numerical execution engine, never
re-samples and never re-runs MAC arithmetic.  Output goes to an independent
directory; the source run (manifest, samples, summary, plots) is never
modified.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..metrics.bitmatch import AUX_PAIRS, BUCKET_LABELS, PRIMARY_PAIRS, QUANTILE_METHOD, BitMatchBuilder
from ..version import PACKAGE_VERSION, SCHEMA_VERSION, SUPPORTED_SCHEMAS
from .enhance import _file_identity
from .reader import iter_records, load_manifest, require_schema, run_path
from .schema import BITMATCH_SUMMARY_VERSION
from .writer import write_json

SOURCE_FILES = ("manifest.json", "samples.jsonl.gz", "summary.json", "summary.csv")

POLICY = {
    "rounding": "RNE (round-to-nearest, ties-to-even)",
    "subnormals": "kept (no FTZ)",
    "overflow": "rounds to +/-Inf, no saturation",
    "bf16_direct": "o16 = RN_BF16(S_B) computed directly from the exact dyadic; no FP32/FP64 intermediate",
    "zero_sign": "exact zero S_B has no stored sign; canonical reference zero is +0",
    "bit_match": "full encoding comparison; +0 and -0 are different encodings",
    "ulp_distance": (
        "steps between values in the ordered sequence of representable finite numbers of the target "
        "format; exact integer arithmetic; +0/-0 merged; finite pairs only"
    ),
    "conditional_quantiles": QUANTILE_METHOD,
}

NOTES = [
    "offline analysis of saved records; the sampler and the numerical execution engine were not imported or executed",
    "source run files were not modified",
    "candidate a_B is not used anywhere in these comparisons",
    "aux_double_rounding and aux_accum_bf16 are auxiliary diagnostics; their rates/distances must not be added together",
    "W sweep: per-W cells are kept separate; the same sample's S_B/r_B repeats across W by construction and is never pooled across W",
    "seeds are pooled by sample; quantiles are computed from pooled per-sample distances, never by averaging per-seed quantiles",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def analyze_bitmatch(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    write_plots: bool = True,
) -> Path:
    source = run_path(run_dir)
    manifest = load_manifest(source)
    require_schema(manifest, SUPPORTED_SCHEMAS)

    if out_dir is None:
        out = Path("results/numerical/dot_product/analysis/bitmatch") / f"{manifest['run_id']}_bitmatch_ulp_{_timestamp()}"
    else:
        out = Path(out_dir)
    source_resolved = source.resolve()
    out_resolved = out.resolve()
    if out_resolved == source_resolved or source_resolved in out_resolved.parents:
        raise ValueError(f"output directory must not be inside the source run: {out}")
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True)

    builder = BitMatchBuilder(mode=manifest["mode"], x_name=manifest["x_name"])
    n_records = 0
    n_snapshots = 0
    for record in iter_records(source):
        n_records += 1
        if record.get("record_type") == "snapshot":
            n_snapshots += 1
        builder.add(record)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {
        "schema_version": SCHEMA_VERSION,
        "summary_version": BITMATCH_SUMMARY_VERSION,
        "input_schema_version": manifest.get("schema_version"),
        "run_id": f"{manifest['run_id']}+bitmatch_ulp",
        "source_run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "config": manifest["config"],
        "config_hash": manifest["config_hash"],
        "fixed": manifest["fixed"],
        "created_utc": created,
        "source_created_utc": manifest.get("created_utc"),
        "source_digest": manifest.get("source_digest"),
        "n_records_read": n_records,
        "n_snapshots_analyzed": n_snapshots,
        "policy": POLICY,
        "primary_pairs": list(PRIMARY_PAIRS),
        "aux_pairs": list(AUX_PAIRS),
    }
    summary = builder.finalize(base)
    write_json(out / "summary.json", summary)
    _write_csv(out / "summary.csv", summary)

    source_files = {}
    for name in SOURCE_FILES:
        path = source / name
        if path.exists():
            source_files[name] = _file_identity(path)
    analysis_manifest = {
        "analysis_version": PACKAGE_VERSION,
        "analysis_kind": "bitmatch_ulp",
        "summary_version": BITMATCH_SUMMARY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "source": {
            "run_dir": str(source_resolved),
            "run_id": manifest["run_id"],
            "schema_version": manifest.get("schema_version"),
            "summary_version": manifest.get("summary_version"),
            "mode": manifest["mode"],
            "x_name": manifest["x_name"],
            "config_hash": manifest.get("config_hash"),
            "status": manifest.get("status"),
            "files": source_files,
        },
        "records_read": n_records,
        "snapshots_analyzed": n_snapshots,
        "policy": POLICY,
        "primary_pairs": list(PRIMARY_PAIRS),
        "aux_pairs": list(AUX_PAIRS),
        "notes": NOTES,
    }
    write_json(out / "analysis_manifest.json", analysis_manifest)

    if write_plots:
        from ..plotting.bitmatch_plots import plot_bitmatch

        plot_bitmatch(summary, out / "plots")
    return out


def _write_csv(path: Path, summary: dict) -> None:
    import csv

    fieldnames = [
        "format",
        "distribution",
        "x_name",
        "x",
        "comparison",
        "metric",
        "value",
        "status",
        "denominator",
        "note",
    ]
    rows = []
    for cell in summary["cells"]:
        for pair_name, payload in cell["pairs"].items():
            base = {
                "format": cell["format"],
                "distribution": cell["distribution"],
                "x_name": summary["x_name"],
                "x": cell["x"],
                "comparison": pair_name,
            }

            def emit(metric, value, status, denominator="", note=""):
                if value is None:
                    text = ""
                elif isinstance(value, float):
                    text = repr(value)
                else:
                    text = str(value)
                rows.append(
                    {**base, "metric": metric, "value": text, "status": status, "denominator": denominator, "note": note}
                )

            emit("n_total", payload["counts"]["n_total"], "ok")
            emit("bit_match", payload["counts"]["bit_match"], "ok")
            emit("bit_mismatch", payload["counts"]["bit_mismatch"], "ok")
            emit("signed_zero_mismatch", payload["counts"]["signed_zero_mismatch"], "ok")
            emit("n_finite_pairs", payload["counts"]["n_finite_pairs"], "ok")
            emit("n_finite_mismatch", payload["counts"]["n_finite_mismatch"], "ok")
            emit("bit_match_rate", payload["bit_match_rate"], "ok", "n_total")
            emit("bit_mismatch_rate", payload["bit_mismatch_rate"], "ok", "n_total")
            conditional = payload["conditional_ulp"]
            for key in ("p50", "p90", "p99", "max"):
                emit(f"conditional_ulp_{key}", conditional[key], conditional["status"], "n_finite_mismatch", conditional["note"])
            for label in BUCKET_LABELS:
                bucket = payload["distance_buckets"]["buckets"][label]
                emit(f"bucket_{label}_count", bucket["count"], "ok", "n_finite_pairs")
                emit(f"bucket_{label}_rate", bucket["rate"], "ok", "n_finite_pairs")
            for event, count in payload["events"].items():
                emit(f"event_{event}", count, "ok")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
