"""Offline enhancement of existing runs with the ``alignment_only`` comparison.

Reads only saved sample records through the streaming reader and re-aggregates
them with the shared :class:`SummaryBuilder`.  It never imports or invokes the
sampler or the numerical execution engine, never re-samples and never re-runs
MAC arithmetic.  Output is written to an independent directory; the source run
(manifest, samples, summary, plots) is never modified.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ..metrics.aggregate import COMPARISONS, SummaryBuilder
from ..version import PACKAGE_VERSION, SCHEMA_VERSION, SUPPORTED_SCHEMAS
from .reader import iter_records, load_manifest, require_schema, run_path
from .schema import SUMMARY_VERSION
from .writer import write_json, write_summary_csv

SOURCE_FILES = ("manifest.json", "samples.jsonl.gz", "summary.json", "summary.csv")


def _file_identity(path: Path, chunk: int = 1 << 20) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return {"size": path.stat().st_size, "sha256": "sha256:" + digest.hexdigest()}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def enhance_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    write_plots: bool = True,
) -> Path:
    source = run_path(run_dir)
    manifest = load_manifest(source)
    require_schema(manifest, SUPPORTED_SCHEMAS)

    if out_dir is None:
        out = Path("results/numerical/dot_product/analysis/alignment_only") / f"{manifest['run_id']}_alignment_only_{_timestamp()}"
    else:
        out = Path(out_dir)
    source_resolved = source.resolve()
    out_resolved = out.resolve()
    if out_resolved == source_resolved or source_resolved in out_resolved.parents:
        raise ValueError(f"output directory must not be inside the source run: {out}")
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True)

    builder = SummaryBuilder(mode=manifest["mode"], x_name=manifest["x_name"])
    n_records = 0
    for record in iter_records(source):
        builder.add(record)
        n_records += 1

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {
        "schema_version": SCHEMA_VERSION,
        "summary_version": SUMMARY_VERSION,
        "input_schema_version": manifest.get("schema_version"),
        "run_id": f"{manifest['run_id']}+alignment_only",
        "source_run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "config": manifest["config"],
        "config_hash": manifest["config_hash"],
        "fixed": manifest["fixed"],
        "created_utc": created,
        "source_created_utc": manifest.get("created_utc"),
        "source_digest": manifest.get("source_digest"),
        "n_records_read": n_records,
    }
    summary = builder.finalize(base)
    write_json(out / "summary.json", summary)
    write_summary_csv(out / "summary.csv", summary)

    source_files = {}
    for name in SOURCE_FILES:
        path = source / name
        if path.exists():
            source_files[name] = _file_identity(path)
    analysis_manifest = {
        "analysis_version": PACKAGE_VERSION,
        "analysis_kind": "alignment_only_enhancement",
        "summary_version": SUMMARY_VERSION,
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
        "comparisons": list(COMPARISONS),
        "notes": [
            "offline re-aggregation of saved records; the sampler and the numerical execution engine were not imported or executed",
            "source run files were not modified",
            "alignment_only uses the stored exact E_align = Q_B - S_B; Q_B is never rounded to FP32",
            "metrics for total/window_effect/fp32_baseline are recomputed from the same records and must match the source summary for valid samples",
        ],
    }
    write_json(out / "analysis_manifest.json", analysis_manifest)

    if write_plots:
        from ..plotting import plot_all

        plot_all(summary, out / "plots")
    return out
