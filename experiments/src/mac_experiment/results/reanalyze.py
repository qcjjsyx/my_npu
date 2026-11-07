"""Offline re-analysis of saved sample records."""

from __future__ import annotations

from pathlib import Path

from ..metrics.aggregate import SummaryBuilder
from ..version import SCHEMA_VERSION, SUPPORTED_SCHEMAS
from .reader import iter_records, load_manifest, load_summary, require_schema, run_path
from .schema import SUMMARY_VERSION
from .writer import REANALYSIS_NAME, write_json


def rebuild_summary(run_dir: str | Path) -> dict:
    manifest = load_manifest(run_dir)
    require_schema(manifest, SUPPORTED_SCHEMAS)
    builder = SummaryBuilder(mode=manifest["mode"], x_name=manifest["x_name"])
    n_records = 0
    for record in iter_records(run_dir):
        builder.add(record)
        n_records += 1
    base = {
        "schema_version": SCHEMA_VERSION,
        "summary_version": SUMMARY_VERSION,
        "input_schema_version": manifest.get("schema_version"),
        "run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "config": manifest["config"],
        "config_hash": manifest["config_hash"],
        "fixed": manifest["fixed"],
        "created_utc": manifest["created_utc"],
        "source_digest": manifest.get("source_digest"),
        "n_records_read": n_records,
    }
    return builder.finalize(base)


def reanalyze(run_dir: str | Path, write: bool = True) -> dict:
    path = run_path(run_dir)
    recomputed = rebuild_summary(path)
    original = load_summary(path)
    differences = compare_summaries(original, recomputed)
    recomputed["matches_original_summary"] = not differences
    recomputed["differences"] = differences
    if write:
        write_json(path / REANALYSIS_NAME, recomputed)
    return recomputed


def compare_summaries(original: dict, recomputed: dict) -> list[str]:
    differences: list[str] = []

    def walk(a, b, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a:
                    differences.append(f"{prefix}.{key}: missing in original")
                elif key not in b:
                    differences.append(f"{prefix}.{key}: missing in recomputed")
                else:
                    walk(a[key], b[key], f"{prefix}.{key}")
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append(f"{prefix}: length {len(a)} != {len(b)}")
                return
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                walk(item_a, item_b, f"{prefix}[{index}]")
            return
        if a != b:
            differences.append(f"{prefix}: {a!r} != {b!r}")

    skip = {
        "created_utc",
        "n_records_read",
        "input_schema_version",
        "matches_original_summary",
        "differences",
        "status",
    }
    for key in sorted(set(original) | set(recomputed)):
        if key in skip:
            continue
        if key not in original:
            differences.append(f"{key}: missing in original")
        elif key not in recomputed:
            differences.append(f"{key}: missing in recomputed")
        else:
            walk(original[key], recomputed[key], key)
    return differences
