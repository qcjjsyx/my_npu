"""Run directory reader with strict JSON and schema/config checks."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

from .schema import loads_strict
from .writer import MANIFEST_NAME, SAMPLES_NAME, SUMMARY_NAME


def run_path(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    return path


def load_manifest(run_dir: str | Path) -> dict:
    path = run_path(run_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return loads_strict(path.read_text(encoding="utf-8"))


def load_summary(run_dir: str | Path) -> dict:
    path = run_path(run_dir) / SUMMARY_NAME
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    return loads_strict(path.read_text(encoding="utf-8"))


def iter_records(run_dir: str | Path) -> Iterator[dict]:
    path = run_path(run_dir) / SAMPLES_NAME
    if not path.exists():
        raise FileNotFoundError(f"sample records not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield loads_strict(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON record on line {line_number} of {path}: {exc}") from exc


def require_schema(manifest: dict, expected) -> None:
    actual = manifest.get("schema_version")
    allowed = {expected} if isinstance(expected, str) else set(expected)
    if actual not in allowed:
        raise ValueError(f"schema version mismatch: run has {actual!r}, expected one of {sorted(allowed)!r}")


def assert_config_matches(manifest: dict, config: dict) -> None:
    """Cache-reuse guard: full config and semantic versions must match."""
    from .schema import config_hash

    if manifest.get("config") != config:
        raise ValueError("run manifest config does not match the requested config")
    if manifest.get("config_hash") != config_hash(config):
        raise ValueError("run manifest config hash is inconsistent with its config")
