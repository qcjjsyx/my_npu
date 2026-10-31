"""Width-select run IO: distinct file names and streaming iterators.

The file names differ from both the dot-product and the MX run conventions so
that a width-select run directory can never be misread by historical readers
(and vice versa).
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Iterator

from ..results.reader import run_path
from ..results.schema import loads_strict

WIDTH_INPUTS_NAME = "width_inputs.jsonl.gz"
WIDTH_SAMPLES_NAME = "width_samples.jsonl.gz"


def iter_width_inputs(run_dir: str | Path) -> Iterator[dict]:
    return _iter_records(run_path(run_dir) / WIDTH_INPUTS_NAME)


def iter_width_samples(run_dir: str | Path) -> Iterator[dict]:
    return _iter_records(run_path(run_dir) / WIDTH_SAMPLES_NAME)


def _iter_records(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(f"width-select records not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield loads_strict(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON record on line {line_number} of {path}: {exc}") from exc


def file_identity(path: str | Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return {"sha256": "sha256:" + digest.hexdigest(), "size": Path(path).stat().st_size}
