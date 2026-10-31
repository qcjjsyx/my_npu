"""MX matrix run IO: file names and streaming iterators.

The dot-product reader/writer (``reader.py``/``writer.py``) stays untouched;
MX runs use their own file names and schema so that neither result type can be
misread as the other.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

from .reader import run_path
from .schema import loads_strict

MX_INPUTS_NAME = "inputs.jsonl.gz"
MX_SAMPLES_NAME = "samples.jsonl.gz"


def iter_mx_inputs(run_dir: str | Path) -> Iterator[dict]:
    path = run_path(run_dir) / MX_INPUTS_NAME
    if not path.exists():
        raise FileNotFoundError(f"MX input records not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield loads_strict(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON record on line {line_number} of {path}: {exc}") from exc


def iter_mx_samples(run_dir: str | Path) -> Iterator[dict]:
    path = run_path(run_dir) / MX_SAMPLES_NAME
    if not path.exists():
        raise FileNotFoundError(f"MX sample records not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield loads_strict(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON record on line {line_number} of {path}: {exc}") from exc
