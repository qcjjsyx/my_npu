"""Deterministic width-select input streams (new namespace).

Key: ``mac-width-fp32-stream-v1|seed|matrix|distribution_label|pair_index|row|col|purpose``.

Design points required by the experiment:

* the master seed, the matrix identity ("x"/"w"), the distribution label, the
  matrix-pair index and the full logical coordinate (row, column) all select the
  stream, so every element is reproducible independently of traversal order;
* extending ``Kmax`` or the number of pairs cannot change an existing element
  (paired-prefix stability by construction, not by sequential drawing);
* the namespace differs from the historical ``mac-mx-fp32-stream-v1`` key, so
  the regenerated inputs never coincide with historical run data;
* Python's process-randomised ``hash()`` is never used.
"""

from __future__ import annotations

import hashlib
import random

from ..version import WIDTHSELECT_INPUT_STREAM_VERSION

WIDTH_INPUT_PURPOSES = ("value",)


def width_input_stream_rng(
    master_seed: int,
    matrix: str,
    distribution_label: str,
    pair_index: int,
    row: int,
    col: int,
    purpose: str = "value",
) -> random.Random:
    if purpose not in WIDTH_INPUT_PURPOSES:
        raise ValueError(f"unknown width-select stream purpose {purpose!r}")
    if matrix not in ("x", "w"):
        raise ValueError(f"matrix must be 'x' or 'w', got {matrix!r}")
    key = (
        f"mac-width-fp32-stream-v{WIDTHSELECT_INPUT_STREAM_VERSION}"
        f"|{master_seed}|{matrix}|{distribution_label}|{pair_index}|{row}|{col}|{purpose}"
    )
    digest = hashlib.sha256(key.encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))
