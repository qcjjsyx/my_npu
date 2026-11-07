"""Event keys shared by execution and offline metrics.

Kept in a dependency-free module so that offline analysis can consume saved
records without importing the execution engine or the sampler.
"""

from __future__ import annotations

EVENT_KEYS = (
    "window_overflow",
    "non_finite_fp32",
    "unexpected_zero",
    "alignment_zeroed",
    "update_absorbed",
    "all_zero_block",
)


def zero_events() -> dict:
    return {key: 0 for key in EVENT_KEYS}
