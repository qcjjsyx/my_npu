"""W sweep driver: fixed B, full scan W = 18..41."""

from __future__ import annotations

from pathlib import Path

from .common import execute
from .config import RunConfig, load_config


def run(config: RunConfig, verbose: bool = True) -> Path:
    if config.mode != "width_sweep":
        raise ValueError("width_sweep driver requires mode=width_sweep")
    return execute(config, verbose=verbose)


def run_from_file(path: str | Path, verbose: bool = True) -> Path:
    return run(load_config(path), verbose=verbose)
