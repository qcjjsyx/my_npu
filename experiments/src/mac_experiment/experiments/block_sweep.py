"""B sweep driver: fixed explicit W, paired-prefix snapshots over B."""

from __future__ import annotations

from pathlib import Path

from .common import execute
from .config import RunConfig, load_config


def run(config: RunConfig, verbose: bool = True) -> Path:
    if config.mode != "block_sweep":
        raise ValueError("block_sweep driver requires mode=block_sweep")
    if config.W is None:
        raise ValueError("block_sweep requires an explicit W")
    return execute(config, verbose=verbose)


def run_from_file(path: str | Path, verbose: bool = True) -> Path:
    return run(load_config(path), verbose=verbose)
