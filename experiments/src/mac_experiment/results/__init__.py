"""Result schema, writing, reading and re-analysis."""

from .bitmatch_analysis import analyze_bitmatch
from .enhance import enhance_run
from .reader import (
    assert_config_matches,
    iter_records,
    load_manifest,
    load_summary,
    require_schema,
    run_path,
)
from .reanalyze import rebuild_summary, reanalyze
from .schema import SCHEMA_VERSION, SUMMARY_VERSION, canonical_json, config_hash, dumps_strict, loads_strict
from .writer import (
    JsonlSink,
    create_run_dir,
    read_json,
    summary_to_csv_rows,
    write_json,
    write_summary_csv,
)

__all__ = [
    "analyze_bitmatch",
    "assert_config_matches",
    "enhance_run",
    "iter_records",
    "load_manifest",
    "load_summary",
    "require_schema",
    "run_path",
    "rebuild_summary",
    "reanalyze",
    "SCHEMA_VERSION",
    "SUMMARY_VERSION",
    "canonical_json",
    "config_hash",
    "dumps_strict",
    "loads_strict",
    "JsonlSink",
    "create_run_dir",
    "read_json",
    "summary_to_csv_rows",
    "write_json",
    "write_summary_csv",
]
