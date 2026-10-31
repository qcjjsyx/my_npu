"""Run directory writer.

The writer refuses to reuse an existing run directory and never writes bare
NaN/Infinity tokens: undefined values are serialised as ``null`` plus a
status.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from .schema import dumps_strict

MANIFEST_NAME = "manifest.json"
SAMPLES_NAME = "samples.jsonl.gz"
SUMMARY_NAME = "summary.json"
SUMMARY_CSV_NAME = "summary.csv"
REANALYSIS_NAME = "summary_recomputed.json"


def create_run_dir(output_root: str | Path, run_id: str) -> Path:
    root = Path(output_root)
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "plots").mkdir()
    return run_dir


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(dumps_strict(obj) + "\n", encoding="utf-8")


def read_json(path: str | Path):
    from .schema import loads_strict

    return loads_strict(Path(path).read_text(encoding="utf-8"))


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "JsonlSink":
        self._handle = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=6)
        return self

    def write(self, record: dict) -> None:
        if self._handle is None:
            raise RuntimeError("sink not open")
        self._handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


CSV_FIELDS = [
    "format",
    "distribution",
    "x_name",
    "x",
    "comparison",
    "scope",
    "metric",
    "value",
    "status",
    "reference_rms",
    "n_total",
    "n_valid",
    "n_invalid",
    "invalid_rate",
    "note",
]


def _coverage_columns(coverage: dict) -> dict:
    return {
        "n_total": coverage.get("n_total", ""),
        "n_valid": coverage.get("n_valid", ""),
        "n_invalid": coverage.get("n_invalid", ""),
        "invalid_rate": "" if coverage.get("invalid_rate") is None else repr(coverage.get("invalid_rate")),
    }


def summary_to_csv_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for cell in summary["cells"]:
        for comparison, payload in cell["comparisons"].items():
            coverage = payload.get("coverage") or {}

            def add_scope(scope: str, metrics: dict, reference_rms: dict | None, cov: dict, note: str = "") -> None:
                ref_value = ""
                if reference_rms is not None and reference_rms.get("value") is not None:
                    ref_value = repr(reference_rms["value"])
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "format": cell["format"],
                            "distribution": cell["distribution"],
                            "x_name": summary["x_name"],
                            "x": cell["x"],
                            "comparison": comparison,
                            "scope": scope,
                            "metric": metric,
                            "value": "" if value["value"] is None else repr(value["value"]),
                            "status": value["status"],
                            "reference_rms": ref_value,
                            **_coverage_columns(cov),
                            "note": note,
                        }
                    )

            add_scope("whole_batch", payload["metrics"], payload.get("reference_rms"), coverage)
            subset = payload.get("finite_subset")
            if subset:
                add_scope(
                    "finite_subset",
                    subset["metrics"],
                    subset.get("reference_rms"),
                    subset.get("coverage") or coverage,
                    subset.get("note", ""),
                )
    return rows


def write_summary_csv(path: str | Path, summary: dict) -> None:
    import csv

    rows = summary_to_csv_rows(summary)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
