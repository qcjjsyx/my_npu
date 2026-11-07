"""Strict JSON helpers and schema constants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..version import MX_SCHEMA_VERSION, MX_STREAM_VERSION, SCHEMA_VERSION, STREAM_VERSION

SOURCE_ROOT = Path(__file__).resolve().parent.parent


def source_digest() -> str:
    """SHA-256 over every package ``*.py`` file (shared by run manifests).

    Kept in this lightweight module so offline analysis can fingerprint the
    analyzer source without importing execution/sampling engines.
    """
    digest = hashlib.sha256()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        digest.update(str(path.relative_to(SOURCE_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


SUMMARY_VERSION = "mac-summary-v3"

BITMATCH_SUMMARY_VERSION = "mac-bitmatch-v1"

MX_SUMMARY_VERSION = "mac-mx-summary-v3"

MX_ANALYSIS_VERSION = "mac-mx-analysis-v3"

MX_ALIGN_SUMMARY_VERSION = "mac-mx-align-summary-v1"

MX_ALIGN_ANALYSIS_VERSION = "mac-mx-align-analysis-v1"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def dumps_strict(obj, indent: int | None = 2) -> str:
    return json.dumps(obj, sort_keys=True, indent=indent, allow_nan=False)


def loads_strict(text: str):
    def _reject(token: str):
        raise ValueError(f"non-finite JSON token {token!r} is not allowed")

    return json.loads(text, parse_constant=_reject)


def config_hash(config: dict) -> str:
    payload = canonical_json({"schema": SCHEMA_VERSION, "stream": STREAM_VERSION, "config": config})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mx_align_config_hash(config: dict) -> str:
    """Config hash for alignment runs; never collides with other MX configs."""
    from ..version import MX_ALIGN_SCHEMA_VERSION, MX_ALIGN_DEFAULT

    payload = canonical_json(
        {"schema": MX_ALIGN_SCHEMA_VERSION, "w_align_default": MX_ALIGN_DEFAULT, "config": config}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mx_config_hash(config: dict) -> str:
    """Config hash for MX matrix runs; never collides with dot-product configs."""
    payload = canonical_json({"schema": MX_SCHEMA_VERSION, "stream": MX_STREAM_VERSION, "config": config})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def widthselect_config_hash(config: dict) -> str:
    """Config hash for W_align/W_acc runs; never collides with other schemas."""
    from ..version import WIDTHSELECT_SCHEMA_VERSION

    payload = canonical_json({"schema": WIDTHSELECT_SCHEMA_VERSION, "config": config})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
