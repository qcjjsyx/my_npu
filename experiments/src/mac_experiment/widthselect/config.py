"""Configuration loading and validation for the W_align / W_acc experiment.

Strict: unknown keys, legacy keys (``source_run``, ``B``/``B_list``/``W``,
``samples_per_seed``) and custom distribution parameters are rejected
explicitly.  ``W_align_list`` and ``W_acc_list`` define the (one-dimensional)
scans; a large two-dimensional sweep is never a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..version import WIDTHSELECT_ALIGN_MODE, WIDTHSELECT_ACC_MODE, WIDTHSELECT_JOINT_MODE
from .acc_format import validate_w_acc
from .distributions import WIDTH_DISTRIBUTIONS, validate_distribution
from .reference import validate_w_align
from .tensors import TARGET_FORMATS

WIDTHSELECT_MODES = (WIDTHSELECT_ALIGN_MODE, WIDTHSELECT_ACC_MODE, WIDTHSELECT_JOINT_MODE)
MAC_PARALLELISM = 16
GROUP_LENGTH = 32
PASS_CRITERIA_KEYS = ("min_bit_match_rate", "max_ulp_p99", "max_ulp_max", "nonfinite_policy")
NONFINITE_POLICIES = ("reject", "ignore")
ALLOWED_KEYS = (
    "mode",
    "formats",
    "distribution_pairs",
    "seeds",
    "pairs_per_seed",
    "M",
    "N",
    "K_list",
    "W_align_list",
    "W_acc_list",
    "joint_pairs",
    "pass_criteria",
    "output_root",
    "run_name",
)


@dataclass
class WidthSelectConfig:
    mode: str
    formats: list[str]
    distribution_pairs: list[dict]
    seeds: list[int]
    pairs_per_seed: int
    M: int
    N: int
    K_list: list[int]
    W_align_list: list[int] = field(default_factory=list)
    W_acc_list: list[int] = field(default_factory=list)
    joint_pairs: list[dict] = field(default_factory=list)
    pass_criteria: dict | None = None
    output_root: str = "results/numerical/widthselect/runs"
    run_name: str | None = None

    @property
    def x_name(self) -> str:
        return "K"

    @property
    def B_list(self) -> list[int]:
        return [k // MAC_PARALLELISM for k in self.K_list]

    def n_pairs(self) -> int:
        return len(self.distribution_pairs) * len(self.seeds) * self.pairs_per_seed

    def effective(self) -> dict:
        return {
            "mode": self.mode,
            "formats": list(self.formats),
            "distribution_pairs": [dict(pair) for pair in self.distribution_pairs],
            "seeds": list(self.seeds),
            "pairs_per_seed": self.pairs_per_seed,
            "M": self.M,
            "N": self.N,
            "K_list": list(self.K_list),
            "B_rule": "B = K // 16",
            "W_align_list": list(self.W_align_list),
            "W_acc_list": list(self.W_acc_list),
            "joint_pairs": [dict(pair) for pair in self.joint_pairs],
            "pass_criteria": self.pass_criteria,
        }

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "formats": list(self.formats),
            "distribution_pairs": [dict(pair) for pair in self.distribution_pairs],
            "seeds": list(self.seeds),
            "pairs_per_seed": self.pairs_per_seed,
            "M": self.M,
            "N": self.N,
            "K_list": list(self.K_list),
            "W_align_list": list(self.W_align_list),
            "W_acc_list": list(self.W_acc_list),
            "joint_pairs": [dict(pair) for pair in self.joint_pairs],
            "pass_criteria": self.pass_criteria,
            "output_root": self.output_root,
            "run_name": self.run_name,
        }


def default_distribution_pairs() -> list[dict]:
    return [
        {"activation_distribution": name, "weight_distribution": name}
        for name in WIDTH_DISTRIBUTIONS
    ]


def _parse_distribution_pairs(raw) -> list[dict]:
    if raw is None:
        return default_distribution_pairs()
    if not isinstance(raw, list) or not raw:
        raise ValueError("distribution_pairs must be a non-empty list of mappings")
    pairs: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"each distribution_pairs entry must be a mapping, got {entry!r}")
        custom = sorted(key for key in ("activation_parameters", "weight_parameters") if key in entry)
        if custom:
            raise ValueError(
                "custom distribution parameters are not supported: the six input distributions "
                "are fixed unit-RMS experiment definitions; remove activation_parameters / "
                "weight_parameters from distribution_pairs"
            )
        unknown = sorted(set(entry) - {"activation_distribution", "weight_distribution"})
        if unknown:
            raise ValueError(
                f"unknown distribution_pairs keys {unknown}; allowed: "
                "['activation_distribution', 'weight_distribution']"
            )
        for key in ("activation_distribution", "weight_distribution"):
            if key not in entry:
                raise ValueError(f"distribution_pairs entry missing {key!r}: {entry!r}")
            validate_distribution(entry[key])
        pairs.append(
            {
                "activation_distribution": entry["activation_distribution"],
                "weight_distribution": entry["weight_distribution"],
            }
        )
    return pairs


def _parse_int_list(raw, label: str) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must be a non-empty list of integers")
    values = sorted({int(value) for value in raw})
    return values


def _parse_pass_criteria(raw) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("pass_criteria must be a mapping")
    unknown = sorted(set(raw) - set(PASS_CRITERIA_KEYS))
    if unknown:
        raise ValueError(f"unknown pass_criteria keys {unknown}; allowed: {list(PASS_CRITERIA_KEYS)}")
    criteria: dict = {}
    if "min_bit_match_rate" in raw:
        rate = float(raw["min_bit_match_rate"])
        if not 0.0 <= rate <= 1.0:
            raise ValueError("pass_criteria.min_bit_match_rate must be in [0, 1]")
        criteria["min_bit_match_rate"] = rate
    for key in ("max_ulp_p99", "max_ulp_max"):
        if key in raw:
            value = int(raw[key])
            if value < 0:
                raise ValueError(f"pass_criteria.{key} must be >= 0")
            criteria[key] = value
    if "nonfinite_policy" in raw:
        policy = raw["nonfinite_policy"]
        if policy not in NONFINITE_POLICIES:
            raise ValueError(f"pass_criteria.nonfinite_policy must be one of {list(NONFINITE_POLICIES)}")
        criteria["nonfinite_policy"] = policy
    return criteria or None


def load_widthselect_config(path: str | Path) -> WidthSelectConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config file must contain a YAML mapping")
    unknown = sorted(set(data) - set(ALLOWED_KEYS))
    if unknown:
        raise ValueError(
            f"unknown configuration keys {unknown}; this experiment has no source_run and no "
            f"historical input reuse; allowed keys: {list(ALLOWED_KEYS)}"
        )
    mode = data.get("mode")
    if mode not in WIDTHSELECT_MODES:
        raise ValueError(f"mode must be one of {WIDTHSELECT_MODES}, got {mode!r}")

    formats = list(data.get("formats") or list(TARGET_FORMATS))
    unknown_formats = sorted(set(formats) - set(TARGET_FORMATS))
    if unknown_formats:
        raise ValueError(f"unknown formats {unknown_formats}; known: {list(TARGET_FORMATS)}")
    if not formats:
        raise ValueError("at least one target format is required")

    M = data.get("M")
    N = data.get("N")
    if M is None or N is None:
        raise ValueError("M and N are required")
    M, N = int(M), int(N)
    if M < 1 or N < 1:
        raise ValueError("M and N must be >= 1")

    k_raw = data.get("K_list")
    if k_raw is None:
        raise ValueError("K_list is required")
    K_list = _parse_int_list(k_raw, "K_list")
    for k in K_list:
        if k < MAC_PARALLELISM or k % MAC_PARALLELISM:
            raise ValueError(f"each K must be a positive multiple of {MAC_PARALLELISM}: {k}")
        if "mx_fp8" in formats and k % GROUP_LENGTH:
            raise ValueError(f"each K must be a multiple of the group length {GROUP_LENGTH} for mx_fp8: {k}")

    distribution_pairs = _parse_distribution_pairs(data.get("distribution_pairs"))
    seeds = [int(seed) for seed in (data.get("seeds") or [1])]
    if not seeds:
        raise ValueError("at least one seed is required")
    pairs_per_seed = int(data.get("pairs_per_seed", 1))
    if pairs_per_seed < 1:
        raise ValueError("pairs_per_seed must be >= 1")

    w_align_list: list[int] = []
    w_acc_list: list[int] = []
    joint_pairs: list[dict] = []
    if mode == WIDTHSELECT_ALIGN_MODE:
        if data.get("W_acc_list") is not None or data.get("joint_pairs") is not None:
            raise ValueError("widthselect_align accepts only W_align_list")
        if data.get("W_align_list") is None:
            raise ValueError("widthselect_align requires W_align_list")
        w_align_list = _parse_int_list(data["W_align_list"], "W_align_list")
        for value in w_align_list:
            validate_w_align(value)
    elif mode == WIDTHSELECT_ACC_MODE:
        if data.get("W_align_list") is not None or data.get("joint_pairs") is not None:
            raise ValueError("widthselect_acc accepts only W_acc_list")
        if data.get("W_acc_list") is None:
            raise ValueError("widthselect_acc requires W_acc_list")
        w_acc_list = _parse_int_list(data["W_acc_list"], "W_acc_list")
        for value in w_acc_list:
            validate_w_acc(value)
    else:
        if data.get("W_align_list") is not None or data.get("W_acc_list") is not None:
            raise ValueError("widthselect_joint takes its widths from joint_pairs, not scan lists")
        raw = data.get("joint_pairs")
        if not isinstance(raw, list) or not raw:
            raise ValueError("widthselect_joint requires a non-empty joint_pairs list")
        seen: set[tuple[int, int]] = set()
        for entry in raw:
            if not isinstance(entry, dict) or set(entry) != {"w_align", "w_acc"}:
                raise ValueError(f"each joint_pairs entry must be exactly {{w_align, w_acc}}: {entry!r}")
            w_align, w_acc = int(entry["w_align"]), int(entry["w_acc"])
            validate_w_align(w_align)
            validate_w_acc(w_acc)
            if (w_align, w_acc) in seen:
                raise ValueError(f"duplicate joint_pairs entry: {(w_align, w_acc)}")
            seen.add((w_align, w_acc))
            joint_pairs.append({"w_align": w_align, "w_acc": w_acc})

    pass_criteria = _parse_pass_criteria(data.get("pass_criteria"))
    output_root = str(data.get("output_root", "results/numerical/widthselect/runs"))
    run_name = data.get("run_name")
    run_name = None if run_name in (None, "") else str(run_name)

    return WidthSelectConfig(
        mode=mode,
        formats=formats,
        distribution_pairs=distribution_pairs,
        seeds=seeds,
        pairs_per_seed=pairs_per_seed,
        M=M,
        N=N,
        K_list=K_list,
        W_align_list=w_align_list,
        W_acc_list=w_acc_list,
        joint_pairs=joint_pairs,
        pass_criteria=pass_criteria,
        output_root=output_root,
        run_name=run_name,
    )
