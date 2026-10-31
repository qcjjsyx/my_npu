"""Run configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..formats import FORMATS
from ..matrix.inputs import INPUT_DISTRIBUTIONS
from ..sampling.sampler import CANCELLATION_SIGN_MODES, DISTRIBUTIONS
from ..version import (
    MX_ALIGN_DEFAULT,
    MX_ALIGN_POLICY,
    MX_ALIGN_SCHEMA_VERSION,
    MX_INPUT_QUANTIZATION_POLICY,
    MX_INPUT_SAMPLING_VERSION,
    MX_INPUT_STREAM_VERSION,
    SCHEMA_VERSION,
    STREAM_VERSION,
)

WIDTH_SCAN = list(range(18, 42))
DEFAULT_B_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MODES = ("width_sweep", "block_sweep", "mx_matrix_sweep", "mx_alignment_sweep")
MX_MODE = "mx_matrix_sweep"
MX_ALIGN_MODE = "mx_alignment_sweep"
MX_GROUP_LENGTH = 32
MX_MAC_PARALLELISM = 16
MX_DISTRIBUTION_PAIR_KEYS = ("activation_distribution", "weight_distribution")


@dataclass
class RunConfig:
    mode: str
    formats: list[str]
    distributions: list[str]
    seeds: list[int]
    samples_per_seed: int
    cancellation_sign_mode: str = "greedy"
    output_root: str = "results/numerical/dot_product/runs"
    run_name: str | None = None
    record_block_rho: bool = False
    B: int | None = None
    W: int | None = None
    B_list: list[int] = field(default_factory=list)
    M: int | None = None
    N: int | None = None
    K_list: list[int] = field(default_factory=list)
    distribution_pairs: list[dict] = field(default_factory=list)
    pairs_per_seed: int = 1
    W_align: int | None = None
    source_run: str | None = None

    @property
    def is_mx_matrix(self) -> bool:
        return self.mode == MX_MODE

    @property
    def is_mx_align(self) -> bool:
        return self.mode == MX_ALIGN_MODE

    @property
    def is_mx(self) -> bool:
        return self.is_mx_matrix or self.is_mx_align

    @property
    def x_name(self) -> str:
        if self.is_mx:
            return "K"
        return "W" if self.mode == "width_sweep" else "B"

    @property
    def x_values(self) -> list[int]:
        if self.is_mx:
            return list(self.K_list)
        return WIDTH_SCAN if self.mode == "width_sweep" else list(self.B_list)

    @property
    def windows(self) -> list[int]:
        if self.is_mx:
            return []
        return WIDTH_SCAN if self.mode == "width_sweep" else [self.W]

    @property
    def snapshots(self) -> list[int]:
        if self.is_mx:
            return list(self.K_list)
        return [self.B] if self.mode == "width_sweep" else list(self.B_list)

    @property
    def fixed(self) -> dict:
        if self.is_mx:
            payload = {
                "M": self.M,
                "N": self.N,
                "group_length": MX_GROUP_LENGTH,
                "mac_parallelism": MX_MAC_PARALLELISM,
            }
            if self.is_mx_align:
                payload["w_align"] = self.W_align
            return payload
        return {"B": self.B} if self.mode == "width_sweep" else {"W": self.W}

    def effective(self) -> dict:
        payload = {
            "mode": self.mode,
            "formats": list(self.formats),
            "distributions": list(self.distributions),
            "seeds": list(self.seeds),
            "samples_per_seed": self.samples_per_seed,
            "cancellation_sign_mode": self.cancellation_sign_mode,
            "record_block_rho": self.record_block_rho,
            "B": self.B,
            "W": self.W,
            "B_list": list(self.B_list),
            "width_scan": WIDTH_SCAN if self.mode == "width_sweep" else None,
            "schema_version": SCHEMA_VERSION,
            "stream_version": STREAM_VERSION,
        }
        if self.is_mx:
            payload.update(
                {
                    "M": self.M,
                    "N": self.N,
                    "K_list": list(self.K_list),
                    "distribution_pairs": [dict(pair) for pair in self.distribution_pairs],
                    "pairs_per_seed": self.pairs_per_seed,
                    "group_length": MX_GROUP_LENGTH,
                    "mac_parallelism": MX_MAC_PARALLELISM,
                    "input_sampling_version": MX_INPUT_SAMPLING_VERSION,
                    "input_stream_version": MX_INPUT_STREAM_VERSION,
                    "input_quantization_policy": MX_INPUT_QUANTIZATION_POLICY,
                }
            )
            if self.is_mx_align:
                payload.update(
                    {
                        "schema_version": MX_ALIGN_SCHEMA_VERSION,
                        "w_align": self.W_align,
                        "align_policy": MX_ALIGN_POLICY,
                        "source_run": self.source_run,
                    }
                )
        return payload

    def describe(self) -> dict:
        payload = {
            "mode": self.mode,
            "formats": list(self.formats),
            "distributions": list(self.distributions),
            "seeds": list(self.seeds),
            "samples_per_seed": self.samples_per_seed,
            "cancellation_sign_mode": self.cancellation_sign_mode,
            "output_root": self.output_root,
            "run_name": self.run_name,
            "record_block_rho": self.record_block_rho,
            "B": self.B,
            "W": self.W,
            "B_list": list(self.B_list),
        }
        if self.is_mx:
            payload.update(
                {
                    "M": self.M,
                    "N": self.N,
                    "K_list": list(self.K_list),
                    "distribution_pairs": [dict(pair) for pair in self.distribution_pairs],
                    "pairs_per_seed": self.pairs_per_seed,
                }
            )
            if self.is_mx_align:
                payload.update({"W_align": self.W_align, "source_run": self.source_run})
        return payload


def load_config(path: str | Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config file must contain a YAML mapping")
    mode = data.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == MX_MODE:
        return _load_mx_config(data)
    if mode == MX_ALIGN_MODE:
        return _load_mx_alignment_config(data)

    formats = list(data.get("formats") or [])
    if not formats:
        raise ValueError("at least one format is required")
    unknown_formats = [name for name in formats if name not in FORMATS]
    if unknown_formats:
        raise ValueError(f"unknown formats: {unknown_formats}")
    distributions = list(data.get("distributions") or list(DISTRIBUTIONS))
    unknown_dists = [name for name in distributions if name not in DISTRIBUTIONS]
    if unknown_dists:
        raise ValueError(f"unknown distributions: {unknown_dists}")
    seeds = [int(seed) for seed in (data.get("seeds") or [1])]
    if not seeds:
        raise ValueError("at least one seed is required")
    samples_per_seed = int(data.get("samples_per_seed", 1))
    if samples_per_seed < 1:
        raise ValueError("samples_per_seed must be >= 1")
    sign_mode = data.get("cancellation_sign_mode", "greedy")
    if sign_mode not in CANCELLATION_SIGN_MODES:
        raise ValueError(f"cancellation_sign_mode must be one of {CANCELLATION_SIGN_MODES}")
    output_root = str(data.get("output_root", "results/numerical/dot_product/runs"))
    run_name = data.get("run_name")
    run_name = None if run_name in (None, "") else str(run_name)
    record_block_rho = bool(data.get("record_block_rho", False))

    b_value = data.get("B")
    w_value = data.get("W")
    b_list_raw = data.get("B_list")

    if mode == "width_sweep":
        if w_value is not None:
            raise ValueError("width_sweep scans W=18..41; 'W' is not a valid key (use 'B' to fix the update count)")
        if b_value is None:
            raise ValueError("width_sweep requires an explicit fixed 'B'")
        b_value = int(b_value)
        if b_value < 1:
            raise ValueError("B must be >= 1")
        if b_list_raw is not None:
            raise ValueError("width_sweep does not use 'B_list'")
        b_list = []
        w_final = None
    else:
        if w_value is None:
            raise ValueError("block_sweep requires an explicit 'W' (there is no default and no optimal W)")
        w_final = int(w_value)
        if w_final < 0:
            raise ValueError("W must be >= 0")
        if b_value is not None:
            raise ValueError("block_sweep does not use 'B'; use 'B_list'")
        b_list = [int(x) for x in (b_list_raw if b_list_raw is not None else DEFAULT_B_LIST)]
        if not b_list or any(x < 1 for x in b_list):
            raise ValueError("B_list must contain integers >= 1")
        b_list = sorted(set(b_list))
        b_value = None

    return RunConfig(
        mode=mode,
        formats=formats,
        distributions=distributions,
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        cancellation_sign_mode=sign_mode,
        output_root=output_root,
        run_name=run_name,
        record_block_rho=record_block_rho,
        B=b_value,
        W=w_final,
        B_list=b_list,
    )


def _parse_distribution_pairs(pair_raw) -> list[dict]:
    if pair_raw is None:
        raise ValueError(
            "mx_matrix_sweep/mx_alignment_sweep requires 'distribution_pairs' "
            "(activation_distribution / weight_distribution); "
            f"known distributions: {list(INPUT_DISTRIBUTIONS)}"
        )
    if not isinstance(pair_raw, list) or not pair_raw:
        raise ValueError("distribution_pairs must be a non-empty list of mappings")
    distribution_pairs: list[dict] = []
    for entry in pair_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"each distribution_pairs entry must be a mapping, got {entry!r}")
        custom = sorted(key for key in ("activation_parameters", "weight_parameters") if key in entry)
        if custom:
            raise ValueError(
                "custom distribution parameters are not supported: the four input distributions "
                "are fixed unit-RMS experiment definitions; remove "
                f"{custom} from distribution_pairs"
            )
        unknown_keys = sorted(set(entry) - set(MX_DISTRIBUTION_PAIR_KEYS))
        if unknown_keys:
            raise ValueError(
                f"unknown distribution_pairs keys {unknown_keys}; allowed: "
                f"{list(MX_DISTRIBUTION_PAIR_KEYS)}"
            )
        for key in MX_DISTRIBUTION_PAIR_KEYS:
            if key not in entry:
                raise ValueError(f"distribution_pairs entry missing {key!r}: {entry!r}")
            if entry[key] not in INPUT_DISTRIBUTIONS:
                raise ValueError(
                    f"unknown input distribution {entry[key]!r}; known: {list(INPUT_DISTRIBUTIONS)}"
                )
        distribution_pairs.append(
            {
                "activation_distribution": entry["activation_distribution"],
                "weight_distribution": entry["weight_distribution"],
            }
        )
    return distribution_pairs


def _load_mx_config(data: dict) -> RunConfig:
    if data.get("B") is not None or data.get("B_list") is not None or data.get("W") is not None:
        raise ValueError("mx_matrix_sweep scans K; 'B', 'W' and 'B_list' are not valid keys")
    if data.get("formats") not in (None, [], "mx_fp8_e4m3"):
        raise ValueError("mx_matrix_sweep fixes the MX element format to fp8_e4m3; 'formats' is not configurable")
    legacy_keys = sorted(
        key
        for key in ("scale_scenarios", "distributions", "cancellation_sign_mode", "record_block_rho")
        if key in data
    )
    if legacy_keys:
        raise ValueError(
            "the v1 MX input scheme (independent E4M3 element + E8M0 scale sampling, "
            f"scale_scenarios) is retired; remove {legacy_keys} and configure "
            "'distribution_pairs' with activation_distribution / weight_distribution"
        )
    M = data.get("M")
    N = data.get("N")
    if M is None or N is None:
        raise ValueError("mx_matrix_sweep requires explicit M and N")
    M = int(M)
    N = int(N)
    if M < 1:
        raise ValueError("M must be >= 1")
    if N < MX_GROUP_LENGTH or N % MX_GROUP_LENGTH:
        raise ValueError(f"N must be a positive multiple of the group length {MX_GROUP_LENGTH}")
    k_list_raw = data.get("K_list")
    if k_list_raw is None:
        raise ValueError("mx_matrix_sweep requires an explicit K_list")
    K_list = sorted({int(k) for k in k_list_raw})
    if not K_list:
        raise ValueError("K_list must contain at least one K")
    for k in K_list:
        if k < MX_GROUP_LENGTH or k % MX_GROUP_LENGTH:
            raise ValueError(f"each K must be a positive multiple of the group length {MX_GROUP_LENGTH}: {k}")
    distribution_pairs = _parse_distribution_pairs(data.get("distribution_pairs"))
    seeds = [int(seed) for seed in (data.get("seeds") or [1])]
    if not seeds:
        raise ValueError("at least one seed is required")
    pairs_per_seed = int(data.get("pairs_per_seed", data.get("samples_per_seed", 1)))
    if pairs_per_seed < 1:
        raise ValueError("pairs_per_seed must be >= 1")
    output_root = str(data.get("output_root", "results/numerical/mxfp8/runs"))
    run_name = data.get("run_name")
    run_name = None if run_name in (None, "") else str(run_name)
    return RunConfig(
        mode=MX_MODE,
        formats=[],
        distributions=[],
        seeds=seeds,
        samples_per_seed=pairs_per_seed,
        output_root=output_root,
        run_name=run_name,
        M=M,
        N=N,
        K_list=K_list,
        distribution_pairs=distribution_pairs,
        pairs_per_seed=pairs_per_seed,
    )


def _load_mx_alignment_config(data: dict) -> RunConfig:
    mode_name = "mx_alignment_sweep"
    if data.get("B") is not None or data.get("B_list") is not None or data.get("W") is not None:
        raise ValueError(f"{mode_name} scans K; 'B', 'W' and 'B_list' are not valid keys")
    if data.get("formats") not in (None, [], "mx_fp8_e4m3"):
        raise ValueError(f"{mode_name} fixes the MX element format to fp8_e4m3; 'formats' is not configurable")
    legacy_keys = sorted(
        key
        for key in ("scale_scenarios", "distributions", "cancellation_sign_mode", "record_block_rho")
        if key in data
    )
    if legacy_keys:
        raise ValueError(
            "the v1 MX input scheme is retired; remove "
            f"{legacy_keys} and configure 'distribution_pairs' with activation_distribution / "
            "weight_distribution"
        )
    M = data.get("M")
    N = data.get("N")
    if M is None or N is None:
        raise ValueError(f"{mode_name} requires explicit M and N")
    M = int(M)
    N = int(N)
    if M < 1:
        raise ValueError("M must be >= 1")
    if N < MX_GROUP_LENGTH or N % MX_GROUP_LENGTH:
        raise ValueError(f"N must be a positive multiple of the group length {MX_GROUP_LENGTH}")
    k_list_raw = data.get("K_list")
    if k_list_raw is None:
        raise ValueError(f"{mode_name} requires an explicit K_list")
    K_list = sorted({int(k) for k in k_list_raw})
    if not K_list:
        raise ValueError("K_list must contain at least one K")
    for k in K_list:
        if k < MX_GROUP_LENGTH or k % MX_GROUP_LENGTH:
            raise ValueError(
                f"each K must be a positive multiple of the group length {MX_GROUP_LENGTH}: {k}"
            )
    w_align_raw = data.get("W_align", MX_ALIGN_DEFAULT)
    if w_align_raw is None:
        w_align = MX_ALIGN_DEFAULT
    else:
        w_align = int(w_align_raw)
    if not 1 <= w_align <= 60:
        raise ValueError(f"W_align must be an integer in [1, 60], got {w_align}")
    distribution_pairs = _parse_distribution_pairs(data.get("distribution_pairs"))
    seeds = [int(seed) for seed in (data.get("seeds") or [1])]
    if not seeds:
        raise ValueError("at least one seed is required")
    pairs_per_seed = int(data.get("pairs_per_seed", data.get("samples_per_seed", 1)))
    if pairs_per_seed < 1:
        raise ValueError("pairs_per_seed must be >= 1")
    source_run = data.get("source_run")
    source_run = None if source_run in (None, "") else str(source_run)
    output_root = str(data.get("output_root", "results/numerical/mx_alignment/runs"))
    run_name = data.get("run_name")
    run_name = None if run_name in (None, "") else str(run_name)
    return RunConfig(
        mode=MX_ALIGN_MODE,
        formats=[],
        distributions=[],
        seeds=seeds,
        samples_per_seed=pairs_per_seed,
        output_root=output_root,
        run_name=run_name,
        M=M,
        N=N,
        K_list=K_list,
        distribution_pairs=distribution_pairs,
        pairs_per_seed=pairs_per_seed,
        W_align=w_align,
        source_run=source_run,
    )
