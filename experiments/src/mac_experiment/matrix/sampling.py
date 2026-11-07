"""MX input sampling: element-code strategy x scale strategy.

Element codes are sampled directly on the legal E4M3 encoding set (never
continuous values quantised afterwards) by reusing the framework magnitude
PMFs.  Scale exponents are sampled independently.  The two strategies are
composable and each records its own version and parameters.

``cancellation`` is intentionally not offered for matrices: the legacy
distribution assigns product signs so that *one* dot product cancels, which
cannot simultaneously hold for every output of a matrix product.  A matrix
cancellation distribution would need its own definition and validation.

Stable streams: element draws are sequential per row over the global column
index and scale draws are keyed by the global group-grid coordinates
(``sampling/streams.py::mx_stream_rng``).  Growing Kmax, switching scale
scenario or adding a distribution therefore never perturbs existing draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..formats.e4m3_element import E4M3Element
from ..formats.e8m0 import exponent_to_code
from ..sampling.pmf import DiscretePMF, build_magnitude_pmf
from ..sampling.sampler import DISTRIBUTIONS, DistributionSpec
from ..sampling.streams import mx_stream_rng
from ..version import MX_STREAM_VERSION

ELEMENT_STRATEGY_VERSION = "mx-element-v2"
SCALE_STRATEGY_VERSION = "mx-scale-v2"

ELEMENT_DISTRIBUTIONS = (
    "positive",
    "uniform",
    "exponent_gaussian",
    "laplace",
    "student_t3",
    "outlier",
)

SCALE_SCENARIOS = ("unit", "narrow", "rare_large")

SCALE_SCENARIO_PARAMETERS = {
    "unit": {"rule": "all exponents 0", "support": [0], "probabilities": [1.0]},
    "narrow": {
        "rule": "exponent uniform on {-2,-1,0,1,2}; activation and weight independent",
        "support": [-2, -1, 0, 1, 2],
        "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
    },
    "rare_large": {
        "rule": "exponent 0 with p=0.99, exponent 8 with p=0.01; activation and weight independent",
        "support": [0, 8],
        "probabilities": [0.99, 0.01],
    },
}


@dataclass
class MxElementSampler:
    distribution: str
    spec: DistributionSpec = field(init=False)
    pmf: DiscretePMF = field(init=False)
    element: E4M3Element = field(init=False, default_factory=E4M3Element)

    def __post_init__(self) -> None:
        if self.distribution not in ELEMENT_DISTRIBUTIONS:
            if self.distribution == "cancellation":
                raise ValueError(
                    "the cancellation distribution is defined for a single dot product and is not "
                    "offered for matrix inputs; provide a separately defined matrix cancellation scheme"
                )
            raise ValueError(
                f"unknown MX element distribution {self.distribution!r}; known: {list(ELEMENT_DISTRIBUTIONS)}"
            )
        self.spec = DISTRIBUTIONS[self.distribution]
        self.pmf = build_magnitude_pmf(
            self.element, self.spec.magnitude_kind, self.spec.shape
        )

    def metadata(self) -> dict:
        return {
            "strategy_version": ELEMENT_STRATEGY_VERSION,
            "distribution": self.distribution,
            "magnitude": self.spec.magnitude_kind,
            "shape": self.spec.shape,
            "sign_mode": self.spec.sign_mode,
            "note": self.spec.note,
            "pmf": self.pmf.metadata,
            "stream_version": MX_STREAM_VERSION,
            "stream_key": (
                "mac-mx-stream-v2|seed|element|<matrix>|<pair_index>|<distribution>|row|0|<amplitude|sign>"
            ),
            "stream_pair_index": "matrix-pair index is part of every element stream key",
            "synthetic_disclaimer": "synthetic shape distributions do not claim to represent real LLM tensors",
        }

    def sample(self, seed: int, matrix: str, rows: int, cols: int, pair_index: int = 0) -> list[int]:
        """Row-major element codes for a ``rows x cols`` matrix of one pair."""
        if rows < 0 or cols < 0:
            raise ValueError("rows/cols must be non-negative")
        if pair_index < 0:
            raise ValueError("pair_index must be >= 0")
        fair = self.spec.sign_mode != "positive"
        codes: list[int] = []
        for row in range(rows):
            rng_mag = mx_stream_rng(
                seed, "element", matrix, self.distribution, pair_index, row, 0, "amplitude"
            )
            rng_sign = (
                mx_stream_rng(seed, "element", matrix, self.distribution, pair_index, row, 0, "sign")
                if fair
                else None
            )
            for _ in range(cols):
                code = self.pmf.sample(rng_mag)
                operand = self.element.decode(code)
                sign = rng_sign.getrandbits(1) if rng_sign is not None else 0
                codes.append((sign << 7) | (operand.raw_exp << 3) | operand.frac)
        return codes


@dataclass
class MxScaleSampler:
    scenario: str

    def __post_init__(self) -> None:
        if self.scenario not in SCALE_SCENARIOS:
            raise ValueError(f"unknown scale scenario {self.scenario!r}; known: {list(SCALE_SCENARIOS)}")

    def metadata(self) -> dict:
        return {
            "strategy_version": SCALE_STRATEGY_VERSION,
            "scenario": self.scenario,
            "parameters": SCALE_SCENARIO_PARAMETERS[self.scenario],
            "stream_version": MX_STREAM_VERSION,
            "stream_key": (
                "mac-mx-stream-v2|seed|scale|<matrix>|<pair_index>|<scenario>|<grid_row>|<grid_col>|scale"
            ),
            "stream_pair_index": "matrix-pair index is part of every scale stream key",
        }

    def exponent(self, rng) -> int:
        if self.scenario == "unit":
            return 0
        if self.scenario == "narrow":
            return rng.randrange(-2, 3)
        if rng.random() < 0.99:
            return 0
        return 8

    def sample(
        self, seed: int, matrix: str, grid_rows: int, grid_cols: int, pair_index: int = 0
    ) -> list[int]:
        """Row-major E8M0 codes over the logical scale grid of one pair."""
        if pair_index < 0:
            raise ValueError("pair_index must be >= 0")
        codes: list[int] = []
        for grid_row in range(grid_rows):
            for grid_col in range(grid_cols):
                if self.scenario == "unit":
                    exponent = 0
                else:
                    rng = mx_stream_rng(
                        seed, "scale", matrix, self.scenario, pair_index, grid_row, grid_col, "scale"
                    )
                    exponent = self.exponent(rng)
                codes.append(exponent_to_code(exponent))
        return codes
