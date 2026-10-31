"""Input samplers.

Sampling happens directly on the legal encoding set (never continuous values
that are quantised afterwards).  Magnitude draws, sign draws and lane
permutations use independent deterministic streams.  The cancellation
distribution assigns product signs greedily against the running exact block
sum and never rejects amplitude samples.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..formats.base import Format
from .pmf import DiscretePMF, build_magnitude_pmf
from .streams import stream_rng

LANES = 16


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    magnitude_kind: str
    shape: str | None
    sign_mode: str
    note: str = ""


DISTRIBUTIONS: dict[str, DistributionSpec] = {
    "positive": DistributionSpec(
        "positive", "exponent_uniform", None, "positive", "both operands positive"
    ),
    "uniform": DistributionSpec(
        "uniform", "exponent_uniform", None, "fair", "positive amplitude rule with fair independent signs"
    ),
    "exponent_gaussian": DistributionSpec(
        "exponent_gaussian", "exponent_gaussian", None, "fair", "truncated discrete Gaussian raw exponents"
    ),
    "cancellation": DistributionSpec(
        "cancellation", "exponent_gaussian", None, "cancellation", "greedy opposite-sign assignment, no rejection"
    ),
    "laplace": DistributionSpec("laplace", "code_interval", "laplace", "fair", "synthetic unit-variance Laplace"),
    "student_t3": DistributionSpec("student_t3", "code_interval", "student_t3", "fair", "synthetic t(3)/sqrt(3)"),
    "outlier": DistributionSpec("outlier", "code_interval", "outlier", "fair", "synthetic Gaussian outlier mixture"),
}

DEFAULT_DISTRIBUTIONS = tuple(DISTRIBUTIONS)

CANCELLATION_SIGN_MODES = ("greedy", "independent")


@dataclass
class BlockDraw:
    pairs: list[tuple[int, int]]
    magnitudes: list[Dyadic]
    product_signs: list[int]
    rho: float | None = None


class Sampler:
    def __init__(self, fmt: Format, distribution: str, cancellation_sign_mode: str = "greedy"):
        if distribution not in DISTRIBUTIONS:
            raise KeyError(f"unknown distribution {distribution!r}; known: {list(DISTRIBUTIONS)}")
        if cancellation_sign_mode not in CANCELLATION_SIGN_MODES:
            raise ValueError(f"unknown cancellation sign mode {cancellation_sign_mode!r}")
        self.fmt = fmt
        self.spec = DISTRIBUTIONS[distribution]
        self.cancellation_sign_mode = cancellation_sign_mode
        self.pmf: DiscretePMF = build_magnitude_pmf(fmt, self.spec.magnitude_kind, self.spec.shape)

    def metadata(self) -> dict:
        meta = {
            "name": self.spec.name,
            "magnitude": self.spec.magnitude_kind,
            "shape": self.spec.shape,
            "sign_mode": self.spec.sign_mode,
            "note": self.spec.note,
            "pmf": self.pmf.metadata,
            "synthetic_disclaimer": "synthetic shape distributions do not claim to represent real LLM tensors",
        }
        if self.spec.sign_mode == "cancellation":
            meta["cancellation_sign_mode"] = self.cancellation_sign_mode
        return meta

    def sample_block(self, master_seed: int, sample_index: int, block_index: int) -> BlockDraw:
        if self.spec.sign_mode == "cancellation":
            return self._sample_cancellation(master_seed, sample_index, block_index)
        return self._sample_independent(master_seed, sample_index, block_index)

    def _sample_independent(self, master_seed: int, sample_index: int, block_index: int) -> BlockDraw:
        fmt = self.fmt
        rng_mag = stream_rng(master_seed, fmt.name, self.spec.name, sample_index, block_index, "amplitude")
        rng_sign = stream_rng(master_seed, fmt.name, self.spec.name, sample_index, block_index, "sign")
        magnitudes: list[Dyadic] = []
        pairs: list[tuple[int, int]] = []
        product_signs: list[int] = []
        for _ in range(LANES):
            signs = []
            operands = []
            for _side in range(2):
                code = self.pmf.sample(rng_mag)
                operand = fmt.decode(code)
                if self.spec.sign_mode == "positive":
                    sign = 0
                else:
                    sign = rng_sign.getrandbits(1)
                signs.append(sign)
                operands.append(fmt.encode(sign, operand.raw_exp, operand.frac))
            magnitudes.append(fmt.operand_value(fmt.decode(operands[0])) * fmt.operand_value(fmt.decode(operands[1])))
            product_signs.append(1 if signs[0] == signs[1] else -1)
            pairs.append((operands[0], operands[1]))
        return BlockDraw(pairs=pairs, magnitudes=magnitudes, product_signs=product_signs)

    def _sample_cancellation(self, master_seed: int, sample_index: int, block_index: int) -> BlockDraw:
        fmt = self.fmt
        rng_mag = stream_rng(master_seed, fmt.name, self.spec.name, sample_index, block_index, "amplitude")
        rng_sign = stream_rng(master_seed, fmt.name, self.spec.name, sample_index, block_index, "sign")
        rng_lane = stream_rng(master_seed, fmt.name, self.spec.name, sample_index, block_index, "lane")

        raw_codes = [self.pmf.sample(rng_mag) for _ in range(2 * LANES)]
        operands = [fmt.decode(code) for code in raw_codes]
        magnitudes = [
            fmt.operand_value(operands[2 * i]) * fmt.operand_value(operands[2 * i + 1]) for i in range(LANES)
        ]

        if self.cancellation_sign_mode == "greedy":
            product_signs = self._greedy_signs(magnitudes, rng_sign)
        else:
            product_signs = [1 if rng_sign.getrandbits(1) else -1 for _ in range(LANES)]

        lanes: list[tuple[int, int, int, Dyadic]] = []
        for i in range(LANES):
            operand_a, operand_b = operands[2 * i], operands[2 * i + 1]
            sign_a = rng_sign.getrandbits(1)
            sign_b = sign_a ^ (0 if product_signs[i] > 0 else 1)
            code_a = fmt.encode(sign_a, operand_a.raw_exp, operand_a.frac)
            code_b = fmt.encode(sign_b, operand_b.raw_exp, operand_b.frac)
            lanes.append((code_a, code_b, product_signs[i], magnitudes[i]))
        rng_lane.shuffle(lanes)

        total = Dyadic.zero()
        l1 = Dyadic.zero()
        for magnitude, sign in zip(magnitudes, product_signs):
            total = total + (magnitude if sign > 0 else -magnitude)
            l1 = l1 + magnitude
        rho = float(abs(total).to_fraction() / l1.to_fraction()) if not l1.is_zero() else 0.0
        return BlockDraw(
            pairs=[(a, b) for a, b, _, _ in lanes],
            magnitudes=[m for _, _, _, m in lanes],
            product_signs=[s for _, _, s, _ in lanes],
            rho=rho,
        )

    @staticmethod
    def _greedy_signs(magnitudes: list[Dyadic], rng_sign) -> list[int]:
        order = sorted(range(len(magnitudes)), key=lambda i: (-magnitudes[i].to_fraction(), i))
        signs = [0] * len(magnitudes)
        running = Dyadic.zero()
        for i in order:
            if running.m > 0:
                sign = -1
            elif running.m < 0:
                sign = 1
            else:
                sign = 1 if rng_sign.getrandbits(1) else -1
            signs[i] = sign
            running = running + (magnitudes[i] if sign > 0 else -magnitudes[i])
        return signs
