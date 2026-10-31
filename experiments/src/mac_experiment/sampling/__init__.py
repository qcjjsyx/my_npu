"""Input sampling: magnitude PMFs, sign policies, deterministic streams."""

from .pmf import DiscretePMF, build_magnitude_pmf
from .sampler import DEFAULT_DISTRIBUTIONS, DISTRIBUTIONS, BlockDraw, DistributionSpec, Sampler
from .streams import CodeFingerprint, stream_rng

__all__ = [
    "DiscretePMF",
    "build_magnitude_pmf",
    "DEFAULT_DISTRIBUTIONS",
    "DISTRIBUTIONS",
    "BlockDraw",
    "DistributionSpec",
    "Sampler",
    "CodeFingerprint",
    "stream_rng",
]
