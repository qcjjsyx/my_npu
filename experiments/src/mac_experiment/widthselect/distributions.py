"""Six fixed-parameter input distributions and shared FP32 synthesis.

The four historical generators (gaussian / laplace / student_t3 / outlier) are
imported verbatim from ``matrix.inputs`` so the new experiment reuses the same
numeric definitions without redefining or altering them.  Two additional fixed
definitions are added for this experiment and are defined **directly on the
finite binary32 encoding set** (no continuous shape, no target-format codes):

* ``uniform``: every finite binary32 encoding equally likely -- fair sign and a
  uniform 31-bit magnitude field over ``[0, 0x7F7FFFFF]`` (zeros, subnormals and
  normal values; the special exponent field 255 is excluded);
* ``positive``: the same encoding-uniform rule restricted to sign 0 and a
  nonzero magnitude -- every generated value is a positive finite binary32
  number (subnormals included, no zero).

Because these two distributions are already exact finite binary32 encodings,
the explicit RNE-to-FP32 pipeline step is the identity for them (documented in
the metadata); the four historical distributions still go through the numeric
generator and the exact integer-RNE conversion.

No custom parameters are supported; any attempt raises explicitly.  Values are
synthesised per element from the new ``mac-width-fp32-stream-v1`` namespace.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..matrix.inputs import (
    CUSTOM_PARAMETER_ERROR,
    DISTRIBUTION_METADATA as _LEGACY_METADATA,
    GENERATORS as _LEGACY_GENERATORS,
    fp32_bits_from_float,
)
from ..version import WIDTHSELECT_INPUT_SAMPLING_VERSION
from .streams import width_input_stream_rng

# finite binary32 magnitude field range: exponent field 0..254, any fraction
MAGNITUDE_FIELD_LIMIT = 0x7F80_0000  # first special encoding (Inf/NaN)
LEGACY_DISTRIBUTIONS = ("gaussian", "laplace", "student_t3", "outlier")
ENCODING_DISTRIBUTIONS = ("uniform", "positive")
WIDTH_DISTRIBUTIONS = LEGACY_DISTRIBUTIONS + ENCODING_DISTRIBUTIONS

FRACTION_COUNT = 1 << 23


def _uniform_encoding_bits(rng) -> int:
    """Uniform draw over all finite binary32 encodings."""
    sign = rng.getrandbits(1)
    magnitude = rng.randrange(0, MAGNITUDE_FIELD_LIMIT)
    return (sign << 31) | magnitude


def _positive_encoding_bits(rng) -> int:
    """Uniform draw over the positive finite binary32 encodings (zero excluded)."""
    return rng.randrange(1, MAGNITUDE_FIELD_LIMIT)


BITS_GENERATORS = {
    "uniform": _uniform_encoding_bits,
    "positive": _positive_encoding_bits,
}

FLOAT_GENERATORS = {
    "gaussian": _LEGACY_GENERATORS["gaussian"],
    "laplace": _LEGACY_GENERATORS["laplace"],
    "student_t3": _LEGACY_GENERATORS["student_t3"],
    "outlier": _LEGACY_GENERATORS["outlier"],
}

GENERATORS = {**FLOAT_GENERATORS, **BITS_GENERATORS}


def _encoding_uniform_moments() -> dict:
    """Exact closed-form moments of the encoding-uniform draws (binary64 floats)."""
    fraction_count = FRACTION_COUNT
    magnitude_count = 255 * fraction_count  # exponent fields 0..254

    def sum_of_squares(n: int) -> int:
        return n * (n + 1) * (2 * n + 1) // 6

    # sum of squared magnitudes over the positive finite encoding set
    subnormal_squares = Fraction(sum_of_squares(fraction_count - 1), 1 << 298)
    normal_significand_squares = sum_of_squares(2 * fraction_count - 1) - sum_of_squares(
        fraction_count - 1
    )
    exponent_squares = Fraction(4**255 - 4, 3) * Fraction(1, 1 << 300)
    total_squares = subnormal_squares + normal_significand_squares * exponent_squares

    # sum of positive finite magnitudes (zero contributes nothing)
    subnormal_sum = Fraction(fraction_count * (fraction_count - 1), 2) * Fraction(1, 1 << 149)
    exponent_sum = sum(
        (Fraction(2) ** (exponent - 150) for exponent in range(1, 255)), Fraction(0)
    )
    normal_sum = exponent_sum * Fraction(fraction_count * (3 * fraction_count - 1), 2)
    positive_count = magnitude_count - 1

    return {
        # signs double both the squared sum and the sample count, so the mean
        # square for the fair-sign uniform draw is total_squares/magnitude_count
        "uniform_rms": math.sqrt(float(total_squares / magnitude_count)),
        "positive_mean": float((subnormal_sum + normal_sum) / positive_count),
        "positive_rms": math.sqrt(float(total_squares / positive_count)),
        "uniform_support_size": 2 * magnitude_count,
        "positive_support_size": positive_count,
    }


_MOMENTS = _encoding_uniform_moments()

_NEW_METADATA: dict[str, dict] = {
    "uniform": {
        "family": "uniform_finite_encoding",
        "parameters": {
            "support": "all finite binary32 encodings",
            "support_size": _MOMENTS["uniform_support_size"],
            "rule": "fair sign; 31-bit magnitude field uniform on [0, 0x7F7FFFFF]",
            "excluded": "exponent field 255 (Inf/NaN)",
            "includes": "signed zeros, subnormals, normal values",
        },
        "theoretical_mean": 0.0,
        "theoretical_rms": _MOMENTS["uniform_rms"],
        "algorithm": "uniform draw over the finite binary32 encoding set (encoding-domain)",
        "note": "not unit-RMS by construction; no per-matrix or per-group normalization is applied",
    },
    "positive": {
        "family": "uniform_finite_encoding_positive",
        "parameters": {
            "support": "positive finite binary32 encodings (zero excluded)",
            "support_size": _MOMENTS["positive_support_size"],
            "rule": "sign fixed to 0; magnitude field uniform on [1, 0x7F7FFFFF]",
            "includes": "positive subnormals and normal values",
        },
        "theoretical_mean": _MOMENTS["positive_mean"],
        "theoretical_rms": _MOMENTS["positive_rms"],
        "algorithm": "uniform draw over the positive finite binary32 encoding set (encoding-domain)",
        "note": "every generated value is strictly positive; no normalization is applied",
    },
}

DISTRIBUTION_METADATA: dict[str, dict] = {}
for _name in LEGACY_DISTRIBUTIONS:
    DISTRIBUTION_METADATA[_name] = {
        **_LEGACY_METADATA[_name],
        "source": "reused verbatim from matrix.inputs (historical generator)",
    }
DISTRIBUTION_METADATA.update(_NEW_METADATA)


def distribution_label(distribution: str) -> str:
    """Stream label of a fixed distribution (no configurable parameters)."""
    return distribution


def validate_distribution(distribution: str, parameters: dict | None = None) -> None:
    if parameters not in (None, {}):
        raise ValueError(CUSTOM_PARAMETER_ERROR)
    if distribution not in WIDTH_DISTRIBUTIONS:
        raise ValueError(
            f"unknown input distribution {distribution!r}; known: {list(WIDTH_DISTRIBUTIONS)}"
        )


def distribution_metadata(distribution: str) -> dict:
    validate_distribution(distribution)
    if distribution in ENCODING_DISTRIBUTIONS:
        conversion = (
            "encoding-domain draw: the value is already an exact finite binary32 encoding, "
            "so the RNE-to-FP32 step is the identity"
        )
    else:
        conversion = (
            "each binary64 draw -> exact dyadic -> integer RNE to binary32; "
            "final FP32 bits are recorded; non-finite draws raise"
        )
    return {
        **DISTRIBUTION_METADATA[distribution],
        "distribution": distribution,
        "sampling_version": WIDTHSELECT_INPUT_SAMPLING_VERSION,
        "stream_label": distribution_label(distribution),
        "parameters_configurable": False,
        "fp32_conversion": conversion,
        "stream_namespace": "mac-width-fp32-stream-v1",
        "synthetic_disclaimer": (
            "numeric-domain/encoding-domain experiment distributions; they do not claim to "
            "represent real LLM tensors"
        ),
    }


def sample_fp32_matrix(
    distribution: str,
    seed: int,
    matrix: str,
    rows: int,
    cols: int,
    pair_index: int,
) -> list[int]:
    """Row-major FP32 bits, one independent stream per element coordinate."""
    validate_distribution(distribution)
    if rows < 0 or cols < 0:
        raise ValueError("rows/cols must be non-negative")
    if pair_index < 0:
        raise ValueError("pair_index must be >= 0")
    label = distribution_label(distribution)
    bits_generator = BITS_GENERATORS.get(distribution)
    float_generator = FLOAT_GENERATORS.get(distribution)
    bits: list[int] = []
    for row in range(rows):
        for col in range(cols):
            rng = width_input_stream_rng(seed, matrix, label, pair_index, row, col)
            if bits_generator is not None:
                bits.append(int(bits_generator(rng)))
            else:
                bits.append(fp32_bits_from_float(float_generator(rng)))
    return bits


def metadata() -> dict:
    return {
        "input_sampling_version": WIDTHSELECT_INPUT_SAMPLING_VERSION,
        "distributions": {name: DISTRIBUTION_METADATA[name] for name in WIDTH_DISTRIBUTIONS},
    }
