"""Numeric-domain FP32 input generation and G32 MXFP8 input quantisation.

Current (active) input scheme:

1. sample values in the numeric domain (gaussian / laplace / student_t3 /
   outlier) with explicit, stable random streams;
2. convert each value to binary32 with one explicit RNE step computed from the
   exact dyadic value (``round_dyadic_to_fp32`` from integer arithmetic) and
   record the final FP32 bits;
3. quantise the FP32 matrices with the existing exact MX quantiser, reusing the
   "smallest binary power covering the group maximum" scale policy, E4M3 RNE
   elements, subnormals and the canonical all-zero group scale.

The distributions are experiment definitions with theoretical mean 0 and
theoretical RMS 1; they do not claim to represent real LLM tensors.  Samples are
produced by finite-precision algorithms (binary64 math + exact dyadic RNE to
FP32) and are **not** described as mathematically exact samples of the
continuous distribution.

The retired v1 scheme (independent E4M3 element + E8M0 scale sampling) lives in
``matrix/sampling.py`` and is no longer called by the active engine, config or
analysis.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..arithmetic.dyadic import Dyadic
from ..arithmetic.fp32 import Fp32
from ..arithmetic.rounding import round_dyadic_to_fp32
from ..formats.e4m3_element import is_nan_code
from ..sampling.streams import mx_input_stream_rng
from ..version import MX_INPUT_SAMPLING_VERSION
from .quantize import quantize_matrix_independent
from .tensor import GROUP_LENGTH, MxTensor

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
OUTLIER_VARIANCE = 0.99 * 1.0 + 0.01 * 64.0  # = 1.63
OUTLIER_NORMALISER = math.sqrt(OUTLIER_VARIANCE)

INPUT_DISTRIBUTIONS = ("gaussian", "laplace", "student_t3", "outlier")

DISTRIBUTION_METADATA: dict[str, dict] = {
    "gaussian": {
        "family": "normal",
        "parameters": {"mean": 0.0, "std": 1.0},
        "theoretical_mean": 0.0,
        "theoretical_rms": 1.0,
        "algorithm": "Box-Muller from two uniform binary64 draws: sqrt(-2 ln u1) cos(2 pi u2)",
    },
    "laplace": {
        "family": "laplace",
        "parameters": {"location": 0.0, "scale": 1.0 / SQRT2},
        "theoretical_mean": 0.0,
        "theoretical_rms": 1.0,
        "algorithm": "inverse CDF from one uniform draw: -b sign(u) ln(1 - 2|u|), b = 1/sqrt(2)",
    },
    "student_t3": {
        "family": "student_t",
        "parameters": {"dof": 3, "normaliser": SQRT3},
        "theoretical_mean": 0.0,
        "theoretical_rms": 1.0,
        "algorithm": "Z / sqrt(G1^2 + G2^2 + G3^2) with iid standard normal Z, Gi (t3 / sqrt(3))",
    },
    "outlier": {
        "family": "gaussian_mixture",
        "parameters": {
            "p_small": 0.99,
            "sigma_small": 1.0,
            "p_large": 0.01,
            "sigma_large": 8.0,
            "variance": OUTLIER_VARIANCE,
            "normaliser": OUTLIER_NORMALISER,
        },
        "theoretical_mean": 0.0,
        "theoretical_rms": 1.0,
        "algorithm": (
            "0.99 * N(0,1) + 0.01 * N(0,64), then divided by sqrt(1.63)"
        ),
    },
}


CUSTOM_PARAMETER_ERROR = (
    "custom distribution parameters are not supported: the four input distributions are fixed "
    "unit-RMS experiment definitions (gaussian, laplace, student_t3, outlier); remove "
    "activation_parameters / weight_parameters from the configuration"
)


def distribution_label(distribution: str) -> str:
    """Stable stream/grouping label of a fixed distribution (no custom parameters)."""
    return distribution


def fp32_bits_from_float(value: float) -> int:
    """Explicit RNE conversion of a finite binary64 value to FP32 bits.

    The conversion is done on the exact dyadic value with integer rounding; no
    implicit host float cast defines the semantics.  Non-finite values raise.
    """
    if not math.isfinite(value):
        raise ValueError(f"non-finite input value generated: {value!r}")
    if value == 0.0:
        return 0x80000000 if math.copysign(1.0, value) < 0 else 0x00000000
    numerator, denominator = value.as_integer_ratio()
    if denominator & (denominator - 1):
        raise ValueError("binary64 value is not a dyadic rational")
    dyadic = Dyadic(numerator, -(denominator.bit_length() - 1))
    bits, _ = round_dyadic_to_fp32(dyadic)
    if not bits.is_finite():
        raise ValueError(f"input value overflowed FP32: {value!r}")
    return bits.bits


def _gaussian(rng) -> float:
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _laplace(rng) -> float:
    u = rng.random() - 0.5
    magnitude = (1.0 / SQRT2) * (-math.log(1.0 - 2.0 * abs(u)))
    return -magnitude if u < 0.0 else magnitude


def _student_t3(rng) -> float:
    z = _gaussian(rng)
    chi2 = _gaussian(rng) ** 2 + _gaussian(rng) ** 2 + _gaussian(rng) ** 2
    return z / math.sqrt(chi2)


def _outlier(rng) -> float:
    if rng.random() < 0.99:
        z = _gaussian(rng)
    else:
        z = 8.0 * _gaussian(rng)
    return z / OUTLIER_NORMALISER


GENERATORS = {
    "gaussian": _gaussian,
    "laplace": _laplace,
    "student_t3": _student_t3,
    "outlier": _outlier,
}


@dataclass
class Fp32InputSampler:
    distribution: str
    parameters: dict | None = None

    def __post_init__(self) -> None:
        if self.parameters not in (None, {}):
            raise ValueError(CUSTOM_PARAMETER_ERROR)
        if self.distribution not in INPUT_DISTRIBUTIONS:
            raise ValueError(
                f"unknown input distribution {self.distribution!r}; known: {list(INPUT_DISTRIBUTIONS)}"
            )
        self._generator = GENERATORS[self.distribution]
        self._label = distribution_label(self.distribution)

    def metadata(self) -> dict:
        meta = dict(DISTRIBUTION_METADATA[self.distribution])
        meta.update(
            {
                "distribution": self.distribution,
                "sampling_version": MX_INPUT_SAMPLING_VERSION,
                "stream_label": self._label,
                "parameters_configurable": False,
                "fp32_conversion": (
                    "each binary64 draw -> exact dyadic -> integer RNE to binary32; "
                    "final FP32 bits are recorded; non-finite draws raise"
                ),
                "synthetic_disclaimer": (
                    "numeric-domain experiment distributions; they do not claim to represent real LLM tensors"
                ),
            }
        )
        return meta

    def value(self, rng) -> float:
        return self._generator(rng)

    def sample_matrix(self, seed: int, matrix: str, rows: int, cols: int, pair_index: int) -> list[int]:
        """Row-major FP32 bits for a ``rows x cols`` matrix of one pair."""
        if rows < 0 or cols < 0:
            raise ValueError("rows/cols must be non-negative")
        if pair_index < 0:
            raise ValueError("pair_index must be >= 0")
        bits: list[int] = []
        for row in range(rows):
            rng = mx_input_stream_rng(seed, matrix, self._label, pair_index, row, "value")
            for _ in range(cols):
                bits.append(fp32_bits_from_float(self.value(rng)))
        return bits


def _mean_of_squares(values: list[Dyadic]) -> float:
    if not values:
        return 0.0
    total = Dyadic.zero()
    for value in values:
        total = total + value * value
    return math.sqrt(float(total.to_fraction() / len(values)))


def _max_abs(values: list[Dyadic]) -> float:
    maximum = Dyadic.zero()
    for value in values:
        magnitude = abs(value)
        if maximum < magnitude:
            maximum = magnitude
    return maximum.to_float()


def fp32_bits_to_dyadics(bits: list[int]) -> tuple[list[Dyadic], list[int]]:
    """Exact dyadic values and zero signs of a saved FP32 bit row-major matrix."""
    values: list[Dyadic] = []
    zero_signs: list[int] = []
    for raw in bits:
        state = Fp32.from_bits(int(raw))
        if not state.is_finite():
            raise ValueError(f"non-finite FP32 bits in saved input: 0x{state.bits:08X}")
        values.append(state.exact_dyadic())
        zero_signs.append(state.sign if state.is_zero() else 0)
    return values, zero_signs


@dataclass
class MxInputMatrix:
    tensor: MxTensor
    events: dict
    diagnostics: dict


def quantize_fp32_bits_to_mx(
    name: str,
    bits: list[int],
    rows: int,
    cols: int,
    group_axis: int,
    group_length: int = GROUP_LENGTH,
) -> MxInputMatrix:
    """Quantise an FP32 bit matrix to MXFP8 with the exact MX quantiser.

    ``group_axis=1`` groups X along K (columns), ``group_axis=0`` groups W along
    K (rows).  All maxima, scale choices and element roundings are computed from
    exact dyadics decoded from the FP32 bits.
    """
    if group_length != GROUP_LENGTH:
        raise ValueError(f"group length must be {GROUP_LENGTH}")
    if len(bits) != rows * cols:
        raise ValueError(f"bit count {len(bits)} != {rows * cols}")
    values, zero_signs = fp32_bits_to_dyadics(bits)
    quant = quantize_matrix_independent(
        values, zero_signs, rows, cols, group_length=group_length, group_axis=group_axis
    )
    groups = quant.groups
    element_codes: list[int] = [0] * (rows * cols)
    scale_codes: list[int] = []
    from .quantize import group_index_lists

    positions = group_index_lists(rows, cols, group_length, group_axis)
    for group, indices in zip(groups, positions):
        if group.invalid or group.element_codes is None or group.scale_code is None:
            raise ValueError(f"input quantisation produced an invalid group: {group.invalid_reason}")
        scale_codes.append(group.scale_code)
        for position, index in zip(group.element_codes, indices):
            element_codes[index] = position
    tensor = MxTensor(name, (rows, cols), group_axis, element_codes, scale_codes)
    diagnostics = _input_diagnostics(values, tensor, bits, group_axis)
    events = {key: int(value) for key, value in sorted(quant.events.items())}
    return MxInputMatrix(tensor=tensor, events=events, diagnostics=diagnostics)


def _input_diagnostics(values: list[Dyadic], tensor: MxTensor, bits: list[int], group_axis: int) -> dict:
    decoded: list[Dyadic] = []
    zero_count = 0
    code_histogram: Counter = Counter()
    scale_exponents: Counter = Counter()
    for row in range(tensor.rows):
        for col in range(tensor.cols):
            code = tensor.element_code(row, col)
            code_histogram[code] += 1
            element = tensor.decoded_element(row, col)
            if element.is_zero():
                zero_count += 1
            decoded.append(tensor.decoded_value(row, col))
    for row in range(tensor.rows):
        for col in range(tensor.cols):
            scale_exponents[tensor.scale_exponent(row, col)] += 1
    fp32_nonzero = sum(1 for value in values if not value.is_zero())
    return {
        "definition": "input diagnostics only; not error metrics of this experiment",
        "group_axis": group_axis,
        "fp32_rms": _mean_of_squares(values),
        "fp32_max_abs": _max_abs(values),
        "fp32_nonzero_elements": fp32_nonzero,
        "mx_decoded_rms": _mean_of_squares(decoded),
        "mx_max_decoded_abs": _max_abs(decoded),
        "mx_zero_elements": zero_count,
        "mx_zero_fraction": (zero_count / len(values)) if values else None,
        "scale_exponent_histogram": {
            str(key): scale_exponents[key] for key in sorted(scale_exponents)
        },
        "element_code_histogram": {str(key): code_histogram[key] for key in sorted(code_histogram)},
    }


def validate_finite_bits(bits: list[int], label: str) -> None:
    """Reject NaN/Inf FP32 bits in a saved or generated input matrix."""
    for raw in bits:
        state = Fp32.from_bits(int(raw))
        if not state.is_finite():
            raise ValueError(f"non-finite FP32 bits in {label}: 0x{state.bits:08X}")


def bits_are_finite(bits) -> bool:
    """Fast integer-only check: true when no bit pattern has a NaN/Inf exponent."""
    try:
        for raw in bits:
            if ((int(raw) >> 23) & 0xFF) == 0xFF:
                return False
    except (TypeError, ValueError):
        return False
    return True


def element_codes_are_finite(codes: list[int]) -> bool:
    return not any(is_nan_code(int(code)) for code in codes)


def metadata() -> dict:
    return {
        "input_sampling_version": MX_INPUT_SAMPLING_VERSION,
        "distributions": {name: DISTRIBUTION_METADATA[name] for name in INPUT_DISTRIBUTIONS},
        "fp32_conversion": "explicit integer RNE from the exact dyadic of each binary64 draw",
        "input_quantization": {
            "group_length": GROUP_LENGTH,
            "x_group_axis": 1,
            "w_group_axis": 0,
            "scale_policy": "minimal binary power covering the group maximum magnitude",
            "element_policy": "E4M3 RNE, subnormals kept, saturation to +/-448",
            "all_zero_group": "canonical scale code 0x7F (value 1.0)",
            "exactness": "maxima, thresholds, scale choices and roundings computed from exact dyadics",
        },
    }
