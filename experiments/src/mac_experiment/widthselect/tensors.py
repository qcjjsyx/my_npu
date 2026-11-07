"""Target-format input tensors (MXFP8 and TF32) and FP32 -> target quantisation.

Both formats are quantised from the *same* freshly generated FP32 input bits of
one matrix pair.  MXFP8 reuses the verified quantiser in ``matrix/inputs.py``
(scale covering the group maximum, E4M3 RNE, subnormals kept, canonical zero
group scale).  TF32 uses the new explicit RNE quantiser in ``tf32_rne.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..formats.e4m3_element import decode_element
from ..formats.e8m0 import code_to_exponent
from ..matrix.inputs import quantize_fp32_bits_to_mx, validate_finite_bits
from ..matrix.tensor import MxTensor
from .tf32_rne import decode_tf32_value, is_special_code, round_dyadic_to_tf32

TARGET_FORMATS = ("mx_fp8", "tf32")


@dataclass
class TargetTensor:
    format: str
    rows: int
    cols: int
    codes: list[int]
    scale_codes: list[int] | None = None
    group_axis: int | None = None
    group_length: int = 32
    element_format: str | None = None
    scale_format: str | None = None

    def __post_init__(self) -> None:
        if self.format not in TARGET_FORMATS:
            raise ValueError(f"unknown target format {self.format!r}")
        if len(self.codes) != self.rows * self.cols:
            raise ValueError("code count does not match shape")
        if self.format == "mx_fp8":
            if self.scale_codes is None:
                raise ValueError("MXFP8 tensor requires scale codes")
            grid_cols = self.cols // self.group_length if self.group_axis == 1 else self.cols
            grid_rows = self.rows // self.group_length if self.group_axis == 0 else self.rows
            if len(self.scale_codes) != grid_rows * grid_cols:
                raise ValueError("scale code count does not match the group grid")

    @staticmethod
    def from_mx(name: str, mx: MxTensor) -> "TargetTensor":
        return TargetTensor(
            format="mx_fp8",
            rows=mx.rows,
            cols=mx.cols,
            codes=list(mx.element_codes),
            scale_codes=list(mx.scale_codes),
            group_axis=mx.group_axis,
            group_length=mx.group_length,
            element_format=mx.element_format,
            scale_format=mx.scale_format,
        )

    def scale_index(self, row: int, col: int) -> int:
        grid_cols = self.cols // self.group_length if self.group_axis == 1 else self.cols
        if self.group_axis == 0:
            return (row // self.group_length) * grid_cols + col
        return row * grid_cols + (col // self.group_length)

    def decoded_value(self, row: int, col: int) -> Dyadic:
        code = self.codes[row * self.cols + col]
        if self.format == "tf32":
            return decode_tf32_value(code)
        element = decode_element(code)
        if element.is_zero():
            return element
        scale = code_to_exponent(self.scale_codes[self.scale_index(row, col)])
        return Dyadic(element.m, element.e + scale)

    def code_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for code in self.codes:
            digest.update(int(code).to_bytes(4, "little"))
        return "sha256:" + digest.hexdigest()

    def scale_fingerprint(self) -> str:
        if self.scale_codes is None:
            return None
        digest = hashlib.sha256()
        for code in self.scale_codes:
            digest.update(int(code).to_bytes(1, "little"))
        return "sha256:" + digest.hexdigest()

    def to_dict(self) -> dict:
        payload = {
            "format": self.format,
            "shape": [self.rows, self.cols],
            "codes": list(self.codes),
            "code_fingerprint": self.code_fingerprint(),
        }
        if self.format == "mx_fp8":
            payload.update(
                {
                    "element_format": self.element_format or "fp8_e4m3",
                    "scale_format": self.scale_format or "e8m0",
                    "group_axis": self.group_axis,
                    "group_length": self.group_length,
                    "scale_codes": list(self.scale_codes),
                    "scale_fingerprint": self.scale_fingerprint(),
                }
            )
        return payload

    @staticmethod
    def from_dict(payload: dict) -> "TargetTensor":
        fmt = payload["format"]
        shape = payload["shape"]
        return TargetTensor(
            format=fmt,
            rows=int(shape[0]),
            cols=int(shape[1]),
            codes=[int(code) for code in payload["codes"]],
            scale_codes=(
                [int(code) for code in payload["scale_codes"]]
                if payload.get("scale_codes") is not None
                else None
            ),
            group_axis=payload.get("group_axis"),
            group_length=int(payload.get("group_length", 32)),
            element_format=payload.get("element_format"),
            scale_format=payload.get("scale_format"),
        )


@dataclass
class TargetInput:
    format: str
    tensor: TargetTensor
    events: dict
    diagnostics: dict


def _mean_of_squares(values: list[Dyadic]) -> float:
    import math

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


def quantize_fp32_bits(
    fmt: str,
    name: str,
    bits: list[int],
    rows: int,
    cols: int,
    group_axis: int,
) -> TargetInput:
    """Quantise one FP32 bit matrix to the requested target format."""
    from ..arithmetic.fp32 import Fp32

    validate_finite_bits(bits, f"{name}/{fmt}")
    if len(bits) != rows * cols:
        raise ValueError(f"bit count {len(bits)} != {rows * cols}")
    if fmt == "mx_fp8":
        mx_input = quantize_fp32_bits_to_mx(name, bits, rows, cols, group_axis)
        return TargetInput(
            format="mx_fp8",
            tensor=TargetTensor.from_mx(name, mx_input.tensor),
            events={key: int(value) for key, value in mx_input.events.items()},
            diagnostics=mx_input.diagnostics,
        )
    if fmt == "tf32":
        codes: list[int] = []
        decoded: list[Dyadic] = []
        n_saturated = 0
        n_zeroed = 0
        n_signed_zero = 0
        fp32_values: list[Dyadic] = []
        for raw in bits:
            state = Fp32.from_bits(int(raw))
            value = state.exact_dyadic()
            fp32_values.append(value)
            zero_sign = state.sign if state.is_zero() else 0
            result = round_dyadic_to_tf32(value, zero_sign=zero_sign, overflow="saturate")
            if result.overflowed:
                n_saturated += 1
            if result.zeroed:
                n_zeroed += 1
                if zero_sign:
                    n_signed_zero += 1
            if is_special_code(result.code):
                raise AssertionError("TF32 input quantization must never emit a special code")
            codes.append(result.code)
            decoded.append(decode_tf32_value(result.code))
        nonzero = sum(1 for value in decoded if not value.is_zero())
        tensor = TargetTensor(format="tf32", rows=rows, cols=cols, codes=codes)
        n_elements = rows * cols
        diagnostics = {
            "definition": "input diagnostics only; not error metrics of this experiment",
            "fp32_rms": _mean_of_squares(fp32_values),
            "fp32_max_abs": _max_abs(fp32_values),
            "fp32_nonzero_elements": sum(1 for value in fp32_values if not value.is_zero()),
            "tf32_decoded_rms": _mean_of_squares(decoded),
            "tf32_max_decoded_abs": _max_abs(decoded),
            "tf32_saturated_elements": n_saturated,
            "tf32_zero_elements": n_elements - nonzero,
            "tf32_zero_fraction": (n_elements - nonzero) / n_elements if n_elements else None,
            "signed_zero_elements": n_signed_zero,
        }
        events = {"element_overflow_saturated": n_saturated, "element_zeroed": n_zeroed}
        return TargetInput(format="tf32", tensor=tensor, events=events, diagnostics=diagnostics)
    raise ValueError(f"unknown target format {fmt!r}")


def metadata() -> dict:
    from .tf32_rne import metadata as tf32_metadata

    return {
        "target_formats": list(TARGET_FORMATS),
        "shared_fp32_input": "one freshly generated FP32 matrix pair is quantised to every configured format",
        "mx_fp8": {
            "element_format": "fp8_e4m3",
            "scale_format": "e8m0",
            "group_length": 32,
            "x_group_axis": 1,
            "w_group_axis": 0,
            "scale_policy": "minimal binary power covering the group maximum magnitude",
            "element_policy": "E4M3 RNE, subnormals kept, saturation to +/-448",
        },
        "tf32": tf32_metadata(),
    }
