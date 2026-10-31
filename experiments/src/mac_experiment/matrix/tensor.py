"""MX tensors: element codes + scale codes + explicit logical grouping.

A tensor stores its raw E4M3 element codes and E8M0 scale codes together with
the logical shape, the grouping axis and the group length.  The logical layout
is authoritative: ``X`` groups along K (axis 1), ``W`` groups along K (axis 0),
the output groups along N (axis 1).  Storage is row-major; scale storage is
row-major over the logical group grid.

No incomplete groups are representable: the grouped dimension must be a
multiple of the group length.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..arithmetic.dyadic import Dyadic
from ..formats.e4m3_element import E4M3Element, decode_element, is_nan_code
from ..formats.e8m0 import code_to_exponent, is_finite_code

GROUP_LENGTH = 32
ELEMENT_FORMAT = "fp8_e4m3"
SCALE_FORMAT = "e8m0"


@dataclass
class MxTensor:
    name: str
    shape: tuple[int, int]
    group_axis: int
    element_codes: list[int]
    scale_codes: list[int]
    group_length: int = GROUP_LENGTH
    element_format: str = ELEMENT_FORMAT
    scale_format: str = SCALE_FORMAT

    def __post_init__(self) -> None:
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.validate()

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def grouped_size(self) -> int:
        return self.rows if self.group_axis == 0 else self.cols

    @property
    def group_grid(self) -> tuple[int, int]:
        if self.group_axis == 0:
            return (self.rows // self.group_length, self.cols)
        return (self.rows, self.cols // self.group_length)

    def validate(self) -> None:
        if self.group_axis not in (0, 1):
            raise ValueError(f"group_axis must be 0 or 1, got {self.group_axis}")
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"tensor shape must be positive, got {self.shape}")
        if self.group_length != GROUP_LENGTH:
            raise ValueError(f"MXFP8 group_length must be {GROUP_LENGTH}, got {self.group_length}")
        if self.grouped_size % self.group_length:
            raise ValueError(
                f"grouped dimension {self.grouped_size} of {self.name!r} is not a multiple of {self.group_length}"
            )
        if len(self.element_codes) != self.rows * self.cols:
            raise ValueError(
                f"element code count {len(self.element_codes)} != {self.rows * self.cols} for {self.name!r}"
            )
        grid_rows, grid_cols = self.group_grid
        if len(self.scale_codes) != grid_rows * grid_cols:
            raise ValueError(
                f"scale code count {len(self.scale_codes)} != {grid_rows * grid_cols} for {self.name!r}"
            )
        for code in self.element_codes:
            if not 0 <= code <= 0xFF:
                raise ValueError(f"element code out of range: {code}")
        for code in self.scale_codes:
            if not is_finite_code(code):
                raise ValueError(f"reserved E8M0 scale code used in {self.name!r}: {code}")

    def has_nan_elements(self) -> bool:
        return any(is_nan_code(code) for code in self.element_codes)

    def element_code(self, row: int, col: int) -> int:
        return self.element_codes[row * self.cols + col]

    def scale_index(self, row: int, col: int) -> int:
        grid_cols = self.group_grid[1]
        if self.group_axis == 0:
            return (row // self.group_length) * grid_cols + col
        return row * grid_cols + (col // self.group_length)

    def scale_code(self, row: int, col: int) -> int:
        return self.scale_codes[self.scale_index(row, col)]

    def scale_exponent(self, row: int, col: int) -> int:
        return code_to_exponent(self.scale_code(row, col))

    def decoded_element(self, row: int, col: int) -> Dyadic:
        return decode_element(self.element_code(row, col))

    def decoded_value(self, row: int, col: int) -> Dyadic:
        element = self.decoded_element(row, col)
        if element.is_zero():
            return element
        return Dyadic(element.m, element.e + self.scale_exponent(row, col))

    def element_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for code in self.element_codes:
            digest.update(code.to_bytes(1, "little"))
        return "sha256:" + digest.hexdigest()

    def scale_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for code in self.scale_codes:
            digest.update(code.to_bytes(1, "little"))
        return "sha256:" + digest.hexdigest()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "element_format": self.element_format,
            "scale_format": self.scale_format,
            "shape": [self.rows, self.cols],
            "group_axis": self.group_axis,
            "group_length": self.group_length,
            "element_codes": list(self.element_codes),
            "scale_codes": list(self.scale_codes),
            "element_fingerprint": self.element_fingerprint(),
            "scale_fingerprint": self.scale_fingerprint(),
        }

    @staticmethod
    def from_dict(payload: dict) -> "MxTensor":
        shape = payload["shape"]
        return MxTensor(
            name=payload["name"],
            shape=(int(shape[0]), int(shape[1])),
            group_axis=int(payload["group_axis"]),
            element_codes=[int(code) for code in payload["element_codes"]],
            scale_codes=[int(code) for code in payload["scale_codes"]],
            group_length=int(payload["group_length"]),
            element_format=payload.get("element_format", ELEMENT_FORMAT),
            scale_format=payload.get("scale_format", SCALE_FORMAT),
        )


def element_metadata() -> dict:
    return E4M3Element().metadata()
