"""Run directed entry/M stall and invalid-input simulations."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RTL = (
    "hardware/rtl/systemverilog/mac_types_pkg.sv",
    "hardware/rtl/systemverilog/mac_fp19_convert.sv",
    "hardware/rtl/systemverilog/mac_fp9_convert.sv",
    "hardware/rtl/systemverilog/mac_operand_pack.sv",
    "hardware/rtl/systemverilog/mac_mul_frontend.sv",
    "hardware/rtl/systemverilog/mac_entry_m_stage.sv",
)


def _compile(tb: str, image: Path) -> None:
    result = subprocess.run(
        ["iverilog", "-g2012", "-s", tb, "-o", str(image), *RTL,
         f"hardware/verification/tb/{tb}.sv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_entry_m_int16_stalls_refill_and_metadata(tmp_path: Path):
    image = tmp_path / "entry_m.vvp"
    _compile("tb_mac_entry_m_stage", image)
    result = subprocess.run(["vvp", str(image)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mac_entry_m_stage PASS" in result.stdout


@pytest.mark.parametrize(
    "bad_case,marker",
    [
        (1, "illegal source-format combination"),
        (2, "reserved INT16 input"),
        (3, "E8M0 255"),
        (4, "invalid group_valid"),
        (5, "INT4 raw upper byte"),
        (6, "reserved INT8 input"),
        (7, "reserved INT4 input"),
        (8, "illegal source-format combination or MX enable"),
    ],
)
def test_entry_m_contract_violation_is_detected(tmp_path: Path, bad_case: int, marker: str):
    image = tmp_path / "entry_m_invalid.vvp"
    _compile("tb_mac_entry_m_invalid", image)
    result = subprocess.run(
        ["vvp", str(image), f"+CASE={bad_case}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert marker in result.stdout + result.stderr
    assert "silently accepted" not in result.stdout + result.stderr
