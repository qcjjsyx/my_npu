"""Direct M-to-ALIGN interface and shared backpressure smoke test."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_m_align_int16_phases_fp9_and_backpressure(tmp_path: Path):
    image = tmp_path / "m_align.vvp"
    sources = (
        "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/rtl/systemverilog/mac_fp19_convert.sv",
        "hardware/rtl/systemverilog/mac_fp9_convert.sv", "hardware/rtl/systemverilog/mac_operand_pack.sv",
        "hardware/rtl/systemverilog/mac_mul_frontend.sv", "hardware/rtl/systemverilog/mac_entry_m_stage.sv",
        "hardware/rtl/systemverilog/mac_align_max16.sv", "hardware/rtl/systemverilog/mac_align_rne.sv",
        "hardware/rtl/systemverilog/mac_align_lane0.sv", "hardware/rtl/systemverilog/mac_align_lane1.sv",
        "hardware/rtl/systemverilog/mac_align_stage.sv", "hardware/verification/tb/tb_mac_m_align.sv",
    )
    compiled = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mac_m_align", "-o", str(image),
         *sources], cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(["vvp", str(image)], cwd=ROOT,
                               capture_output=True, text=True)
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert "mac_m_align PASS" in simulated.stdout
