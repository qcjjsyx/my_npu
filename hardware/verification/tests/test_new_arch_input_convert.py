"""Verilog-2001 converter co-simulation against the new-architecture model."""

import subprocess
from pathlib import Path

import pytest

from reference.new_arch.input_convert import SourceFormat as Fmt, convert_raw


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def converter_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    image = tmp_path_factory.mktemp("new_arch_input_convert") / "convert.vvp"
    compiled = subprocess.run(
        ["iverilog", "-g2001", "-s", "tb_new_arch_mac_input_convert", "-o", str(image),
         "-f", "hardware/filelists/new_arch_convert.f",
         "hardware/verification/tb/tb_new_arch_mac_input_convert.v"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return image


def _run_simulation(image: Path, vectors: Path, cycles: int) -> None:
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"PASS: {cycles} cycles" in simulated.stdout.splitlines(), (
        simulated.stdout + simulated.stderr
    )


@pytest.mark.parametrize("fmt", list(Fmt), ids=[fmt.name for fmt in Fmt])
def test_all_raw_words(converter_image: Path, tmp_path: Path, fmt: Fmt) -> None:
    vectors = tmp_path / "raw_words.txt"
    count = 1 << 16
    with vectors.open("w", encoding="ascii") as out:
        out.write("0 0 0 0000 0 000000\n")
        # Integer minimum-negative raw codes also pass through. They remain
        # outside the MAC input domain specified by IP_DESIGN §1.3.
        for raw in range(count):
            fields = convert_raw(raw, fmt)
            out.write(f"1 1 {fmt:x} {raw:04x} 1 {fields:06x}\n")
    _run_simulation(converter_image, vectors, count + 1)


def test_directed_fields_and_timing(converter_image: Path, tmp_path: Path) -> None:
    # Each row holds rst_n, input valid, format, raw, and the fields after
    # the sampling edge. The FP8 words list g1 and g0 as separate fields.
    cycles = [
        (0, 1, Fmt.INT16, 0x1234, 0),
        (1, 1, Fmt.FP4, 0xF391, 0b10100_00011_11111_00001),
        (1, 1, Fmt.INT4, 0xF792, 0b11111_00111_11001_00010),
        (1, 0, Fmt.FP16, 0x7BFF, 0b11111_00111_11001_00010),
        (1, 0, Fmt.INT16, 0x4321, 0b11111_00111_11001_00010),
        (1, 1, Fmt.E4M3FN, 0xFE7E, (0x590 << 11) | 0x570),  # exp=15 finite
        (1, 1, Fmt.E5M2, 0xFC7C, (0x7C8 << 11) | 0x738),  # ±Inf
        (1, 1, Fmt.BF16, 0x7F80, 0x1FCFF),
        (1, 1, Fmt.FP16, 0xFC00, 0x3E004),
        (1, 1, Fmt.E4M3FN, 0x8701, (0x2F2 << 11) | 0x202),  # ±subnormal
        (1, 1, Fmt.E5M2, 0x8301, (0x0F4 << 11) | 0x004),
        (1, 1, Fmt.BF16, 0x807F, 0x00381),
        (1, 1, Fmt.FP16, 0x0001, 0x00002),
        (1, 1, Fmt.BF16, 0x0001, 0x00201),
        (1, 1, Fmt.FP16, 0x83FF, 0x07802),
        (1, 1, Fmt.E4M3FN, 0x7F5F, 0x478),  # NaN and largest folded magnitude
        (1, 1, Fmt.E5M2, 0xFD6F, 0x670),
        (1, 1, Fmt.BF16, 0xFFC0, 0),
        (1, 1, Fmt.FP16, 0x7C01, 0),
        (1, 1, Fmt.FP16, 0x6FFF, 0x33FF8),
        (1, 1, Fmt.INT8, 0x817F, 0x817F),
        (0, 1, Fmt.INT16, 0xFEDC, 0),  # Clear the previous valid word before the edge.
        (0, 1, Fmt.FP16, 0x7BFF, 0),  # Clock and valid input cannot load while reset is low.
        (1, 0, Fmt.E5M2, 0xFC7C, 0),
        (1, 1, Fmt.INT16, 0xFEDC, 0xFEDC),
        (1, 1, Fmt.FP4, 0x8080, 0),  # Signed zero encodings
        (1, 1, Fmt.E4M3FN, 0x8000, 0),
        (1, 1, Fmt.E5M2, 0x0080, 0),
        (1, 1, Fmt.BF16, 0x8000, 0),
        (1, 1, Fmt.FP16, 0x8000, 0),
    ]
    vectors = tmp_path / "directed.txt"
    with vectors.open("w", encoding="ascii") as out:
        for rst_n, valid, fmt, raw, expected in cycles:
            if rst_n and valid:
                assert convert_raw(raw, fmt) == expected, (fmt.name, hex(raw))
            out.write(f"{rst_n} {valid} {fmt:x} {raw:04x} {rst_n & valid} {expected:06x}\n")
    _run_simulation(converter_image, vectors, len(cycles))
