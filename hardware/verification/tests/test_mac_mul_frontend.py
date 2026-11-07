"""Compare a physical M frontend with the independent transaction oracle."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

from reference.mac_contract import (
    DOT_N,
    SourceFormat,
    Transaction,
    enter,
    multiply,
)


ROOT = Path(__file__).resolve().parents[3]


def _raw_for(fmt: SourceFormat, rng: random.Random) -> int:
    if fmt == SourceFormat.INT16:
        return rng.randint(-32767, 32767) & 0xFFFF
    if fmt == SourceFormat.INT8:
        return ((rng.randint(-127, 127) & 255) << 8) | (rng.randint(-127, 127) & 255)
    if fmt == SourceFormat.INT4:
        return ((rng.randint(-7, 7) & 15) << 4) | (rng.randint(-7, 7) & 15)
    return rng.randrange(1 << 16)


def _write_vectors(path: Path) -> int:
    rng = random.Random(0x5A7E16)
    count = 0
    pairs = (
        (SourceFormat.FP16, SourceFormat.FP16),
        (SourceFormat.BF16, SourceFormat.BF16),
        (SourceFormat.E4M3, SourceFormat.E4M3),
        (SourceFormat.E5M2, SourceFormat.E5M2),
        (SourceFormat.E4M3, SourceFormat.E5M2),
        (SourceFormat.E5M2, SourceFormat.E4M3),
        (SourceFormat.INT16, SourceFormat.INT16),
        (SourceFormat.INT8, SourceFormat.INT8),
        (SourceFormat.INT4, SourceFormat.INT4),
    )
    directed = {
        SourceFormat.FP16: (0, 1, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x3C00, 0xFBFF),
        SourceFormat.BF16: (0, 1, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x3F80, 0xFF7F),
        SourceFormat.E4M3: (0, 1, 0x7F, 0xFF, 0x7E, 0xFE, 0x38, 0xB8),
        SourceFormat.E5M2: (0, 1, 0x7C, 0xFC, 0x7D, 0xFD, 0x3C, 0xBC),
        SourceFormat.INT16: (0, 1, 255, 256, 32767, 0x8001, 0xFFFF),
        SourceFormat.INT8: (0, 1, 0x7F, 0x81, 0x7F7F, 0x8181),
        SourceFormat.INT4: (0, 1, 7, 9, 0x77, 0x99),
    }
    with path.open("w", encoding="ascii") as out:
        for fmt_a, fmt_b in pairs:
            for index in range(1200):
                if index < 16:
                    raw_a = directed[fmt_a][index % len(directed[fmt_a])]
                    raw_b = directed[fmt_b][(index * 3) % len(directed[fmt_b])]
                    if fmt_a in (SourceFormat.E4M3, SourceFormat.E5M2):
                        raw_a |= (raw_a & 255) << 8
                        raw_b |= (raw_b & 255) << 8
                else:
                    raw_a, raw_b = _raw_for(fmt_a, rng), _raw_for(fmt_b, rng)
                valid = 1 if fmt_a in (SourceFormat.FP16, SourceFormat.BF16, SourceFormat.INT16) else rng.choice((1, 2, 3))
                tx = Transaction(
                    raw_a=(raw_a,) + (0,) * (DOT_N - 1),
                    raw_b=(raw_b,) + (0,) * (DOT_N - 1),
                    fmt_a=fmt_a, fmt_b=fmt_b, group_valid=valid,
                    acc_first=valid, tag=(0, 0),
                )
                entry = enter(tx, tag_w=1)
                for phase in ((0, 1) if fmt_a == SourceFormat.INT16 else (0,)):
                    result = multiply(entry, phase)
                    active = 3 if fmt_a == SourceFormat.INT16 else (1 if fmt_a in (SourceFormat.FP16, SourceFormat.BF16) else valid)
                    fields = (
                        int(entry.mode), valid, phase,
                        entry.payload_a[0], entry.payload_b[0],
                        result.product0[0] & ((1 << 24) - 1),
                        result.product1[0] & ((1 << 18) - 1),
                        result.exp_sum0[0], result.exp_sum1[0],
                        int(result.nz1[0]) << 1 | int(result.nz0[0]), active,
                    )
                    out.write(" ".join(f"{value:x}" for value in fields) + "\n")
                    count += 1
    return count


def test_mul_frontend_against_integer_reference(tmp_path: Path):
    vectors = tmp_path / "mul_vectors.txt"
    image = tmp_path / "mul_frontend.vvp"
    count = _write_vectors(vectors)
    compiled = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mac_mul_frontend", "-o", str(image),
         "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/rtl/systemverilog/mac_mul_frontend.sv",
         "hardware/verification/tb/tb_mac_mul_frontend.sv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"mac_mul_frontend PASS: {count} vectors" in simulated.stdout
