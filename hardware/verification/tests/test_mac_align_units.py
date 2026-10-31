"""Independent integer checks for the four-level MAX and G/R/S RNE units."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

from reference.mac_contract import round_div_pow2

ROOT = Path(__file__).resolve().parents[3]


def test_align_max_and_rne(tmp_path: Path):
    rng = random.Random(0xA11C0DE)
    vectors = tmp_path / "align_units.txt"
    cases = []
    for kind in (0, 1, 2):
        width = 32 if kind != 2 else 24
        limit = 24 if kind else 32
        for d in sorted({0, 1, 2, limit - 1, limit, limit + 1, 58, 506}):
            for y in (-((1 << (limit - 1)) - 1), -3, -2, -1, 0, 1, 2, 3,
                      (1 << (limit - 1)) - 1):
                cases.append((kind, y, d))
        for _ in range(1800):
            y = rng.randrange(-(1 << (limit - 1)), 1 << (limit - 1))
            d = rng.choice((rng.randrange(limit + 2), 58, 506))
            cases.append((kind, y, d))
    with vectors.open("w", encoding="ascii") as out:
        for kind, y, d in cases:
            width = 24 if kind == 2 else 32
            limit = 24 if kind else 32
            expected, lost = (0, y != 0) if d >= limit else round_div_pow2(y, d)
            nz = rng.randrange(1 << 16)
            exps = [rng.randrange(507) for _ in range(16)]
            maximum = max((e for i, e in enumerate(exps) if nz & (1 << i)), default=0)
            out.write(" ".join(f"{v:x}" for v in
                               (kind, y & ((1 << width) - 1), d, int(kind != 0),
                                expected & ((1 << width) - 1), int(lost), nz)) + "\n")
            out.write(" ".join(f"{e:x}" for e in exps) + "\n")
            out.write(f"{maximum:x} {int(bool(nz)):x}\n")
    image = tmp_path / "align_units.vvp"
    compiled = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mac_align_units", "-o", str(image),
         "hardware/rtl/systemverilog/mac_align_max16.sv", "hardware/rtl/systemverilog/mac_align_rne.sv",
         "hardware/verification/tb/tb_mac_align_units.sv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"mac_align_units PASS: {len(cases)} vectors" in simulated.stdout
