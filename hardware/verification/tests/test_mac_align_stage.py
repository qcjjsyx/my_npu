"""Check all 32 physical ALIGN paths and the elastic output against the oracle."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from reference.mac_contract import (
    AccKind, ComputeMode, Entry, MPhase, align,
)

ROOT = Path(__file__).resolve().parents[3]
DOT_N = 16


def _case(rng: random.Random, mode: ComputeMode, group_valid: int,
          directed: int | None = None):
    p0, p1, e0, e1 = [0] * DOT_N, [0] * DOT_N, [0] * DOT_N, [0] * DOT_N
    nz0, nz1 = [False] * DOT_N, [False] * DOT_N
    floating = mode in (ComputeMode.FP19, ComputeMode.FP9)
    for i in range(DOT_N):
        if mode == ComputeMode.FP19:
            p0[i] = rng.randrange(-(1 << 22), 1 << 22) if group_valid & 1 else 0
            e0[i] = rng.randrange(507)
        elif mode == ComputeMode.FP9:
            p0[i] = rng.randrange(-256, 256) if group_valid & 1 else 0
            p1[i] = rng.randrange(-256, 256) if group_valid & 2 else 0
            e0[i], e1[i] = rng.randrange(59), rng.randrange(59)
        else:
            p0[i] = rng.randrange(-(1 << 23), 1 << 23) if group_valid & 1 else 0
            p1[i] = rng.randrange(-(1 << 17), 1 << 17) if (
                group_valid & 2 or mode == ComputeMode.INT16) else 0
        nz0[i], nz1[i] = floating and p0[i] != 0, mode == ComputeMode.FP9 and p1[i] != 0

    if directed == 0:  # Both groups all zero despite valid metadata.
        p0, p1 = [0] * DOT_N, [0] * DOT_N
        nz0, nz1 = [False] * DOT_N, [False] * DOT_N
    elif directed == 1 and floating:  # Negative midpoint: -0.5 -> 0.
        p0, p1 = [0] * DOT_N, [0] * DOT_N
        e0, e1 = [0] * DOT_N, [0] * DOT_N
        p0[0], p0[1] = -1, 1
        e0[1] = 10 if mode == ComputeMode.FP19 else 16
        nz0 = [p != 0 for p in p0]
        if mode == ComputeMode.FP9 and group_valid & 2:
            p1[0], p1[1] = -3, 1  # -1.5 -> -2, independently in group1.
            e1[1] = 16
            nz1 = [p != 0 for p in p1]
    elif directed == 2 and floating:  # d >= W; no distance truncation/wrap.
        p0, p1 = [0] * DOT_N, [0] * DOT_N
        e0, e1 = [0] * DOT_N, [0] * DOT_N
        p0[0], p0[1] = -1, 1
        e0[1] = 506 if mode == ComputeMode.FP19 else 58
        nz0 = [p != 0 for p in p0]
        if mode == ComputeMode.FP9 and group_valid & 2:
            p1[0], p1[1] = -1, 1
            e1[1] = 58
            nz1 = [p != 0 for p in p1]

    if not (group_valid & 1):
        p0, nz0 = [0] * DOT_N, [False] * DOT_N
    if not (group_valid & 2) and mode != ComputeMode.INT16:
        p1, nz1 = [0] * DOT_N, [False] * DOT_N
    entry = Entry(mode, AccKind.TC_FLOAT if floating else AccKind.INT32,
                  (0,) * DOT_N, (0,) * DOT_N, 0, group_valid)
    m = MPhase(0, tuple(p0), tuple(p1), tuple(e0), tuple(e1),
               tuple(nz0), tuple(nz1))
    return m, align(entry, m)


def _run_align_cases(tmp_path: Path, modes, *, simulate=True):
    rng = random.Random(0xA119012)
    cases = []
    for mode in modes:
        valids = (1,) if mode in (ComputeMode.FP19, ComputeMode.INT16) else (1, 2, 3)
        for valid in valids:
            for directed in (0, 1, 2):
                cases.append((mode, valid, directed))
            for _ in range(80):
                cases.append((mode, valid, None))
    vectors = tmp_path / "align_stage.txt"
    with vectors.open("w", encoding="ascii") as out:
        for mode, valid, directed in cases:
            m, result = _case(rng, mode, valid, directed)
            mx = int(mode == ComputeMode.FP9 and rng.randrange(2) == 1 or
                     mode == ComputeMode.INT8 and rng.randrange(2) == 1)
            kind = int(mode in (ComputeMode.FP19, ComputeMode.FP9) or mx)
            bias = rng.randrange(255) if mode in (ComputeMode.FP19, ComputeMode.FP9) else 0
            first = rng.randrange(4) & valid
            phase = rng.randrange(2) if mode == ComputeMode.INT16 else 0
            if phase:
                first = 0
            tags = (rng.randrange(16), rng.randrange(16))
            sa = (rng.randrange(255), rng.randrange(255))
            sb = (rng.randrange(255), rng.randrange(255))
            fields = (int(mode), kind, valid, first, bias, mx, phase,
                      *tags, *sa, *sb, *result.emax,
                      int(result.has_nonzero[0]) | int(result.has_nonzero[1]) << 1,
                      int(result.inexact[0]) | int(result.inexact[1]) << 1)
            out.write(" ".join(f"{x:x}" for x in fields) + "\n")
            for i in range(DOT_N):
                lane = (m.product0[i] & ((1 << 24) - 1),
                        m.product1[i] & ((1 << 18) - 1),
                        m.exp_sum0[i], m.exp_sum1[i],
                        int(m.nz0[i]) | int(m.nz1[i]) << 1,
                        result.lane0[i] & ((1 << 32) - 1),
                        result.lane1[i] & ((1 << 24) - 1))
                out.write(" ".join(f"{x:x}" for x in lane) + "\n")
    image = tmp_path / "align_stage.vvp"
    compiled = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mac_align_stage", "-o", str(image),
         "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/rtl/systemverilog/mac_align_max16.sv",
         "hardware/rtl/systemverilog/mac_align_rne.sv", "hardware/rtl/systemverilog/mac_align_lane0.sv",
         "hardware/rtl/systemverilog/mac_align_lane1.sv", "hardware/rtl/systemverilog/mac_align_stage.sv",
         "hardware/verification/tb/tb_mac_align_stage.sv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    if not simulate:
        return
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"mac_align_stage PASS: {len(cases)} vectors" in simulated.stdout


@pytest.mark.parametrize("mode", list(ComputeMode))
def test_align_stage_against_integer_reference_and_stalls(tmp_path, mode):
    _run_align_cases(tmp_path, (mode,))


@pytest.mark.parametrize("bad_case,marker", [
    (1, "FP19 ALIGN input redundancy"),
    (2, "FP19 ALIGN input redundancy"),
    (3, "invalid ALIGN mode or group_valid"),
])
def test_align_stage_contract_violation_is_detected(tmp_path: Path,
                                                     bad_case: int, marker: str):
    # Reuse a valid first FP19 vector; only the selected field is corrupted.
    _run_align_cases(tmp_path, (ComputeMode.FP19,), simulate=False)
    image = tmp_path / "align_stage.vvp"
    vectors = tmp_path / "align_stage.txt"
    result = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}", f"+BAD={bad_case}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert marker in result.stdout + result.stderr
