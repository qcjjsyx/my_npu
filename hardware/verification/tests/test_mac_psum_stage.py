"""Compare the PSUM RTL stage with the independent integer contract model."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

from reference.mac_contract import (
    AccKind,
    Aligned,
    ComputeMode,
    Entry,
    MPhase,
    partial_sum,
)

ROOT = Path(__file__).resolve().parents[3]
DOT_N = 16
MASK32 = (1 << 32) - 1
MASK24 = (1 << 24) - 1


def _rand_int16(rng: random.Random) -> int:
    value = rng.randrange(-32767, 32768)
    return value


def _int16_partial_terms(rng: random.Random, phase: int, directed: bool):
    lane0: list[int] = []
    lane1: list[int] = []
    for lane in range(DOT_N):
        if directed:
            a = (32767, -32767, 0x1234, 0x7F01)[lane % 4]
            b = (32767, -32767, 0x80FF - 0x10000, 0x01FF)[lane % 4]
        else:
            a = _rand_int16(rng)
            b = _rand_int16(rng)
        a_low = a & 0xFF
        a_high = a >> 8
        b_low = b & 0xFF
        b_high = b >> 8
        if phase == 0:
            lane0.append(a_low * b_low)
            lane1.append(a_high * b_low)
        else:
            lane0.append(a_low * b_high)
            lane1.append(a_high * b_high)
    return lane0, lane1


def _make_case(
    rng: random.Random,
    mode: ComputeMode,
    group_valid: int,
    *,
    phase: int = 0,
    directed: str | None = None,
    serial: int,
):
    lane0 = [0] * DOT_N
    lane1 = [0] * DOT_N
    mx_enable = False

    if mode == ComputeMode.FP19:
        max_term = 4_190_209 << 9
        if directed == "negative_low_growth":
            lane0 = [-1] * DOT_N
        elif directed == "limb_edges":
            lane0 = [-max_term, max_term, -1, 1] + [0] * 12
        else:
            lane0 = [rng.randrange(-max_term, max_term + 1) for _ in range(DOT_N)]
        kind, bias = AccKind.TC_FLOAT, rng.choice((30, 254))
        emax0, emax1 = rng.randrange(509), 0
    elif mode == ComputeMode.FP9:
        max_term = 225 << 15
        if directed == "opposite_groups":
            lane0 = [max_term] * DOT_N if group_valid & 1 else [0] * DOT_N
            lane1 = [-max_term] * DOT_N if group_valid & 2 else [0] * DOT_N
        elif directed == "one_group":
            lane0 = [-(225 << 15)] * DOT_N if group_valid & 1 else [0] * DOT_N
            lane1 = [max_term] * DOT_N if group_valid & 2 else [0] * DOT_N
        else:
            lane0 = [rng.randrange(-max_term, max_term + 1)
                     if group_valid & 1 else 0 for _ in range(DOT_N)]
            lane1 = [rng.randrange(-max_term, max_term + 1)
                     if group_valid & 2 else 0 for _ in range(DOT_N)]
        mx_enable = bool(rng.randrange(2))
        kind, bias = AccKind.TC_FLOAT, rng.choice((14, 22, 30))
        emax0, emax1 = rng.randrange(61), rng.randrange(61)
    elif mode == ComputeMode.INT16:
        lane0, lane1 = _int16_partial_terms(rng, phase, directed == "phase_edges")
        kind, bias = AccKind.INT32, 0
        emax0 = emax1 = 0
    elif mode == ComputeMode.INT8:
        mx_enable = bool(rng.randrange(2))
        lane0 = [rng.randrange(-16129, 16130) if group_valid & 1 else 0
                 for _ in range(DOT_N)]
        lane1 = [rng.randrange(-16129, 16130) if group_valid & 2 else 0
                 for _ in range(DOT_N)]
        kind, bias = (AccKind.TC_FLOAT if mx_enable else AccKind.INT32), 0
        emax0 = emax1 = 0
    else:
        lane0 = [rng.randrange(-49, 50) if group_valid & 1 else 0
                 for _ in range(DOT_N)]
        lane1 = [rng.randrange(-49, 50) if group_valid & 2 else 0
                 for _ in range(DOT_N)]
        kind, bias = AccKind.INT32, 0
        emax0 = emax1 = 0

    if mode in (ComputeMode.FP19, ComputeMode.INT16):
        group_valid = 1
    if mode == ComputeMode.FP19:
        lane1 = [0] * DOT_N

    entry = Entry(mode, kind, (0,) * DOT_N, (0,) * DOT_N, bias, group_valid)
    m = MPhase(
        phase,
        (0,) * DOT_N,
        (0,) * DOT_N,
        (0,) * DOT_N,
        (0,) * DOT_N,
        (False,) * DOT_N,
        (False,) * DOT_N,
    )
    has_nz = [bool(rng.randrange(2)) if group_valid & (1 << g) else False
              for g in range(2)]
    inexact = [bool(rng.randrange(2)) if group_valid & (1 << g) else False
               for g in range(2)]
    aligned = Aligned(tuple(lane0), tuple(lane1), (emax0, emax1),
                      tuple(has_nz), tuple(inexact))
    result = partial_sum(entry, m, aligned)

    first = rng.randrange(4) & group_valid
    if mode == ComputeMode.INT16 and phase == 1:
        first = 0
    scale_a = [rng.randrange(255) if mx_enable else 127 for _ in range(2)]
    scale_b = [rng.randrange(255) if mx_enable else 127 for _ in range(2)]
    if mode in (ComputeMode.FP19, ComputeMode.INT16):
        scale_a = [127, 127]
        scale_b = [127, 127]

    return {
        "mode": int(mode),
        "kind": int(kind),
        "group_valid": group_valid,
        "first": first,
        "bias": bias,
        "mx": int(mx_enable),
        "phase": phase,
        "tag0": (serial * 3 + 1) & 0xF,
        "tag1": (serial * 5 + 2) & 0xF,
        "scale_a0": scale_a[0],
        "scale_a1": scale_a[1],
        "scale_b0": scale_b[0],
        "scale_b1": scale_b[1],
        "emax0": emax0,
        "emax1": emax1,
        "group_nz": int(has_nz[0]) | (int(has_nz[1]) << 1),
        "inexact": int(inexact[0]) | (int(inexact[1]) << 1),
        "expected": result.packed56,
        "lane0": lane0,
        "lane1": lane1,
    }


def _write_vectors(path: Path, cases: list[dict[str, object]]) -> None:
    with path.open("w", encoding="ascii") as out:
        for case in cases:
            header = (
                "mode kind group_valid first bias mx phase tag0 tag1 "
                "scale_a0 scale_a1 scale_b0 scale_b1 emax0 emax1 "
                "group_nz inexact expected"
            ).split()
            out.write(" ".join(f"{int(case[key]):x}" for key in header) + "\n")
            for lane0, lane1 in zip(case["lane0"], case["lane1"]):
                out.write(f"{int(lane0) & MASK32:09x} {int(lane1) & MASK24:06x}\n")


def test_psum_stage_against_integer_reference_and_backpressure(tmp_path: Path):
    rng = random.Random(0x5A7E_2026)
    cases: list[dict[str, object]] = []
    serial = 0

    for directed in ("negative_low_growth", "limb_edges"):
        cases.append(_make_case(rng, ComputeMode.FP19, 1,
                                directed=directed, serial=serial))
        serial += 1
    for _ in range(32):
        cases.append(_make_case(rng, ComputeMode.FP19, 1, serial=serial))
        serial += 1

    for group_valid in (1, 2, 3):
        for directed in ("opposite_groups", "one_group"):
            cases.append(_make_case(rng, ComputeMode.FP9, group_valid,
                                    directed=directed, serial=serial))
            serial += 1
        for _ in range(20):
            cases.append(_make_case(rng, ComputeMode.FP9, group_valid,
                                    serial=serial))
            serial += 1

    for phase in (0, 1):
        cases.append(_make_case(rng, ComputeMode.INT16, 1, phase=phase,
                                directed="phase_edges", serial=serial))
        serial += 1
        for _ in range(24):
            cases.append(_make_case(rng, ComputeMode.INT16, 1, phase=phase,
                                    serial=serial))
            serial += 1

    for mode in (ComputeMode.INT8, ComputeMode.INT4):
        for group_valid in (1, 2, 3):
            for _ in range(20):
                cases.append(_make_case(rng, mode, group_valid, serial=serial))
                serial += 1

    vectors = tmp_path / "psum_vectors.txt"
    _write_vectors(vectors, cases)
    image = tmp_path / "mac_psum_stage.vvp"
    compiled = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mac_psum_stage", "-o", str(image),
         "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/verification/tb/dwbb_sim_models.sv",
         "hardware/rtl/systemverilog/mac_psum_stage.sv", "hardware/verification/tb/tb_mac_psum_stage.sv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"mac_psum_stage PASS: {len(cases)} vectors" in simulated.stdout
