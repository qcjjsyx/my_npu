"""Exercise the real ALIGN-to-PSUM ready/valid boundary."""

from __future__ import annotations

import pytest
import random
import subprocess
from pathlib import Path

from reference.mac_contract import (
    AccKind,
    ComputeMode,
    Entry,
    MPhase,
    align,
    partial_sum,
)

ROOT = Path(__file__).resolve().parents[3]
DOT_N = 16


def _build_case(rng: random.Random, mode: ComputeMode,
                group_valid: int, phase: int, serial: int) -> dict[str, object]:
    p0 = [0] * DOT_N
    p1 = [0] * DOT_N
    e0 = [0] * DOT_N
    e1 = [0] * DOT_N
    nz0 = [False] * DOT_N
    nz1 = [False] * DOT_N
    mx_enable = False

    if mode == ComputeMode.FP19:
        group_valid = 1
        sig_a = [rng.randrange(-2047, 2048) for _ in range(DOT_N)]
        sig_b = [rng.randrange(-2047, 2048) for _ in range(DOT_N)]
        p0 = [a * b for a, b in zip(sig_a, sig_b)]
        e0 = [rng.randrange(509) for _ in range(DOT_N)]
        nz0 = [value != 0 for value in p0]
        bias = rng.choice((30, 254))
        kind = AccKind.TC_FLOAT
    elif mode == ComputeMode.FP9:
        sig_a0 = [rng.randrange(-15, 16) for _ in range(DOT_N)]
        sig_b0 = [rng.randrange(-15, 16) for _ in range(DOT_N)]
        sig_a1 = [rng.randrange(-15, 16) for _ in range(DOT_N)]
        sig_b1 = [rng.randrange(-15, 16) for _ in range(DOT_N)]
        p0 = [a * b if group_valid & 1 else 0
              for a, b in zip(sig_a0, sig_b0)]
        p1 = [a * b if group_valid & 2 else 0
              for a, b in zip(sig_a1, sig_b1)]
        e0 = [rng.randrange(61) for _ in range(DOT_N)]
        e1 = [rng.randrange(61) for _ in range(DOT_N)]
        nz0 = [bool(group_valid & 1) and value != 0 for value in p0]
        nz1 = [bool(group_valid & 2) and value != 0 for value in p1]
        mx_enable = bool(rng.randrange(2))
        bias = rng.choice((14, 22, 30))
        kind = AccKind.TC_FLOAT
    elif mode == ComputeMode.INT16:
        group_valid = 1
        for lane in range(DOT_N):
            a = rng.randrange(-32767, 32768)
            b = rng.randrange(-32767, 32768)
            a_low, a_high = a & 0xFF, a >> 8
            b_low, b_high = b & 0xFF, b >> 8
            if phase == 0:
                p0[lane] = a_low * b_low
                p1[lane] = a_high * b_low
            else:
                p0[lane] = a_low * b_high
                p1[lane] = a_high * b_high
        bias = 0
        kind = AccKind.INT32
    elif mode == ComputeMode.INT8:
        mx_enable = bool(rng.randrange(2))
        for lane in range(DOT_N):
            a0, b0 = rng.randrange(-127, 128), rng.randrange(-127, 128)
            a1, b1 = rng.randrange(-127, 128), rng.randrange(-127, 128)
            p0[lane] = a0 * b0 if group_valid & 1 else 0
            p1[lane] = a1 * b1 if group_valid & 2 else 0
        bias = 0
        kind = AccKind.TC_FLOAT if mx_enable else AccKind.INT32
    else:
        for lane in range(DOT_N):
            a0, b0 = rng.randrange(-7, 8), rng.randrange(-7, 8)
            a1, b1 = rng.randrange(-7, 8), rng.randrange(-7, 8)
            p0[lane] = a0 * b0 if group_valid & 1 else 0
            p1[lane] = a1 * b1 if group_valid & 2 else 0
        bias = 0
        kind = AccKind.INT32

    entry = Entry(mode, kind, (0,) * DOT_N, (0,) * DOT_N, bias, group_valid)
    m = MPhase(phase, tuple(p0), tuple(p1), tuple(e0), tuple(e1),
               tuple(nz0), tuple(nz1))
    aligned = align(entry, m)
    psum = partial_sum(entry, m, aligned)

    first = rng.randrange(4) & group_valid
    if mode == ComputeMode.INT16 and phase == 1:
        first = 0
    if mx_enable:
        scale_a = (rng.randrange(255), rng.randrange(255))
        scale_b = (rng.randrange(255), rng.randrange(255))
    else:
        scale_a = scale_b = (127, 127)
    if mode in (ComputeMode.FP19, ComputeMode.INT16):
        scale_a = scale_b = (127, 127)

    return {
        "mode": int(mode), "kind": int(kind), "group_valid": group_valid,
        "first": first, "bias": bias, "mx": int(mx_enable), "phase": phase,
        "tag0": (serial * 3 + 1) & 0xF, "tag1": (serial * 5 + 2) & 0xF,
        "scale_a0": scale_a[0], "scale_a1": scale_a[1],
        "scale_b0": scale_b[0], "scale_b1": scale_b[1],
        "emax0": aligned.emax[0], "emax1": aligned.emax[1],
        "group_nz": int(aligned.has_nonzero[0]) |
                    (int(aligned.has_nonzero[1]) << 1),
        "inexact": int(aligned.inexact[0]) | (int(aligned.inexact[1]) << 1),
        "expected": psum.packed56, "phase_data": m,
    }


def _write_records(path: Path, cases: list[dict[str, object]]) -> None:
    with path.open("w", encoding="ascii") as out:
        header_keys = (
            "mode kind group_valid first bias mx phase tag0 tag1 "
            "scale_a0 scale_a1 scale_b0 scale_b1 emax0 emax1 "
            "group_nz inexact expected"
        ).split()
        for case in cases:
            out.write(" ".join(f"{int(case[key]):x}" for key in header_keys) + "\n")
            m = case["phase_data"]
            for lane in range(DOT_N):
                nz = int(m.nz0[lane]) | (int(m.nz1[lane]) << 1)
                out.write(
                    f"{m.product0[lane] & 0xFFFFFF:06x} "
                    f"{m.product1[lane] & 0x3FFFF:05x} "
                    f"{m.exp_sum0[lane]:03x} {m.exp_sum1[lane]:02x} {nz:x}\n"
                )


@pytest.mark.parametrize("fp19", [True, False])
def test_real_align_to_psum_level_handshake(tmp_path: Path, fp19):
    rng = random.Random(0xA115_2026)
    cases: list[dict[str, object]] = []
    serial = 0
    modes = (ComputeMode.FP19,) if fp19 else (ComputeMode.FP9, ComputeMode.INT16, ComputeMode.INT8, ComputeMode.INT4)
    for mode in modes:
        valids = (1,) if mode in (ComputeMode.FP19, ComputeMode.INT16) else (1, 2, 3)
        phases = (0, 1) if mode == ComputeMode.INT16 else (0,)
        for group_valid in valids:
            for phase in phases:
                for _ in range(3):
                    cases.append(_build_case(rng, mode, group_valid, phase, serial))
                    serial += 1

    vectors = tmp_path / "align_psum_vectors.txt"
    _write_records(vectors, cases)
    image = tmp_path / "mac_align_psum.vvp"
    sources = (
        "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/verification/tb/dwbb_sim_models.sv",
        "hardware/rtl/systemverilog/mac_align_max16.sv", "hardware/rtl/systemverilog/mac_align_rne.sv",
        "hardware/rtl/systemverilog/mac_align_lane0.sv", "hardware/rtl/systemverilog/mac_align_lane1.sv",
        "hardware/rtl/systemverilog/mac_align_stage.sv", "hardware/rtl/systemverilog/mac_psum_stage.sv",
        "hardware/verification/tb/tb_mac_align_psum.sv",
    )
    compiled = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mac_align_psum", "-o", str(image),
         *sources], cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"mac_align_psum PASS: {len(cases)} vectors" in simulated.stdout
