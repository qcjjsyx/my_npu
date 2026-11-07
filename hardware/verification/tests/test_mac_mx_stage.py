"""Check MX against the independent integer model across a real PSUM boundary."""

from __future__ import annotations

import pytest
import random
import subprocess
from pathlib import Path

from reference.mac_contract import (
    AccKind,
    Aligned,
    ComputeMode,
    Entry,
    MPhase,
    SourceFormat,
    Transaction,
    mx,
    partial_sum,
)

ROOT = Path(__file__).resolve().parents[3]
DOT_N = 16


def _case(rng: random.Random, mode: ComputeMode, group_valid: int,
          phase: int, serial: int, *, forced_mx: bool | None = None):
    lane0 = [0] * DOT_N
    lane1 = [0] * DOT_N
    if mode == ComputeMode.FP19:
        lane0 = [rng.randrange(-(1 << 31), 1 << 31) for _ in range(DOT_N)]
        group_valid, bias = 1, rng.choice((30, 254))
        emax0, emax1 = rng.randrange(max(0,bias-131), min(509,bias+123)), 0
        mx_enable = False
        fmt_a = fmt_b = SourceFormat.FP16
    elif mode == ComputeMode.FP9:
        limit = 225 << 15
        lane0 = [rng.randrange(-limit, limit + 1) if group_valid & 1 else 0
                 for _ in range(DOT_N)]
        lane1 = [rng.randrange(-limit, limit + 1) if group_valid & 2 else 0
                 for _ in range(DOT_N)]
        bias = rng.choice((14, 22, 30))
        emax0, emax1 = rng.randrange(61), rng.randrange(61)
        mx_enable = bool(rng.randrange(2))
        fmt_a = fmt_b = SourceFormat.E4M3
    elif mode == ComputeMode.INT16:
        for i in range(DOT_N):
            a, b = rng.randrange(-32767, 32768), rng.randrange(-32767, 32768)
            a_low, a_high = a & 0xFF, a >> 8
            b_low, b_high = b & 0xFF, b >> 8
            if phase == 0:
                lane0[i], lane1[i] = a_low * b_low, a_high * b_low
            else:
                lane0[i], lane1[i] = a_low * b_high, a_high * b_high
        group_valid, bias = 1, 0
        emax0 = emax1 = 0
        mx_enable = False
        fmt_a = fmt_b = SourceFormat.INT16
    elif mode == ComputeMode.INT8:
        lane0 = [rng.randrange(-16129, 16130) if group_valid & 1 else 0
                 for _ in range(DOT_N)]
        lane1 = [rng.randrange(-16129, 16130) if group_valid & 2 else 0
                 for _ in range(DOT_N)]
        bias = 0
        emax0 = emax1 = 0
        mx_enable = bool(rng.randrange(2))
        fmt_a = fmt_b = SourceFormat.INT8
    else:
        lane0 = [rng.randrange(-49, 50) if group_valid & 1 else 0
                 for _ in range(DOT_N)]
        lane1 = [rng.randrange(-49, 50) if group_valid & 2 else 0
                 for _ in range(DOT_N)]
        bias = 0
        emax0 = emax1 = 0
        mx_enable = False
        fmt_a = fmt_b = SourceFormat.INT4

    if forced_mx is not None:
        mx_enable = forced_mx
    kind = (AccKind.TC_FLOAT if mode in (ComputeMode.FP19, ComputeMode.FP9) or
            (mode == ComputeMode.INT8 and mx_enable) else AccKind.INT32)
    acc_first = rng.randrange(4) & group_valid
    if mode == ComputeMode.INT16 and phase:
        acc_first = 0
    # Deliberately use different group scales.  Invalid/no-scale fields are
    # poisoned with E8M0's reserved code to check that MX ignores them.
    if mx_enable:
        scale_a = tuple(rng.randrange(255) if group_valid & (1 << g) else 255
                        for g in range(2))
        scale_b = tuple(rng.randrange(255) if group_valid & (1 << g) else 255
                        for g in range(2))
    else:
        scale_a = scale_b = (255, 255)
    if mx_enable:
        # Exercise only the currently defined encoded MX input domain.
        # Out-of-domain E has a separate explicit contract-rejection test.
        sa, sb = list(scale_a), list(scale_b)
        for g in range(2):
            if group_valid & (1 << g):
                base = (emax0 if g == 0 else emax1)-bias+132 if mode == ComputeMode.FP9 else 132
                while not 1 <= base + sa[g] + sb[g] - 254 <= 254:
                    sa[g], sb[g] = rng.randrange(255), rng.randrange(255)
        scale_a, scale_b = tuple(sa), tuple(sb)
    group_nz = rng.randrange(4) & group_valid
    inexact = rng.randrange(4) & group_valid

    entry = Entry(mode, kind, (0,) * DOT_N, (0,) * DOT_N, bias, group_valid)
    m = MPhase(phase, (0,) * DOT_N, (0,) * DOT_N, (0,) * DOT_N,
               (0,) * DOT_N, (False,) * DOT_N, (False,) * DOT_N)
    aligned = Aligned(tuple(lane0), tuple(lane1), (emax0, emax1),
                      tuple(bool(group_nz & (1 << g)) for g in range(2)),
                      tuple(bool(inexact & (1 << g)) for g in range(2)))
    psum = partial_sum(entry, m, aligned)
    tx = Transaction((0,) * DOT_N, (0,) * DOT_N, fmt_a, fmt_b, group_valid,
                     acc_first, ((serial * 3 + 1) & 0xF,
                                 (serial * 5 + 2) & 0xF), mx_enable,
                     scale_a, scale_b)
    result = mx(entry, tx, m, aligned, psum)
    return {
        "mode": int(mode), "kind": int(kind), "group_valid": group_valid,
        "first": acc_first, "bias": bias, "mx": int(mx_enable), "phase": phase,
        "tag0": tx.tag[0], "tag1": tx.tag[1],
        "scale_a0": scale_a[0], "scale_a1": scale_a[1],
        "scale_b0": scale_b[0], "scale_b1": scale_b[1],
        "emax0": emax0, "emax1": emax1, "group_nz": group_nz,
        "inexact": inexact, "psum": psum.packed56,
        "c0": result.contribution[0], "c1": result.contribution[1],
        "exp0": result.exp[0] if result.contribution[0] else 0,
        "exp1": result.exp[1] if result.contribution[1] else 0,
        "lane0": lane0, "lane1": lane1,
    }


def _write_vectors(path: Path, cases: list[dict[str, object]]) -> None:
    fields = ("mode kind group_valid first bias mx phase tag0 tag1 scale_a0 "
              "scale_a1 scale_b0 scale_b1 emax0 emax1 group_nz inexact psum "
              "c0 c1 exp0 exp1").split()
    with path.open("w", encoding="ascii") as out:
        for case in cases:
            out.write(" ".join(f"{int(case[key]) & ((1 << 64) - 1):x}"
                               for key in fields) + "\n")
            for a, b in zip(case["lane0"], case["lane1"]):
                out.write(f"{int(a) & ((1 << 32) - 1):09x} "
                          f"{int(b) & ((1 << 24) - 1):06x}\n")


@pytest.mark.parametrize("floating", [True, False])
def test_mx_stage_against_reference_with_psum_handshake(tmp_path: Path, floating):
    rng = random.Random(0x4D58_2026)
    cases: list[dict[str, object]] = []
    serial = 0
    for _ in range(24):
        cases.append(_case(rng, ComputeMode.FP19, 1, 0, serial))
        serial += 1
    for group_valid in (1, 2, 3):
        for enabled in (False, True):
            for _ in range(16):
                cases.append(_case(rng, ComputeMode.FP9, group_valid, 0,
                                   serial, forced_mx=enabled))
                serial += 1
    for phase in (0, 1):
        for _ in range(20):
            cases.append(_case(rng, ComputeMode.INT16, 1, phase, serial))
            serial += 1
    for group_valid in (1, 2, 3):
        for enabled in (False, True):
            for _ in range(16):
                cases.append(_case(rng, ComputeMode.INT8, group_valid, 0,
                                   serial, forced_mx=enabled))
                serial += 1
        for _ in range(12):
            cases.append(_case(rng, ComputeMode.INT4, group_valid, 0, serial))
            serial += 1

    # Directed Q2.34 interface invariants through the real PSUM->MX pair.
    directed = _case(rng, ComputeMode.FP9, 3, 0, serial, forced_mx=True)
    directed.update(bias=14, emax0=32, emax1=19, group_nz=3,
                    scale_a0=127, scale_b0=127, scale_a1=129, scale_b1=126,
                    lane0=[1<<21]+[0]*15, lane1=[-(1<<22)]+[0]*15,
                    psum=((-(1<<22) & ((1<<28)-1))<<28)|(1<<21),
                    c0=1<<29, c1=-(1<<30), exp0=150, exp1=138)
    cases.append(directed)
    directed = _case(rng, ComputeMode.INT8, 3, 0, serial+1, forced_mx=True)
    directed.update(scale_a0=129, scale_b0=126, scale_a1=124, scale_b1=128,
                    lane0=[16129]*16, lane1=[-16129]*16,
                    psum=((-258064 & ((1<<28)-1))<<28)|258064,
                    c0=258064<<17, c1=-258064<<17, exp0=133, exp1=130)
    cases.append(directed)

    cases = [case for case in cases if bool(case["kind"]) == floating]
    vectors = tmp_path / "mx_vectors.txt"
    _write_vectors(vectors, cases)
    image = tmp_path / "mac_mx_stage.vvp"
    sources = (
        "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/verification/tb/dwbb_sim_models.sv",
        "hardware/rtl/systemverilog/mac_psum_stage.sv", "hardware/rtl/systemverilog/mac_mx_stage.sv",
        "hardware/verification/tb/tb_mac_mx_stage.sv",
    )
    compiled = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mac_mx_stage", "-o", str(image),
         *sources], cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"mac_mx_stage PASS: {len(cases)} vectors" in simulated.stdout

    # E8M0=255 has no defined recovery behavior; verify the valid-fire
    # simulation contract catches it instead of silently assigning a value.
    invalid = _case(rng, ComputeMode.FP9, 1, 0, serial, forced_mx=True)
    invalid["scale_a0"] = 255
    invalid_vectors = tmp_path / "mx_invalid_scale_vectors.txt"
    _write_vectors(invalid_vectors, [invalid])
    rejected = subprocess.run(
        ["vvp", str(image), f"+VEC={invalid_vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert "MX active group requires finite E8M0 scales" in (
        rejected.stdout + rejected.stderr
    )

    # The explicitly deferred MX E domain must fail before an encoded wrap
    # can be consumed; this assertion does not define a recovery value.
    out_of_domain = _case(rng, ComputeMode.INT8, 1, 0, serial+2, forced_mx=True)
    out_of_domain["scale_a0"] = out_of_domain["scale_b0"] = 254
    domain_vectors = tmp_path / "mx_deferred_vectors.txt"
    _write_vectors(domain_vectors, [out_of_domain])
    rejected = subprocess.run(
        ["vvp", str(image), f"+VEC={domain_vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert "MX input exponent boundary deferred group0" in rejected.stdout + rejected.stderr
