"""Q2.34 ACC RTL versus the independent arbitrary-precision oracle."""

from __future__ import annotations

import pytest
import random
import subprocess
from pathlib import Path

from reference.mac_contract import (
    FloatAcc36State,
    saturate_int32,
    update_float_acc36,
)

ROOT = Path(__file__).resolve().parents[3]
MASK36 = (1 << 36) - 1

MODE_FP19 = 0
MODE_FP9 = 1
MODE_INT16 = 2
MODE_INT8 = 3
MODE_INT4 = 4


class AccumulatorScoreboard:
    def __init__(self) -> None:
        self.kind = [0, 0]
        self.mode: list[int | None] = [None, None]
        self.valid = [False, False]
        self.integer = [0, 0]
        self.floating = [FloatAcc36State(0, 0), FloatAcc36State(0, 0)]
        self.int16_mid: int | None = None

    def apply(self, tx: dict[str, int]) -> dict[str, int]:
        mode = tx["mode"]
        kind = tx["kind"]
        group_valid = tx["group_valid"]
        first = tx["first"]
        phase = tx["phase"]
        c0, c1 = tx["c0"], tx["c1"]
        e0, e1 = tx["e0"], tx["e1"]

        update_valid = not (mode == MODE_INT16 and phase == 0)
        update_group_valid = group_valid if update_valid else 0
        if mode == MODE_INT16:
            assert group_valid == 1 and kind == 0
            if phase == 0:
                old = 0 if first & 1 else self.integer[0]
                self.int16_mid = old + c0
                assert -(1 << 35) <= self.int16_mid < (1 << 35)
                self.mode[0] = mode
            else:
                assert self.int16_mid is not None
                self.integer[0] = saturate_int32(self.int16_mid + c0)
                self.int16_mid = None
                self.kind[0] = 0
                self.mode[0] = mode
                self.valid[0] = True
        else:
            assert self.int16_mid is None
            for group in (0, 1):
                if not (group_valid & (1 << group)):
                    continue
                is_first = bool(first & (1 << group))
                contribution = c0 if group == 0 else c1
                exponent = e0 if group == 0 else e1
                if kind:
                    old = self.floating[group]
                    self.floating[group] = update_float_acc36(
                        old, contribution, exponent, first=is_first
                    )
                else:
                    old_int = 0 if is_first else self.integer[group]
                    self.integer[group] = saturate_int32(old_int + contribution)
                self.kind[group] = kind
                self.mode[group] = mode
                self.valid[group] = True

        return {
            "update_valid": int(update_valid),
            "update_group_valid": update_group_valid,
            "update_kind": kind,
            "state_valid": int(self.valid[0]) | (int(self.valid[1]) << 1),
            "kind0": self.kind[0],
            "kind1": self.kind[1],
            "int0": self.integer[0] & 0xFFFFFFFF,
            "int1": self.integer[1] & 0xFFFFFFFF,
            "sig0": self.floating[0].sig & MASK36,
            "exp0": self.floating[0].exp,
            "sig1": self.floating[1].sig & MASK36,
            "exp1": self.floating[1].exp,
            "inf0": self.floating[0].infinity_sign & 3,
            "inf1": self.floating[1].infinity_sign & 3,
        }


def _signed_rand(rng: random.Random, width: int) -> int:
    return rng.randrange(-(1 << (width - 1)), 1 << (width - 1))


def _first_for(acc: AccumulatorScoreboard, group_valid: int, kind: int,
               mode: int) -> int:
    first = 0
    for group in (0, 1):
        if group_valid & (1 << group):
            if (not acc.valid[group] or acc.kind[group] != kind or
                    acc.mode[group] != mode):
                first |= 1 << group
    return first


def _build_cases() -> list[dict[str, int]]:
    rng = random.Random(0xACC3_6037)
    acc = AccumulatorScoreboard()
    cases: list[dict[str, int]] = []

    def add(mode: int, kind: int, gv: int, first: int, phase: int,
            c0: int, c1: int = 0, e0: int = 0, e1: int = 0) -> None:
        tx = {
            "mode": mode, "kind": kind, "group_valid": gv, "first": first,
            "phase": phase, "c0": c0, "c1": c1, "e0": e0, "e1": e1,
            "tag0": (len(cases) * 17 + 5) & 0xFF,
            "tag1": (len(cases) * 29 + 9) & 0xFF,
        }
        expected = acc.apply(tx)
        cases.append({**tx, **expected})

    # Exact values, carry, signed ties, tail-only cancellation, FTZ and Inf.
    for sign in (-1,1):
        for offset in (0,1,2,3):
            add(MODE_FP19,1,1,1,0,sign*((1<<34)+offset),e0=127)
            add(MODE_FP19,1,1,0,0,sign*(1<<34),e0=92)
        add(MODE_FP19,1,1,1,0,sign*(1<<34),e0=127)
        add(MODE_FP19,1,1,0,0,-sign*((1<<35)-1),e0=126)
        add(MODE_FP19,1,1,1,0,sign*(1<<34),e0=1)
        add(MODE_FP19,1,1,0,0,-sign,e0=1)
        add(MODE_FP19,1,1,1,0,sign*((1<<35)-1),e0=254)
        add(MODE_FP19,1,1,0,0,sign*(1<<34),e0=219)  # RNE overflow
        add(MODE_FP19,1,1,0,0,-sign*(1<<34),e0=254) # Inf retained
        add(MODE_FP19,1,1,1,0,0,e0=0)              # first clears Inf
    add(MODE_FP19,1,1,1,0,-(1<<35),e0=127)
    add(MODE_FP19,1,1,0,0,(1<<35)-1,e0=127)

    # Adversarial distances, leading positions and positive/negative tails.
    for lead in (0,1,16,25,33,34):
        for distance in (0,1,2,3,4,5,34,35,36,38,39,40,100,253):
            for sign in (-1,1):
                for tail in (0,1,3,7):
                    c = min((1<<35)-1,(1<<lead)+tail)
                    add(MODE_FP19,1,1,1,0,sign*((1<<34)+tail),e0=254)
                    add(MODE_FP19,1,1,0,0,-sign*c,e0=254-distance)

    # Close opposite-sign operands: exact cancellation can expose any of
    # the 34 fraction bits; d=1 also exercises the negative signed36 endpoint.
    for bit in range(35):
        for sign in (-1,1):
            add(MODE_FP19,1,1,1,0,sign*(1<<34),e0=127)
            add(MODE_FP19,1,1,0,0,-sign*((1<<34)-(1<<bit)),e0=127)
            add(MODE_FP19,1,1,1,0,sign*(1<<34),e0=127)
            add(MODE_FP19,1,1,0,0,-sign*((1<<35)-(1<<bit)),e0=126)

    # Independent groups, finite scale range, zero and invalid-group hold.
    add(MODE_FP9,1,3,3,0,1<<29,-(1<<30),132,133)
    add(MODE_FP9,1,2,0,0,0,1<<30,0,133)
    add(MODE_FP9,1,1,0,0,-(1<<29),0,132,0)
    # INT32 saturates per dot; INT16 phase0 survives a valid bubble.
    add(MODE_INT16,0,1,1,0,1<<33)
    add(MODE_INT16,0,1,0,1,1<<32)
    add(MODE_INT16,0,1,1,0,-(1<<33))
    add(MODE_INT16,0,1,0,1,-(1<<32))
    add(MODE_INT8,0,3,3,0,258064,-258064)
    for _ in range(8400):
        add(MODE_INT8,0,3,0,0,258064,-258064)
    add(MODE_INT4,0,3,3,0,784,-784)

    for index in range(5000):
        mode = rng.choice((MODE_FP19,MODE_FP9,MODE_INT8))
        kind = int(mode != MODE_INT8 or rng.randrange(2))
        gv = 1 if mode == MODE_FP19 else rng.choice((1,2,3))
        first = _first_for(acc,gv,kind,mode)
        # Periodic restarts stop an overflow Inf from hiding finite vectors.
        if index % 7 == 0: first = gv
        if kind:
            if mode == MODE_FP19:
                c0,c1 = _signed_rand(rng,36),0
            elif mode == MODE_FP9:
                c0,c1 = _signed_rand(rng,28)<<8,_signed_rand(rng,28)<<8
            else:
                c0,c1 = rng.randrange(-258064,258065)<<17,rng.randrange(-258064,258065)<<17
            e0,e1 = rng.randrange(1,255),rng.randrange(1,255)
        else:
            c0,c1 = rng.randrange(-258064,258065),rng.randrange(-258064,258065)
            e0=e1=0
        add(mode,kind,gv,first,0,c0,c1,e0,e1)

    return cases


def _write_vectors(path: Path, cases: list[dict[str, int]]) -> None:
    names = (
        "mode kind group_valid first phase c0 c1 e0 e1 tag0 tag1 "
        "update_valid update_group_valid update_kind state_valid kind0 kind1 "
        "int0 int1 sig0 exp0 sig1 exp1 inf0 inf1"
    ).split()
    with path.open("w", encoding="ascii") as out:
        for tx in cases:
            fields = []
            for name in names:
                value = tx[name]
                if name == "c0":
                    fields.append(f"{value & MASK36:010x}")
                elif name == "c1":
                    fields.append(f"{value & MASK36:09x}")
                elif name in ("e0", "e1", "exp0", "exp1"):
                    fields.append(f"{value & 0xFF:02x}")
                elif name in ("sig0", "sig1"):
                    fields.append(f"{value & MASK36:010x}")
                elif name in ("int0", "int1"):
                    fields.append(f"{value & 0xFFFFFFFF:08x}")
                elif name in ("tag0", "tag1"):
                    fields.append(f"{value & 0xFF:02x}")
                else:
                    fields.append(f"{value:x}")
            out.write(" ".join(fields) + "\n")


def test_acc_stage_matches_q234_reference_and_state_handshake(tmp_path: Path):
    cases = _build_cases()
    vectors = tmp_path / "acc_vectors.txt"
    _write_vectors(vectors, cases)
    image = tmp_path / "mac_acc_stage.vvp"
    compiled = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_mac_acc_stage", "-o", str(image),
            "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/rtl/systemverilog/mac_acc_float_prepare36.sv",
            "hardware/rtl/systemverilog/mac_acc_stage.sv", "hardware/verification/tb/tb_mac_acc_stage.sv",
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        ["vvp", str(image), f"+VEC={vectors}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"mac_acc_stage PASS: {len(cases)} vectors" in simulated.stdout


@pytest.mark.parametrize("synthesis_macro", [False, True])
def test_acc_rtl_compiles(tmp_path, synthesis_macro):
    """Compilation is not synthesis or new-format numerical verification."""
    args = ["iverilog", "-g2012"]
    if synthesis_macro:
        args.append("-DSYNTHESIS")
    result = subprocess.run(args + [
        "-s", "tb_mac_acc_stage", "-o", str(tmp_path / "acc.vvp"),
        "hardware/rtl/systemverilog/mac_types_pkg.sv", "hardware/rtl/systemverilog/mac_acc_float_prepare36.sv",
        "hardware/rtl/systemverilog/mac_acc_stage.sv", "hardware/verification/tb/tb_mac_acc_stage.sv",
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('bad,marker', [
    (1,'ACC state requires acc_first'),
    (2,'ACC nonzero input exponent boundary deferred'),
    (3,'ACC nonzero input exponent boundary deferred'),
    (4,'INT16 phase1 arrived without phase0'),
])
def test_acc_rejects_undefined_input_and_phase(tmp_path,bad,marker):
    cases=_build_cases()[:1]
    vec=tmp_path/'bad.txt'
    _write_vectors(vec,cases)
    image=tmp_path/'bad.vvp'
    compiled=subprocess.run([
        'iverilog','-g2012','-s','tb_mac_acc_stage','-o',str(image),
        'hardware/rtl/systemverilog/mac_types_pkg.sv','hardware/rtl/systemverilog/mac_acc_float_prepare36.sv',
        'hardware/rtl/systemverilog/mac_acc_stage.sv','hardware/verification/tb/tb_mac_acc_stage.sv',
    ],cwd=ROOT,capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stdout+compiled.stderr
    result=subprocess.run(['vvp',str(image),f'+VEC={vec}',f'+BAD={bad}'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and marker in result.stdout+result.stderr
