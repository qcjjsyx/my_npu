"""Bit-level checks for the new-architecture accumulator (T0-T5).

T0  contribution normalization and input contract
T1  exhaustive small-p structure (all normalized operand pairs)
T2  p=40 case-space coverage and directed boundaries
T3  metamorphic properties (translation, negation, swap, zero)
T4  real local-contribution traces from the accumulation_precision run
T5  range endpoints and analytic-lemma assertions

T6 (RTL co-simulation) is intentionally absent until the Verilog exists.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from reference.new_arch.acc_fixed import (
    E,
    AccFixedError,
    is_canonical,
    normalize_contribution,
    update,
)
from reference.new_arch.oracle import exact_update

P = 40
REPRESENTATIVES = (
    Path(__file__).resolve().parents[3]
    / "results/numerical/accumulation_precision/runs"
    / "20261110T151440_017599Z_08b5e360/representatives.json"
)


def _small_values(p: int) -> list[int]:
    return [m for m in range(-(1 << p) + 1, 1 << p) if is_canonical(m, p)]


def _p40_grid():
    """Deterministic operand/difference grid used by T2 and T3."""
    base = 1 << (P - 1)
    top = 1 << P
    magnitudes = [
        base, base + 1, base + 2, base + 3, base + 4, base + 7,
        base + (1 << (P - 2)), 3 * (1 << (P - 2)),
        top - 1, top - 2, top - 3, top - 4, top - 8,
    ]
    values = []
    for m in magnitudes:
        values.extend((m, -m))
    for a in values:
        for b in values:
            for d in (0, 1, 2, 3, 4, 5, 6, 8, 40, 41, 1023):
                yield a, max(500, d - 512), b, max(500, d - 512) - d, d


# ---------------------------------------------------------------- T0

@pytest.mark.parametrize("p", [5, 6, 40])
def test_t0_normalization_is_lossless(p):
    for N in _small_values(p)[:200] if p < 40 else [
        (1 << (p - 1)) - 1, (1 << p) - 1, 0x7FFFFFFFFF, (1 << 39) - 1,
        -((1 << 39) - 1), 1, -1,
    ]:
        if N == 0:
            assert normalize_contribution(0, 7, p) == (0, 0)
            continue
        for e in (-306, -1, 0, 100, 291):
            M_v, H_v = normalize_contribution(N, e, p)
            assert is_canonical(M_v, p)
            k = abs(N).bit_length() - 1
            assert H_v == e + k
            assert M_v << k == N << (p - 1)  # exact, no bits lost


@pytest.mark.parametrize("p", [5, 6, 40])
def test_t0_normalization_contract(p):
    assert normalize_contribution(0, -7, p) == (0, 0)
    with pytest.raises(AccFixedError):
        normalize_contribution(1 << p, 0, p)
    with pytest.raises(AccFixedError):
        normalize_contribution(-(1 << p), 0, p)


def test_t0_p40_fp16_max_contribution():
    N = 549_219_074_048  # FP16 |N| bound from the local window
    assert N < (1 << 39)
    M_v, H_v = normalize_contribution(N, -2, P)
    assert M_v == N << 1 and H_v == -2 + 38
    assert is_canonical(M_v, P)


def test_t0_oracle_matches_existing_exact_model():
    """Cross-check the new oracle against the confirmed exact dyadic model."""
    model = pytest.importorskip("mac_experiment.accumulation_precision.model")
    from mac_experiment.arithmetic.dyadic import Dyadic

    State = model.State
    for M_c, H_c, M_v, H_v, _d in _some_cases(400):
        expected = exact_update(M_c, H_c, M_v, H_v, P)
        updated = State(P, M_c, H_c).update(Dyadic(M_v, H_v - (P - 1)))
        assert (updated.M, updated.H) == expected


# ---------------------------------------------------------------- T1

@pytest.mark.parametrize("p", [5, 6, 8])
def test_t1_small_p_all_pairs(p):
    values = _small_values(p)
    ds = (0, 1, 2, 3, 4, 5, 6, 8, 12, 20, 64, 1000)
    H_a = 500
    count = 0
    for M_c in values:
        for M_v in values:
            for d in ds:
                debug: dict = {}
                got = update(M_c, H_a, M_v, H_a - d, p, debug)
                exp = exact_update(M_c, H_a, M_v, H_a - d, p)
                assert got == exp, (p, M_c, M_v, d, got, exp, debug)
                if "sigma" in debug:
                    if debug["tau"]:
                        # Only truncated alignments (d >= 3) may lose bits, and
                        # then the window drops at most one position.
                        assert debug["sigma"] <= 1, (p, M_c, M_v, d, debug)
                else:
                    assert not debug["tau"], (p, M_c, M_v, d, debug)
                count += 1
    assert count == len(values) ** 2 * len(ds)


# ---------------------------------------------------------------- T2

def test_t2_p40_case_space_and_directed():
    coverage = {
        "sigma": set(), "grs": set(), "tau": set(), "renorm": set(),
        "tie_parity": set(), "absorption": set(), "zero": False,
        "sigma1_tau_g1": set(),
    }
    count = 0
    for M_c, H_c, M_v, H_v, d in _p40_grid():
        debug: dict = {}
        got = update(M_c, H_c, M_v, H_v, P, debug)
        exp = exact_update(M_c, H_c, M_v, H_v, P)
        assert got == exp, (M_c, H_c, M_v, H_v, d, got, exp, debug)
        count += 1
        if "sigma" not in debug:
            coverage["zero"] = True
            continue
        sigma, G, R, S = debug["sigma"], debug["G"], debug["R"], bool(debug["S"])
        coverage["sigma"].add(sigma)
        coverage["grs"].add((G, R, S))
        tau_value = (Fraction((M_v << E) - (debug["B"] << d), 1 << d)
                     if debug["tau"] else None)
        if tau_value is not None:
            coverage["tau"].add(tau_value)
            assert sigma <= 1, (M_c, M_v, d, debug)
        if debug["c"]:
            coverage["renorm"].add(1 if debug["M"] > 0 else -1)
        if sigma == 0 and G == 1 and R == 0 and not S:
            coverage["tie_parity"].add(debug["M_pre"] & 1)
        if d == 1023 and debug["tau"]:
            coverage["absorption"].add("pos" if debug["B"] == 0 else "neg")
        if sigma == 1 and G == 1 and tau_value is not None:
            coverage["sigma1_tau_g1"].add(tau_value)
    assert count > 6000
    assert coverage["sigma"] >= {-1, 0, 1}
    assert coverage["grs"] == {(g, r, s) for g in (0, 1) for r in (0, 1) for s in (False, True)}
    assert coverage["tau"] >= {Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)}
    assert coverage["renorm"] == {1, -1}
    assert coverage["tie_parity"] == {0, 1}
    assert coverage["absorption"] == {"pos", "neg"}
    assert coverage["zero"]
    assert coverage["sigma1_tau_g1"] >= {Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)}


def test_t2_p40_directed_boundaries():
    # +2^p carry: M_pre = 2^40-1, G=1, tie resolved by odd LSB.
    M, H = update((1 << P) - 1, 0, 1 << 39, -40, P)
    assert (M, H) == (1 << 39, 1) == exact_update((1 << P) - 1, 0, 1 << 39, -40, P)
    # -2^p intermediate from exact same-exponent cancellation.
    M, H = update(-(1 << 39), 0, -(1 << 39), 0, P)
    assert M == -(1 << 39) and H == 1
    # sigma = p+1: exact cancellation to a negative power of two.
    got = update(-(1 << 39), 0, (1 << P) - 1, -1, P)
    assert got == (-(1 << 39), -40) == exact_update(-(1 << 39), 0, (1 << P) - 1, -1, P)
    # Deep cancellation d=1: 1 - 31/32 -> 1/32.
    got = update(1 << 39, 0, -(31 << 35), -1, P)
    assert got == (1 << 39, -5) == exact_update(1 << 39, 0, -(31 << 35), -1, P)
    # Absorption with a large (legal) exponent difference, both signs.
    for b in (1 << 39, -(1 << 39)):
        got = update(1 << 39, 0, b, -306, P)
        assert got == exact_update(1 << 39, 0, b, -306, P)


def test_t2_d3_sigma1_counterexample():
    """d=3 can produce sigma=1; the lemma is sigma<=1 for all d>=3."""
    a, H_a, b, H_b = 1 << 39, 0, -((1 << P) - 1), -3
    debug: dict = {}
    got = update(a, H_a, b, H_b, P, debug)
    assert (debug["d"], debug["t"], debug["sigma"]) == (3, 40, 1)
    assert debug["tau"] and debug["T"] == 3 * (1 << 39)
    assert abs(debug["T"]) >= (1 << 41) - (1 << (42 - 3)) == 3 * (1 << 39)
    assert got == exact_update(a, H_a, b, H_b, P)


def test_t2_negative_power_sigma41():
    """sigma=p+1 only from T=-2; the signed re-normalization must fire."""
    debug: dict = {}
    got = update(-(1 << 39), 0, (1 << P) - 1, -1, P, debug)
    assert (debug["t"], debug["sigma"], debug["M_pre"], debug["c"]) == (
        0, 41, -(1 << P), 1)
    assert got == (-(1 << 39), -40) == exact_update(-(1 << 39), 0, (1 << P) - 1, -1, P)


# ---------------------------------------------------------------- T3

def _some_cases(limit=120):
    for i, case in enumerate(_p40_grid()):
        if i >= limit:
            return
        yield case


def test_t3_metamorphic_translation_negation_swap_zero():
    for M_c, H_c, M_v, H_v, _d in _some_cases():
        M, H = update(M_c, H_c, M_v, H_v, P)
        if M == 0:
            continue  # zero has no exponent to translate
        for k in (-1, 1):
            if (-512 <= H_c + k <= 511 and -512 <= H_v + k <= 511
                    and -512 <= H + k <= 511):
                assert update(M_c, H_c + k, M_v, H_v + k, P) == (M, H + k)
        assert update(-M_c, H_c, -M_v, H_v, P) == (-M, H)
        assert update(M_v, H_v, M_c, H_c, P) == (M, H)
        assert update(M_c, H_c, 0, 0, P) == (M_c, H_c)
        assert update(0, 0, M_v, H_v, P) == (M_v, H_v)


# ---------------------------------------------------------------- T4

@pytest.fixture(scope="module")
def representatives():
    if not REPRESENTATIVES.is_file():
        pytest.skip(f"accumulation_precision run not present: {REPRESENTATIVES}")
    with REPRESENTATIVES.open() as f:
        return json.load(f)


def test_t4_real_traces(representatives):
    records = [r for r in representatives if r["p_acc"] == P]
    assert records
    chosen = [r for r in records if r["length"] in (128, 16384)][:40]
    assert len(chosen) >= 10
    updates = 0
    for record in chosen:
        state = (0, 0)
        for step in record["trace"]:
            contribution = step["contribution"]
            M_v, H_v = normalize_contribution(int(contribution["m"]), contribution["e"], P)
            state = update(state[0], state[1], M_v, H_v, P)
            assert state == (int(step["M"]), step["H"]), (
                record["mode"], record["length"], record["source_id"], state, step)
            assert -346 <= state[1] <= 306
            updates += 1
        assert state == (int(record["state"]["M"]), record["state"]["H"])
    assert updates > 10_000


def test_t4_trace_update_counts(representatives):
    for record in [r for r in representatives if r["p_acc"] == P][:20]:
        assert len(record["trace"]) == record["updates"]


# ---------------------------------------------------------------- T5

def test_t5_range_endpoints():
    # Largest legal FP8 contribution at the largest MX scale exponent (H_v=291).
    M_v, H_v = normalize_contribution(15_099_494_400, 258, P)
    assert H_v == 291
    state = update(1 << 39, 305, M_v, H_v, P)
    assert state == exact_update(1 << 39, 305, M_v, H_v, P)
    assert -346 <= state[1] <= 306
    # FP16 largest local contribution stays well inside the exponent range.
    M_v, H_v = normalize_contribution(549_219_074_048, -2, P)
    assert update(1 << 39, 0, M_v, H_v, P) == exact_update(1 << 39, 0, M_v, H_v, P)
    # Very small negative contribution absorbed by a large state.
    state = update(1 << 39, 305, 1 << 39, -306, P)
    assert state == (1 << 39, 305)
    # Reachable-domain closure over legal H values.
    for M_c in ((1 << 39), (1 << P) - 1, -((1 << P) - 1)):
        for H_c in (-346, -300, -100, 0, 100, 305, 306):
            for M_v, H_v in (((1 << 39), -306), ((1 << P) - 1, -306),
                             (1 << 39, 291), (-(1 << 39), 291),
                             (1 << 39, 0), (-((1 << P) - 1), 0)):
                if H_c == 306 and M_c != (1 << 39):
                    continue  # only 2^306 is reachable at H=306
                debug: dict = {}
                got = update(M_c, H_c, M_v, H_v, P, debug)
                assert got == exact_update(M_c, H_c, M_v, H_v, P)
                assert -346 <= got[1] <= 306, (M_c, H_c, M_v, H_v, got, debug)


def test_t5_max_adder_operand():
    # d=0 same sign at max magnitude: |T| hits 2^43-8, the 45-bit adder bound.
    debug: dict = {}
    got = update((1 << P) - 1, 0, (1 << P) - 1, 0, P, debug)
    assert debug["T"] == (1 << (P + E + 1)) - (1 << (E + 1))
    assert got == ((1 << P) - 1, 1) == exact_update((1 << P) - 1, 0, (1 << P) - 1, 0, P)
