"""W_acc accumulator tests: independent Fraction oracle and targeted semantics."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.widthselect.acc_format import (
    ACC_EVENT_KEYS,
    HwAccState,
    decode_state,
    fit_operand,
    hw_add,
    metadata,
    operands_bits,
    output_bits,
    result_bits,
    state_bits,
    validate_w_acc,
    zero_state,
)
from mac_experiment.widthselect.reference import (
    FORMAT_ALIGNMENT,
    block_products,
    evaluate_pair_acc,
    evaluate_pair_joint,
)
from mac_experiment.widthselect.tensors import TargetTensor
from mac_experiment.widthselect.tf32_rne import round_dyadic_to_tf32
from oracle import oracle_align_int, oracle_fp32_bits, round_fraction_to_int

ORACLE_BIAS = 126


def _floor_log2(a: Fraction) -> int:
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


def _window_bounds(w_acc: int) -> tuple[int, int]:
    return -(1 << (w_acc - 1)), (1 << (w_acc - 1)) - 1


def oracle_fit(value: Fraction, w_acc: int) -> tuple[int, int]:
    """Independent smallest-window search for an exact rational operand."""
    low, high = _window_bounds(w_acc)
    w = _floor_log2(abs(value)) - w_acc + 2
    while True:
        n = round_fraction_to_int(value / Fraction(2) ** w)
        if low <= n <= high:
            break
        w += 1
    while True:
        candidate = w - 1
        n = round_fraction_to_int(value / Fraction(2) ** candidate)
        if low <= n <= high:
            w = candidate
        else:
            break
    return round_fraction_to_int(value / Fraction(2) ** w), w


def oracle_hw_add(state, value: Fraction, w_acc: int) -> tuple[tuple[int, int], dict]:
    """Independent Fraction model of the confirmed HWAdd semantics."""
    flags = {key: 0 for key in ACC_EVENT_KEYS}
    flags["acc_updates"] = 1
    if state is not None and state[0] == 255:
        flags["acc_marker_propagated"] = 1
        return state, flags
    if value == 0:
        flags["acc_input_zero"] = 1
    m_s, e_s = (state[1], state[0] - ORACLE_BIAS - w_acc) if state else (0, 0)
    if m_s == 0 and value == 0:
        return (0, 0), flags
    v_s = Fraction(m_s) * Fraction(2) ** e_s
    w_s = oracle_fit(v_s, w_acc)[1] if m_s else None
    w_i = oracle_fit(value, w_acc)[1] if value != 0 else None
    w = max(candidate for candidate in (w_s, w_i) if candidate is not None)
    floor = -126 - w_acc
    w = max(w, floor)
    n_s = round_fraction_to_int(v_s / Fraction(2) ** w) if m_s else 0
    n_i = round_fraction_to_int(value / Fraction(2) ** w) if value != 0 else 0
    dropped_s = m_s != 0 and Fraction(n_s) * Fraction(2) ** w != v_s
    dropped_i = value != 0 and Fraction(n_i) * Fraction(2) ** w != value
    if dropped_s:
        flags["acc_state_bits_dropped"] = 1
    if dropped_i:
        flags["acc_input_bits_dropped"] = 1
    if value != 0 and n_i == 0:
        flags["acc_input_absorbed"] = 1
    integer_sum = n_s + n_i
    low, high = _window_bounds(w_acc)
    if integer_sum > high or integer_sum < low:
        flags["acc_carry"] = 1
    if integer_sum == 0:
        if m_s != 0 and value != 0:
            flags["acc_cancellation"] = 1
        return (0, 0), flags
    e = w
    while True:
        top = (integer_sum >> w_acc) & 1
        nxt = (integer_sum >> (w_acc - 1)) & 1
        if top != nxt:
            break
        if e - 1 < floor:
            break
        integer_sum *= 2
        e -= 1
    E_code = e + ORACLE_BIAS + w_acc
    if E_code > 254:
        flags["acc_overflow"] = 1
        return (255, integer_sum), flags
    if E_code == 0:
        top = (integer_sum >> w_acc) & 1
        nxt = (integer_sum >> (w_acc - 1)) & 1
        if top == nxt:
            flags["acc_subnormal"] = 1
    return (E_code, integer_sum), flags


def _state_tuple(state: HwAccState | None):
    if state is None:
        return (0, 0)
    return (state.E_code, state.M)


def test_width_conventions():
    assert state_bits(30) == 39
    assert operands_bits(30) == 30
    assert result_bits(30) == 31
    with pytest.raises(ValueError):
        validate_w_acc(7)
    with pytest.raises(ValueError):
        validate_w_acc(61)


def test_zero_canonical_and_marker_stickiness():
    w = 8
    zero = zero_state(w)
    assert zero.M == 0 and zero.E_code == 0 and decode_state(zero) == Dyadic.zero()
    with pytest.raises(ValueError):
        HwAccState(5, 0, w)
    with pytest.raises(ValueError):
        HwAccState(255, 0, w)
    overflow = HwAccState(255, 42, w)
    sticky, flags = hw_add(overflow, Dyadic(1, 0), w)
    assert sticky == overflow and flags["acc_marker_propagated"] == 1
    assert output_bits(overflow) == 0x7F80_0000
    negative = HwAccState(255, -42, w)
    assert output_bits(negative) == 0xFF80_0000
    with pytest.raises(ValueError):
        decode_state(overflow)


def test_fit_operand_matches_oracle():
    rng = random.Random(4)
    for w in (8, 10):
        for _ in range(300):
            m = rng.randint(-1 << 20, 1 << 20) or 1
            e = rng.randint(-40, 20)
            n, ww = fit_operand(m, e, w)
            expected_n, expected_w = oracle_fit(Fraction(m) * Fraction(2) ** e, w)
            assert (n, ww) == (expected_n - 0, expected_w)
            assert n == expected_n


def test_fit_operand_negative_boundary():
    # W_acc=8, m=-257: the negative end of the two's-complement window allows
    # the finer window w=1 (RNE(-257/2) = -128) instead of the conservative w=2
    assert fit_operand(-257, 0, 8) == (-128, 1)
    # directed negative powers of two and their neighbours, checked against the
    # independent bidirectional oracle
    for w in (8, 10, 12):
        for bits in range(1, 16):
            for m in (-(1 << bits), -(1 << bits) + 1, -(1 << bits) - 1, (1 << bits), (1 << bits) - 1):
                for e in (-3, 0, 5):
                    ours = fit_operand(m, e, w)
                    expected = oracle_fit(Fraction(m) * Fraction(2) ** e, w)
                    assert ours == expected, (w, m, e, ours, expected)
    # the documented sequence: 2 then -257 gives -254, not -256
    state, _ = hw_add(None, Dyadic(2, 0), 8)
    state, _ = hw_add(state, Dyadic(-257, 0), 8)
    assert decode_state(state) == Dyadic(-254, 0)


def test_random_sequences_match_oracle():
    for w in (8, 10):
        rng = random.Random(100 + w)
        state = None
        oracle_state = None
        for index in range(400):
            if index % 5 == 4:
                m = 0  # exercise the zero partial-sum update path
            else:
                m = rng.randint(-255, 255) or 1
            e = rng.randint(-12, 8)
            value = Dyadic(m, e)
            fraction = Fraction(m) * Fraction(2) ** e
            state, flags = hw_add(state, value, w)
            oracle_state, oracle_flags = oracle_hw_add(oracle_state, fraction, w)
            assert _state_tuple(state) == oracle_state, (w, value, _state_tuple(state), oracle_state)
            for key in ACC_EVENT_KEYS:
                assert flags[key] == oracle_flags[key], (w, value, key)
            # the state always decodes back exactly and respects the width contract
            if not state.is_overflow_marker:
                decoded = decode_state(state)
                assert decoded == Dyadic(state.M, state.E_code - ORACLE_BIAS - w)
            assert -(1 << w) <= state.M <= (1 << w) - 1


def test_targeted_semantics():
    w = 8
    # exact addition with a carry into the extra bit
    state, _ = hw_add(None, Dyadic(100, 0), w)
    state, flags = hw_add(state, Dyadic(100, 0), w)
    assert decode_state(state) == Dyadic(200, 0)
    assert flags["acc_carry"] == 1
    # large exponent difference absorbs the tiny operand
    big, _ = hw_add(None, Dyadic((1 << 8) - 1, 20), w)
    state, flags = hw_add(big, Dyadic(1, -20), w)
    assert flags["acc_input_absorbed"] == 1
    assert flags["acc_input_bits_dropped"] == 1
    # cancellation to the canonical zero
    positive, _ = hw_add(None, Dyadic(7, 3), w)
    state, flags = hw_add(positive, Dyadic(-7, 3), w)
    assert state == zero_state(w)
    assert flags["acc_cancellation"] == 1
    # subnormal floor: the smallest representable state is 2**-134 at W_acc=8
    state, flags = hw_add(None, Dyadic(3, -136), w)
    assert state == HwAccState(0, 1, w)
    assert decode_state(state) == Dyadic(1, -134)
    assert flags["acc_subnormal"] == 1
    # below half an LSB flushes to zero
    state, _ = hw_add(None, Dyadic(1, -140), w)
    assert state == zero_state(w)
    # overflow is sticky and maps to +/-Inf
    state, flags = hw_add(None, Dyadic((1 << 7) - 1, 134), w)
    assert flags["acc_overflow"] == 1
    assert state.is_overflow_marker
    assert output_bits(state) == 0x7F80_0000


def test_zero_input_still_narrows_state():
    w = 8
    # build the normalized state 129 = 64 + 65 (no zero bypass, so the exact
    # zero partial sum still runs the state -> operand window narrowing)
    state, _ = hw_add(None, Dyadic(64, 0), w)
    state, _ = hw_add(state, Dyadic(65, 0), w)
    assert decode_state(state) == Dyadic(129, 0)
    zeroed, flags = hw_add(state, Dyadic.zero(), w)
    assert flags["acc_input_zero"] == 1
    assert decode_state(zeroed) == Dyadic(128, 0)
    # an absorbed tiny nonzero value takes the same documented path
    absorbed, flags = hw_add(state, Dyadic(1, -40), w)
    assert flags["acc_input_absorbed"] == 1
    assert absorbed.to_dict() == zeroed.to_dict()


def test_no_hidden_precision_across_updates():
    w = 10
    state, _ = hw_add(None, Dyadic(1, 0), w)
    updated, flags = hw_add(state, Dyadic(1, -40), w)
    assert flags["acc_input_absorbed"] == 1
    narrowed, _ = hw_add(updated, Dyadic.zero(), w)
    # nothing is recovered: the zero update may only drop the state LSB
    assert set(updated.to_dict()) == {"E_code", "M", "w_acc"}
    assert abs(decode_state(narrowed)) <= abs(decode_state(updated))
    assert narrowed.M == updated.M // 2 or abs(narrowed.M) <= abs(updated.M)
    assert len(str(abs(updated.M))) <= 4  # fits the declared width


def _tf32_tensor(values, rows, cols) -> TargetTensor:
    codes = [round_dyadic_to_tf32(value).code for value in values]
    return TargetTensor(format="tf32", rows=rows, cols=cols, codes=codes)


def test_engine_matches_oracle_simulation():
    rng = random.Random(21)
    K, w_acc = 32, 10
    x_values = [Dyadic(rng.randint(-64, 64) or 1, rng.randint(-4, 1)) for _ in range(K)]
    w_values = [Dyadic(rng.randint(-64, 64) or 1, rng.randint(-4, 1)) for _ in range(K)]
    x = _tf32_tensor(x_values, 1, K)
    w = _tf32_tensor(w_values, K, 1)

    expected_states = []
    oracle_state = None
    for block in range(K // 16):
        products = block_products(x, w, 0, 0, block, "tf32")
        s_b = sum((Fraction(p) * Fraction(2) ** (c - 274) for p, c in products), Fraction(0))
        oracle_state, _ = oracle_hw_add(oracle_state, s_b, w_acc)
        expected_states.append(oracle_state)

    result = evaluate_pair_acc(x, w, [16, 32], [w_acc])
    assert len(result.acc_snapshots) == 2
    for index, snapshot in enumerate(result.acc_snapshots):
        state = HwAccState.from_dict(snapshot["C_state"][0])
        assert _state_tuple(state) == expected_states[index]
        decoded = decode_state(state)
        assert Dyadic.from_dict(snapshot["C_exact"][0]) == decoded
        assert snapshot["Y_bits"][0] == oracle_fp32_bits(decoded.to_fraction())


def test_joint_engine_matches_oracle_simulation():
    rng = random.Random(33)
    K, w_acc = 32, 10
    w_align = 24
    x_values = [Dyadic(rng.randint(-64, 64) or 1, rng.randint(-4, 1)) for _ in range(K)]
    w_values = [Dyadic(rng.randint(-64, 64) or 1, rng.randint(-4, 1)) for _ in range(K)]
    x = _tf32_tensor(x_values, 1, K)
    w = _tf32_tensor(w_values, K, 1)
    constants = FORMAT_ALIGNMENT["tf32"]

    oracle_state = None
    expected = []
    for block in range(K // 16):
        products = block_products(x, w, 0, 0, block, "tf32")
        _, q_b_fraction = oracle_align_int(
            products, 0, w_align, constants["k_shift"], constants["window_offset"]
        )
        oracle_state, _ = oracle_hw_add(oracle_state, q_b_fraction, w_acc)
        expected.append(oracle_state)

    result = evaluate_pair_joint(x, w, [16, 32], [{"w_align": w_align, "w_acc": w_acc}])
    assert len(result.joint_snapshots) == 2
    for index, snapshot in enumerate(result.joint_snapshots):
        state = HwAccState.from_dict(snapshot["C_state"][0])
        assert _state_tuple(state) == expected[index]


def test_metadata_reports_confirmed_status():
    meta = metadata()
    assert meta["status"] == "hwacc_v1_confirmed_implemented"
    assert "no_hidden_state" in meta["policy"]
