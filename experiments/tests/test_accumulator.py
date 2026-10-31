from mac_experiment.accumulation.accumulator import Fp32Accumulator
from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32


def test_no_early_rounding_of_partial_sum():
    """acc=-1, partial=1+2^-24 must yield 2^-24, not 0."""
    acc = Fp32Accumulator()
    acc.update(Dyadic(-1, 0))
    assert acc.state.bits == 0xBF800000
    outcome = acc.update(Dyadic(1, 0) + Dyadic(1, -24))
    assert outcome.fp32.bits == 0x33800000
    assert outcome.fp32.to_float() == 2.0**-24
    assert outcome.exact_sum == Dyadic(1, -24)
    assert outcome.delta == Dyadic.zero()


def test_accumulator_gradual_underflow():
    outcome = Fp32Accumulator().update(Dyadic(1, -149))
    assert outcome.fp32.bits == 0x00000001
    outcome = Fp32Accumulator().update(Dyadic(1, -150))
    assert outcome.fp32.bits == 0x00000000
    outcome = Fp32Accumulator().update(Dyadic(3, -150))
    assert outcome.fp32.bits == 0x00000002
    outcome = Fp32Accumulator().update(Dyadic(3, -151))
    assert outcome.fp32.bits == 0x00000001
    outcome = Fp32Accumulator().update(Dyadic(1, -151))
    assert outcome.fp32.bits == 0x00000000


def test_accumulator_delta_and_identity_single_step():
    acc = Fp32Accumulator()
    partial = Dyadic((1 << 24) + 1, -24)
    outcome = acc.update(partial)
    assert outcome.delta is not None
    assert outcome.fp32.exact_dyadic() - partial == outcome.delta


def test_absorbed_update_event():
    acc = Fp32Accumulator()
    acc.update(Dyadic(1, 0))
    outcome = acc.update(Dyadic(1, -30))
    assert outcome.absorbed
    assert outcome.fp32.bits == 0x3F800000
    quiet = acc.update(Dyadic.zero())
    assert not quiet.absorbed


def test_overflow_event_is_distinct_from_window_overflow():
    acc = Fp32Accumulator()
    max_finite = Fp32.from_bits(0x7F7FFFFF)
    acc.update(max_finite.exact_dyadic())
    outcome = acc.update(max_finite.exact_dyadic())
    assert outcome.overflowed
    assert outcome.fp32.is_inf()
    assert outcome.non_finite
    assert outcome.delta is None


def test_non_finite_state_propagates():
    acc = Fp32Accumulator()
    acc.update(Fp32.from_bits(0x7F7FFFFF).exact_dyadic())
    acc.update(Fp32.from_bits(0x7F7FFFFF).exact_dyadic())
    outcome = acc.update(Dyadic(-1, 0))
    assert outcome.fp32.is_inf()
    assert outcome.non_finite
