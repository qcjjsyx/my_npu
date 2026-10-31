from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.execution.engine import SampleEngine
from mac_experiment.formats import get_format
from mac_experiment.sampling.sampler import Sampler


def make_engine(fmt_name="fp8_e4m3", dist="laplace", windows=(18,), snapshots=(4,), mode="width_sweep"):
    fmt = get_format(fmt_name)
    sampler = Sampler(fmt, dist)
    return SampleEngine(fmt, sampler, windows, snapshots, mode)


def test_decomposition_identity_and_exact_arithmetic():
    engine = make_engine()
    outcome = engine.run(1, 0)
    assert outcome.status == "ok"
    for record in outcome.records:
        assert record["record_type"] == "snapshot"
        a = Fp32.from_bits(record["a_bits"]).exact_dyadic()
        s = Dyadic.from_dict(record["S"])
        l1 = Dyadic.from_dict(record["L1"])
        e_align = Dyadic.from_dict(record["E_align"])
        e_acc = Dyadic.from_dict(record["E_acc"])
        assert a - s == e_align + e_acc
        assert l1.sign() >= 0
        assert not l1.is_zero()
        assert Fp32.from_bits(record["r_bits"]).is_finite()


def test_no_dropped_bits_case_candidate_equals_reference():
    """FP8 W=41 leaves the whole legal exponent range exact: a_B == r_B."""
    engine = make_engine(windows=(41,), snapshots=(4,))
    for sample_index in range(4):
        outcome = engine.run(7, sample_index)
        for record in outcome.records:
            assert record["a_bits"] == record["r_bits"]
            assert Dyadic.from_dict(record["E_align"]).is_zero()


def test_b_equals_one_boundary():
    engine = make_engine(windows=(24,), snapshots=(1,))
    outcome = engine.run(1, 0)
    assert outcome.blocks_completed == 1
    assert len(outcome.records) == 1
    record = outcome.records[0]
    assert record["B"] == 1 and record["K"] == 16
    assert record["S"] is not None


def test_prefix_consistency_across_bmax():
    long_engine = make_engine(windows=(24,), snapshots=(1, 2, 4))
    short_engine = make_engine(windows=(24,), snapshots=(1, 2))
    long_records = {r["B"]: r for r in long_engine.run(3, 0).records}
    short_records = {r["B"]: r for r in short_engine.run(3, 0).records}
    for b in (1, 2):
        assert long_records[b] == short_records[b]


def test_same_w_and_b_across_drivers_is_identical():
    width_engine = make_engine(windows=(18, 32), snapshots=(4,), mode="width_sweep")
    block_engine = make_engine(windows=(32,), snapshots=(1, 2, 4), mode="block_sweep")
    width_record = width_engine.run(5, 1).records[1]
    block_record = block_engine.run(5, 1).records[-1]
    assert width_record["W"] == block_record["W"] == 32
    assert width_record["B"] == block_record["B"] == 4
    for key in ("a_bits", "r_bits", "S", "L1", "E_align", "E_acc", "events_cumulative", "events_endpoint", "input_fingerprint"):
        assert width_record[key] == block_record[key], key


def test_snapshots_only_emitted_at_requested_prefixes():
    engine = make_engine(windows=(24,), snapshots=(2, 3))
    outcome = engine.run(1, 0)
    assert [r["B"] for r in outcome.records] == [2, 3]


def test_tf32_engine_smoke():
    engine = make_engine(fmt_name="tf32", dist="outlier", windows=(30,), snapshots=(2,))
    outcome = engine.run(2, 0)
    assert outcome.status == "ok"
    record = outcome.records[0]
    assert record["format"] == "tf32"
    a = Fp32.from_bits(record["a_bits"]).exact_dyadic()
    s = Dyadic.from_dict(record["S"])
    assert a - s == Dyadic.from_dict(record["E_align"]) + Dyadic.from_dict(record["E_acc"])
