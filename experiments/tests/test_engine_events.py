from mac_experiment.execution.engine import SampleEngine, zero_events
from mac_experiment.formats import get_format
from mac_experiment.metrics.aggregate import SummaryBuilder
from mac_experiment.reduction import WindowOverflowError
from mac_experiment.sampling.sampler import Sampler


class AbortingReducer:
    def __init__(self, abort_w):
        self.abort_w = abort_w

    def reduce(self, terms, w):
        if w == self.abort_w:
            raise WindowOverflowError(total=1 << 80, lo=-1, hi=1, w=w)
        from mac_experiment.reduction import BlockReducer

        return BlockReducer(window_offset=14).reduce(terms, w)


def test_window_overflow_aborts_only_the_affected_cell():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "laplace")
    engine = SampleEngine(fmt, sampler, windows=(18, 19), snapshots=(2,), mode="width_sweep")
    engine.reducer = AbortingReducer(abort_w=18)
    outcome = engine.run(1, 0)
    assert outcome.status == "aborted_window_overflow"
    records = outcome.records
    abort_records = [r for r in records if r["record_type"] == "abort"]
    snapshot_records = [r for r in records if r["record_type"] == "snapshot"]
    assert len(abort_records) == 1
    assert abort_records[0]["W"] == 18
    assert abort_records[0]["status"] == "aborted_window_overflow"
    assert len(snapshot_records) == 1
    assert snapshot_records[0]["W"] == 19
    assert snapshot_records[0]["status"] == "ok"


def test_abort_records_are_aggregated_as_events():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "laplace")
    engine = SampleEngine(fmt, sampler, windows=(18,), snapshots=(2,), mode="width_sweep")
    engine.reducer = AbortingReducer(abort_w=18)
    builder = SummaryBuilder("width_sweep", "W")
    outcome = engine.run(1, 0)
    for record in outcome.records:
        builder.add(record)
    summary = builder.finalize({"run_id": "test", "fixed": {"B": 2}})
    cell = summary["cells"][0]
    assert cell["counts"]["aborted"] == 1
    assert cell["counts"]["snapshots"] == 0
    assert cell["events"]["window_overflow"] == 1


def test_nonfinite_sample_marks_metric_undefined():
    fmt = get_format("fp8_e4m3")
    sampler = Sampler(fmt, "laplace")
    engine = SampleEngine(fmt, sampler, windows=(18,), snapshots=(1,), mode="width_sweep")
    outcome = engine.run(1, 0)
    record = dict(outcome.records[0])
    e_acc_value = dict(record["E_acc"])
    record["a_bits"] = 0x7F800000
    record["a_finite"] = False
    record["E_acc"] = None
    record["E_acc_complete"] = False
    record["E_acc_prefix"] = e_acc_value
    record["E_acc_prefix_until_block"] = 0
    record["status"] = "nonfinite_fp32"
    builder = SummaryBuilder("width_sweep", "W")
    builder.add(record)
    summary = builder.finalize({"run_id": "test", "fixed": {"B": 1}})
    cell = summary["cells"][0]
    assert cell["counts"]["nonfinite"] == 1
    assert cell["comparisons"]["total"]["status"] == "no_valid_samples"
    assert cell["comparisons"]["window_effect"]["status"] == "no_valid_samples"
    assert cell["comparisons"]["fp32_baseline"]["status"] == "ok"
    assert cell["comparisons"]["total"]["metrics"]["p99_abs_error"]["value"] is None
    assert cell["comparisons"]["fp32_baseline"]["metrics"]["p99_abs_error"]["value"] is not None
    assert cell["comparison_statuses"]["total"] == "no_valid_samples"
    assert cell["stage"]["E_acc"]["status"] == "no_valid_samples"


def test_zero_events_keys():
    assert set(zero_events()) == {
        "window_overflow",
        "non_finite_fp32",
        "unexpected_zero",
        "alignment_zeroed",
        "update_absorbed",
        "all_zero_block",
    }
