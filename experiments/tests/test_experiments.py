import yaml

from mac_experiment.experiments import block_sweep, width_sweep
from mac_experiment.experiments.config import WIDTH_SCAN
from mac_experiment.results.reader import iter_records, load_summary


def run_config(tmp_path, **overrides):
    config = {
        "mode": "width_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": 1,
        "B": 2,
        "output_root": str(tmp_path / "results"),
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_width_sweep_scan_is_complete_and_shared(tmp_path):
    run_dir = width_sweep.run_from_file(run_config(tmp_path), verbose=False)
    summary = load_summary(run_dir)
    assert summary["x_name"] == "W"
    assert summary["x_values"] == WIDTH_SCAN
    assert summary["fixed"] == {"B": 2}
    records = list(iter_records(run_dir))
    assert len(records) == len(WIDTH_SCAN)
    fingerprints = {r["input_fingerprint"] for r in records}
    assert len(fingerprints) == 1
    total_sums = {r["S"]["m"] for r in records}
    assert len(total_sums) == 1


def test_fp32_baseline_is_w_invariant(tmp_path):
    run_dir = width_sweep.run_from_file(run_config(tmp_path), verbose=False)
    summary = load_summary(run_dir)
    cells = summary["cells"]
    values = [
        cell["comparisons"]["fp32_baseline"]["metrics"]["p99_abs_error"]["value"] for cell in cells
    ]
    assert len(set(values)) == 1
    match_rates = [
        cell["comparisons"]["fp32_baseline"]["metrics"]["match_rate"]["value"] for cell in cells
    ]
    assert len(set(match_rates)) == 1


def test_block_sweep_structure_and_prefixes(tmp_path):
    run_dir = block_sweep.run_from_file(
        run_config(tmp_path, mode="block_sweep", B=None, W=24, B_list=[1, 2, 4, 8]),
        verbose=False,
    )
    summary = load_summary(run_dir)
    assert summary["x_name"] == "B"
    assert summary["x_values"] == [1, 2, 4, 8]
    assert summary["fixed"] == {"W": 24}
    records = list(iter_records(run_dir))
    by_b = {}
    for record in records:
        by_b.setdefault(record["B"], []).append(record)
    assert sorted(by_b) == [1, 2, 4, 8]
    for b, group in by_b.items():
        assert all(r["K"] == 16 * b for r in group)
        assert len({r["input_fingerprint"] for r in group}) == 1


def test_block_sweep_cancellation_records_rho(tmp_path):
    run_dir = block_sweep.run_from_file(
        run_config(
            tmp_path,
            mode="block_sweep",
            distributions=["cancellation"],
            B=None,
            W=24,
            B_list=[1, 4],
            record_block_rho=True,
        ),
        verbose=False,
    )
    for record in iter_records(run_dir):
        assert record["rho"] is not None
        assert record["rho"]["blocks"] == record["B"]
        assert 0.0 <= record["rho"]["min"] <= record["rho"]["max"] <= 1.0
        assert len(record["rho_per_block"]) == record["B"]
        assert min(record["rho_per_block"]) == record["rho"]["min"]


def test_summary_events_and_stage_present(tmp_path):
    run_dir = width_sweep.run_from_file(run_config(tmp_path), verbose=False)
    summary = load_summary(run_dir)
    for cell in summary["cells"]:
        assert set(cell["events"]) >= {
            "window_overflow",
            "non_finite_fp32",
            "unexpected_zero",
            "alignment_zeroed",
            "update_absorbed",
            "all_zero_block",
        }
        assert cell["stage"]["E_align"]["n"] == 1
        assert cell["stage"]["E_acc"]["n"] == 1
        assert cell["counts"]["snapshots"] == 1
