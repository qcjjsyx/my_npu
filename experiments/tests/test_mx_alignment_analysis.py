"""Driver/analysis tests for the fixed-W_align K/B sweep (source reuse, offline
metrics, non-finite policy, transitions, schema isolation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.results.mx_alignment_analysis import AlignedCell, analyze_mx_alignment_run
from mac_experiment.results.mx_io import iter_mx_samples
from mac_experiment.results.reader import load_manifest, load_summary, require_schema
from mac_experiment.version import MX_ALIGN_SCHEMA_VERSION, MX_SCHEMA_VERSION

GROUP = 32
W = 21


def _dyadic(value: Dyadic) -> dict:
    return value.to_dict()


def _snapshot_record(
    sample_id,
    seed,
    pair_index,
    K,
    S,
    T,
    L1,
    r_bits,
    a_bits,
    M=1,
    N=32,
    **extra,
):
    events = {
        "products_total": 16 * M * N,
        "products_nonzero": 16 * M * N,
        "products_changed": 0,
        "products_zeroed": 0,
        "dot16_total": M * N,
        "dot16_all_zero": 0,
        "dot16_with_changed_product": 0,
        "dot16_qb_diff": 0,
    }
    events.update(extra.pop("events_cumulative", {}))
    return {
        "record_type": "align_snapshot",
        "M": M,
        "N": N,
        "K": K,
        "B": K // 16,
        "group_length": GROUP,
        "mac_parallelism": 16,
        "w_align": W,
        "container_bits": W + 7,
        "sample_id": sample_id,
        "seed": seed,
        "pair_index": pair_index,
        "activation_distribution": "gaussian",
        "weight_distribution": "gaussian",
        "S": [_dyadic(v) for v in S],
        "L1": [_dyadic(v) for v in L1],
        "T": [_dyadic(v) for v in T],
        "r_bits": list(r_bits),
        "a_bits": list(a_bits),
        "r_nonfinite_count": 0,
        "a_nonfinite_count": 0,
        "events_cumulative": events,
        "cmax_minus_ci_histogram_cumulative": extra.pop("cmax_minus_ci_histogram_cumulative", {"0": 16}),
        "align_abs_sum": [_dyadic(Dyadic.zero())] * (M * N),
        "align_abs_max": extra.pop("align_abs_max", [_dyadic(Dyadic.zero())] * (M * N)),
        "status": "ok",
    }


def test_analysis_transitions_and_signed_zero_and_cumulative():
    cell = AlignedCell("gaussian", "gaussian", K=32, B=2)
    values = [Dyadic(1, 0)] * GROUP + [Dyadic.zero()] * GROUP
    a_values = [Dyadic(11, -1)] * GROUP + [Dyadic.zero()] * GROUP
    r_bits = [Fp32.from_bits(0x3F800000).bits] * GROUP + [0x80000000] * GROUP
    a_bits = [0x3F8CCCCD] * GROUP + [0x00000000] * GROUP
    record = _snapshot_record(
        "s1",
        1,
        0,
        32,
        values,
        values,
        [Dyadic(1, 0)] * (2 * GROUP),
        r_bits,
        a_bits,
        M=1,
        N=2 * GROUP,
        events_cumulative={"dot16_total": 64, "dot16_qb_diff": 3},
        align_abs_max=[_dyadic(Dyadic(1, -20))] * (2 * GROUP),
    )
    cell.add(record, True, None, {})
    payload = cell.finalize()
    transitions = payload["transitions"]["groups"]
    assert transitions["denominator"] == 2
    assert transitions["baseline_match_candidate_mismatch"] == 1  # 1.0 -> 1.1 code change
    assert transitions["baseline_mismatch_candidate_match"] == 1  # -0 vs +0 in Q_R
    assert payload["transitions"]["elements"]["denominator"] == 64
    assert payload["transitions"]["elements"]["baseline_match_candidate_mismatch"] == GROUP
    assert payload["transitions"]["elements"]["baseline_mismatch_candidate_match"] == GROUP
    mx = payload["mx_comparisons"]["mx_R_vs_S_independent"]
    assert mx["counts"]["n_element_code_only_match"] == GROUP  # +0 code equals +0 code
    assert mx["counts"]["n_decoded_match"] == 2 * GROUP
    assert mx["counts"]["n_element_representation_match"] == GROUP
    assert payload["alignment_diagnostics"]["events_cumulative"]["dot16_qb_diff"] == 3
    assert payload["identity_satisfied"] is True
    assert len(payload["alignment_diagnostics"]["by_pair"]) == 1


def test_analysis_nonfinite_keeps_T_minus_S_and_marks_candidate_invalid():
    cell = AlignedCell("gaussian", "gaussian", K=32, B=2)
    values = [Dyadic(1, 0)] * GROUP
    inf_bits = 0x7F800000
    record = _snapshot_record(
        "s1", 1, 0, 32, values, values, [Dyadic(1, 0)] * GROUP, [inf_bits] * GROUP, [inf_bits] * GROUP
    )
    cell.add(record, True, None, {})
    payload = cell.finalize()
    assert payload["comparisons"]["align_T_minus_S"]["status"] == "ok"
    assert payload["comparisons"]["align_T_minus_S"]["metrics"]["relative_l2"]["status"] == "ok"
    assert payload["comparisons"]["baseline_R_minus_S"]["status"] == "no_valid_samples"
    assert payload["comparisons"]["candidate_A_minus_T"]["status"] == "no_valid_samples"
    assert payload["total_errors"]["decode_QA_minus_S"]["status"] == "no_valid_samples"
    assert payload["identity_satisfied"] is True


def test_analysis_scale_only_group_and_conditional_ulp():
    cell = AlignedCell("gaussian", "gaussian", K=32, B=2)
    S = [Dyadic(1, -1)] * GROUP + [Dyadic.zero()] * GROUP
    T = list(S)
    L1 = [Dyadic(1, -1)] * (2 * GROUP)
    r_bits = [0x3E800000] * GROUP + [0x80000000] * GROUP  # 0.25 and -0
    a_bits = [0x3E800000] * GROUP + [0x00000000] * GROUP  # 0.25 and +0
    record = _snapshot_record("s1", 1, 0, 32, S, T, L1, r_bits, a_bits, M=1, N=2 * GROUP)
    cell.add(record, True, None, {})
    payload = cell.finalize()
    mx = payload["mx_comparisons"]["mx_R_vs_S_independent"]
    assert mx["counts"]["n_scale_mismatch"] == 1
    assert mx["counts"]["n_element_representation_match"] == 0
    assert mx["counts"]["n_element_code_only_match"] == GROUP  # +0 code matches +0 code
    assert mx["counts"]["n_decoded_match"] == GROUP  # -0 decodes equal to +0
    ulp = mx["distributions"]["ulp"]
    assert ulp["n_condition_elements"] == GROUP  # only the equal-scale group
    assert ulp["n_mismatch"] == GROUP
    assert ulp["p50"] == 0 and ulp["n_signed_zero_mismatch"] == GROUP
    same = payload["mx_comparisons"]["mx_A_vs_R_independent"]
    # A and R are numerically equal; only the signed zero code differs
    assert same["metrics"]["group_representation_match_rate"]["value"] == 0.5
    assert same["counts"]["n_decoded_match"] == 2 * GROUP
    transitions = payload["transitions"]["groups"]
    assert transitions["denominator"] == 2
    assert transitions["baseline_mismatch_candidate_match"] == 1  # -0 -> +0
    assert transitions["both_mismatch_candidate_same"] == 1  # 0.5 vs 0.25 scale-only

    # a snapshot whose only group has different scales has no conditional ULP samples
    scale_only = AlignedCell("gaussian", "gaussian", K=32, B=2)
    S = [Dyadic(1, -1)] * GROUP
    record = _snapshot_record(
        "s2",
        1,
        0,
        32,
        S,
        list(S),
        [Dyadic(1, -1)] * GROUP,
        [0x3E800000] * GROUP,
        [0x3E800000] * GROUP,
    )
    scale_only.add(record, True, None, {})
    only = scale_only.finalize()["mx_comparisons"]["mx_R_vs_S_independent"]
    assert only["distributions"]["ulp"]["status"] == "no_mismatch_samples"
    assert only["distributions"]["ulp"]["p50"] is None
    assert only["distributions"]["ulp"]["n_condition_elements"] == 0


def _write_mx_config(tmp_path, **overrides):
    config = {
        "mode": "mx_matrix_sweep",
        "M": 2,
        "N": 32,
        "K_list": [32, 64],
        "distribution_pairs": [
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"},
        ],
        "seeds": [1],
        "pairs_per_seed": 2,
        "output_root": str(tmp_path / "mx_runs"),
        "run_name": "src_mx",
    }
    config.update(overrides)
    path = tmp_path / "mx_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_align_config(tmp_path, source_run, **overrides):
    config = {
        "mode": "mx_alignment_sweep",
        "M": 2,
        "N": 32,
        "K_list": [32, 64],
        "W_align": W,
        "distribution_pairs": [
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"},
        ],
        "seeds": [1],
        "pairs_per_seed": 2,
        "source_run": str(source_run),
        "output_root": str(tmp_path / "align_runs"),
        "run_name": "align_reuse",
    }
    config.update(overrides)
    path = tmp_path / "align_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reuse_source_end_to_end_read_only(tmp_path):
    from mac_experiment.experiments.mx_alignment_sweep import run_from_file as align_run
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)
    before = {
        name: _digest(source / name)
        for name in ("manifest.json", "inputs.jsonl.gz", "samples.jsonl.gz")
    }
    align_dir = align_run(_write_align_config(tmp_path, source), verbose=False)
    manifest = load_manifest(align_dir)
    require_schema(manifest, (MX_ALIGN_SCHEMA_VERSION,))
    assert manifest["status"] == "completed"
    assert manifest["counts"]["matrix_pairs"] == 2
    assert manifest["counts"]["snapshot_records"] == 4
    assert manifest["source"]["kind"] == "reused"
    assert set(manifest["source"]["source_file_identity"]) == {
        "manifest.json",
        "inputs.jsonl.gz",
        "samples.jsonl.gz",
    }
    assert manifest["source"]["source_file_identity"]["samples.jsonl.gz"]["sha256"].startswith("sha256:")
    after = {name: _digest(source / name) for name in before}
    assert before == after

    out = analyze_mx_alignment_run(align_dir, out_dir=tmp_path / "analysis", write_plots=False)
    summary = load_summary(out)
    assert summary["schema_version"] == MX_ALIGN_SCHEMA_VERSION
    assert summary["w_align"] == W
    assert summary["cells"]
    assert summary["source"]["source_run_id"] == "src_mx"
    for cell in summary["cells"]:
        assert cell["identity_satisfied"]
        assert cell["alignment_diagnostics"]["events_cumulative"]["dot16_total"] > 0
        assert len(cell["alignment_diagnostics"]["by_pair"]) == 2
    with pytest.raises(ValueError):
        require_schema(load_manifest(align_dir), (MX_SCHEMA_VERSION,))


def test_reuse_detects_tampered_source_snapshot(tmp_path):
    import shutil

    from mac_experiment.experiments.mx_alignment_sweep import run_from_file as align_run
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)
    tampered = tmp_path / "tampered_src"
    shutil.copytree(source, tampered)
    import gzip

    path = tampered / "samples.jsonl.gz"
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["K"] == 32 and record["pair_index"] == 0:
                payload = dict(record["S"][0])
                payload["m"] = str(int(payload["m"]) + 1)
                record["S"][0] = payload
            records.append(record)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(ValueError, match="S mismatch"):
        align_run(_write_align_config(tmp_path, tampered), verbose=False)


def test_reuse_prefix_subset_and_analysis_cumulative_denominators(tmp_path):
    from mac_experiment.experiments.mx_alignment_sweep import run_from_file as align_run
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)
    align_dir = align_run(
        _write_align_config(tmp_path, source, K_list=[32]), verbose=False
    )
    manifest = load_manifest(align_dir)
    assert manifest["K_list"] == [32]
    assert manifest["counts"]["snapshot_records"] == 2
    snapshots = list(iter_mx_samples(align_dir))
    assert {int(s["K"]) for s in snapshots} == {32}

    out = analyze_mx_alignment_run(align_dir, out_dir=tmp_path / "analysis", write_plots=False)
    summary = load_summary(out)
    cell = summary["cells"][0]
    assert len(cell["alignment_diagnostics"]["by_pair"]) == 2
    # cumulative counters equal the snapshot value (not a sum over K)
    total = sum(
        entry["alignment_diagnostics"]["events_cumulative"]["dot16_qb_diff"]
        for entry in cell["alignment_diagnostics"]["by_pair"]
    )
    assert total == cell["alignment_diagnostics"]["events_cumulative"]["dot16_qb_diff"]
    # per-pair events equal the saved snapshot events at that K
    saved = {(s["sample_id"], int(s["K"])): s for s in snapshots}
    for entry in cell["alignment_diagnostics"]["by_pair"]:
        record = saved[(entry["sample_id"], 32)]
        for key, value in entry["alignment_diagnostics"]["events_cumulative"].items():
            assert value == record["events_cumulative"][key]


def test_alignment_config_rejections(tmp_path):
    from mac_experiment.experiments.config import load_config

    source = tmp_path / "does_not_matter"
    path = _write_align_config(tmp_path, source)
    config = load_config(path)
    assert config.W_align == W
    assert config.source_run == str(source)

    for bad in ({"W_align": 0}, {"W_align": 61}, {"scale_scenarios": ["unit"]}):
        path = _write_align_config(tmp_path, source, **bad)
        with pytest.raises(ValueError):
            load_config(path)

    path = _write_align_config(
        tmp_path,
        source,
        distribution_pairs=[
            {
                "activation_distribution": "gaussian",
                "weight_distribution": "gaussian",
                "activation_parameters": {"mean": 1.0},
            }
        ],
    )
    with pytest.raises(ValueError, match="custom distribution parameters"):
        load_config(path)


def test_reuse_rejects_mismatched_source_identity(tmp_path):
    from mac_experiment.experiments.mx_alignment_sweep import run_from_file as align_run
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)
    path = _write_align_config(
        tmp_path,
        source,
        distribution_pairs=[
            {"activation_distribution": "laplace", "weight_distribution": "laplace"},
        ],
    )
    with pytest.raises(ValueError, match="distribution_pairs must equal"):
        align_run(path, verbose=False)
    path = _write_align_config(tmp_path, source, K_list=[32, 64, 128])
    with pytest.raises(ValueError, match="subset"):
        align_run(path, verbose=False)


# ---------------------------------------------------------------------------
# review-fix regressions
# ---------------------------------------------------------------------------


def _write_align_fresh_config(tmp_path, **overrides):
    config = {
        "mode": "mx_alignment_sweep",
        "M": 2,
        "N": 32,
        "K_list": [32, 64],
        "W_align": W,
        "distribution_pairs": [
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"},
        ],
        "seeds": [1],
        "pairs_per_seed": 2,
        "output_root": str(tmp_path / "fresh_align_runs"),
        "run_name": "align_fresh_src",
    }
    config.update(overrides)
    path = tmp_path / "align_fresh_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _read_gz_jsonl(path):
    import gzip

    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_gz_jsonl(path, records):
    import gzip

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _fresh_align_run(tmp_path):
    from mac_experiment.experiments.mx_alignment_sweep import run_from_file

    return run_from_file(_write_align_fresh_config(tmp_path), verbose=False)


def test_csv_written_to_requested_path_and_cwd_stays_clean(tmp_path, monkeypatch):
    """Regression: output_path must not be shadowed by the quantization path loop."""
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    run_dir = _fresh_align_run(tmp_path)
    out = analyze_mx_alignment_run(run_dir, out_dir=tmp_path / "analysis", write_plots=False)

    csv_path = out / "summary.csv"
    assert csv_path.exists() and csv_path.stat().st_size > 0
    import csv as _csv

    with csv_path.open(encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        assert reader.fieldnames == [
            "activation_distribution",
            "weight_distribution",
            "x_name",
            "K",
            "B",
            "section",
            "metric",
            "value",
            "status",
            "denominator",
            "note",
        ]
        rows = list(reader)
    assert any(row["section"] == "cell" for row in rows)
    assert any(row["section"].startswith("quantization:Q_S_shared") for row in rows)
    for stray in ("Q_S_shared", "Q_S_independent", "Q_A_shared", "Q_R_shared"):
        assert not (workdir / stray).exists(), f"stray CSV written to cwd: {stray}"
    # (tmp_path/Q_S_shared) exists from the pre-fix artifact; the analysis must
    # never write there either.
    assert not (tmp_path / "Q_S_shared").exists()


def test_analysis_input_validation_uses_e8m0_rules(tmp_path):
    """0x7F is a legal E8M0 scale; 0xFF and E4M3 NaN are invalid but supported."""
    from mac_experiment.experiments.mx_alignment_sweep import run_from_file

    base = run_from_file(_write_align_fresh_config(tmp_path), verbose=False)

    def variant(name, transform):
        dest = tmp_path / name
        import shutil

        shutil.copytree(base, dest)
        records = _read_gz_jsonl(dest / "inputs.jsonl.gz")
        for record in records:
            transform(record)
        _write_gz_jsonl(dest / "inputs.jsonl.gz", records)
        return dest

    legal = variant(
        "legal_scale",
        lambda r: r["x"].__setitem__("scale_codes", [0x7F] * len(r["x"]["scale_codes"])),
    )
    out = analyze_mx_alignment_run(legal, out_dir=tmp_path / "a_legal", write_plots=False)
    summary = load_summary(out)
    for cell in summary["cells"]:
        assert cell["counts"]["invalid_input_snapshots"] == 0

    zeros = variant(
        "all_zero",
        lambda r: (
            r["x"].__setitem__("element_codes", [0x00] * len(r["x"]["element_codes"])),
            r["x"].__setitem__("scale_codes", [0x7F] * len(r["x"]["scale_codes"])),
        ),
    )
    out = analyze_mx_alignment_run(zeros, out_dir=tmp_path / "a_zero", write_plots=False)
    summary = load_summary(out)
    assert all(cell["counts"]["invalid_input_snapshots"] == 0 for cell in summary["cells"])

    reserved = variant(
        "reserved_scale",
        lambda r: r["x"].__setitem__("scale_codes", [0xFF] + list(r["x"]["scale_codes"][1:])),
    )
    out = analyze_mx_alignment_run(reserved, out_dir=tmp_path / "a_reserved", write_plots=False)
    summary = load_summary(out)
    assert any(cell["counts"]["invalid_input_snapshots"] > 0 for cell in summary["cells"])

    nan_element = variant(
        "nan_element",
        lambda r: r["x"].__setitem__("element_codes", [0x7F] + list(r["x"]["element_codes"][1:])),
    )
    out = analyze_mx_alignment_run(nan_element, out_dir=tmp_path / "a_nan", write_plots=False)
    summary = load_summary(out)
    assert any(cell["counts"]["invalid_input_snapshots"] > 0 for cell in summary["cells"])


def test_invalid_pair_records_invalid_coverage_per_comparison(tmp_path):
    from mac_experiment.experiments.mx_alignment_sweep import run_from_file

    base = run_from_file(_write_align_fresh_config(tmp_path), verbose=False)
    import shutil

    tampered = tmp_path / "invalid_input"
    shutil.copytree(base, tampered)
    records = _read_gz_jsonl(tampered / "inputs.jsonl.gz")
    victim = next(r for r in records if r["pair_index"] == 0)
    victim["x"]["scale_codes"] = [0xFF] + list(victim["x"]["scale_codes"][1:])
    _write_gz_jsonl(tampered / "inputs.jsonl.gz", records)

    out = analyze_mx_alignment_run(tampered, out_dir=tmp_path / "analysis", write_plots=False)
    summary = load_summary(out)
    for cell in summary["cells"]:
        assert cell["counts"]["invalid_input_snapshots"] == 1
        assert cell["counts"]["snapshots"] == 2
        for name in ("align_T_minus_S", "baseline_R_minus_S", "candidate_A_minus_S"):
            payload = cell["comparisons"][name]
            assert payload["coverage"]["n_invalid"] == 2 * 32  # one pair x (M=2,N=32)
            assert payload["coverage"]["n_total"] == 2 * 64
            assert payload["status"] == "partial_nonfinite"
            assert all(entry["value"] is None for entry in payload["metrics"].values())
            assert payload["finite_subset"] is not None
        mx = cell["mx_comparisons"]["mx_R_vs_S_independent"]
        assert mx["coverage"]["invalid_groups"] == 2
        assert mx["coverage"]["groups"] == 4
        transitions = cell["transitions"]
        assert transitions["coverage"]["invalid_groups"] == 2
        assert transitions["coverage"]["total_groups"] == 4
        assert transitions["groups"]["total_denominator"] == 4
        pair = next(
            entry for entry in cell["alignment_diagnostics"]["by_pair"] if not entry["valid"]
        )
        assert pair["invalid_reason"]
        assert pair["comparisons"]["align_T_minus_S"]["coverage"]["n_invalid"] == 64
    # CSV keeps both whole-batch nulls and the finite subset semantics
    csv_text = (out / "summary.csv").read_text(encoding="utf-8")
    assert "scope:finite_subset" in csv_text
    assert "finite_subset_status" in csv_text
    assert "partial_nonfinite" in csv_text


def test_snapshot_completeness_checks(tmp_path):
    import shutil

    base = _fresh_align_run(tmp_path)

    def variant(name, transform):
        dest = tmp_path / name
        shutil.copytree(base, dest)
        records = _read_gz_jsonl(dest / "samples.jsonl.gz")
        _write_gz_jsonl(dest / "samples.jsonl.gz", transform(records))
        return dest

    cases = {
        "drop_first": lambda rs: rs[1:],
        "drop_last": lambda rs: rs[:-1],
        "drop_one_pair_k": lambda rs: [r for r in rs if not (r["sample_id"].endswith("00001") and r["K"] == 32)],
        "duplicate": lambda rs: rs + [rs[0]],
        "extra_k": lambda rs: rs + [{**rs[0], "K": 128, "B": 8}],
        "unknown_sample": lambda rs: [{**rs[0], "sample_id": "mx|gaussian|gaussian|seed9|00000"}] + rs[1:],
        "wrong_distribution": lambda rs: [{**rs[0], "weight_distribution": "laplace"}] + rs[1:],
        "wrong_w_align": lambda rs: [{**rs[0], "w_align": 20}] + rs[1:],
        "short_array": lambda rs: [{**rs[0], "r_bits": rs[0]["r_bits"][:-1]}] + rs[1:],
    }
    for name, transform in cases.items():
        tampered = variant(name, transform)
        out = tmp_path / f"analysis_{name}"
        with pytest.raises(ValueError):
            analyze_mx_alignment_run(tampered, out_dir=out, write_plots=False)
        assert not out.exists(), f"{name}: a failed analysis must not leave an output directory"

    # complete run still passes
    out = analyze_mx_alignment_run(base, out_dir=tmp_path / "analysis_ok", write_plots=False)
    assert (out / "summary.json").exists()


def test_driver_verifies_the_saved_snapshots(tmp_path, monkeypatch):
    """Fault injection into the new engine result must fail the reuse run."""
    import mac_experiment.experiments.mx_alignment_sweep as driver
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)

    for field in ("S", "L1", "r_bits"):
        original = driver.evaluate_mx_pair_aligned
        field_config = _write_align_config(
            tmp_path, source, run_name=f"align_tamper_{field}"
        )

        def tamper(*args, __field=field, **kwargs):
            result = original(*args, **kwargs)
            snapshot = result.snapshots[0]
            if __field == "r_bits":
                snapshot["r_bits"][0] = int(snapshot["r_bits"][0]) ^ 1
            else:
                from mac_experiment.arithmetic.dyadic import Dyadic as _D

                payload = dict(snapshot[__field][0])
                payload["m"] = str(int(payload["m"]) + 1)
                snapshot[__field][0] = _D.from_dict(payload).to_dict()
            return result

        monkeypatch.setattr(driver, "evaluate_mx_pair_aligned", tamper)
        with pytest.raises(ValueError) as excinfo:
            driver.run_from_file(field_config, verbose=False)
        message = str(excinfo.value)
        assert "mismatch" in message
        assert f"(field '{field}')" in message
        monkeypatch.setattr(driver, "evaluate_mx_pair_aligned", original)

    # normal reuse still passes
    run_dir = driver.run_from_file(
        _write_align_config(tmp_path, source, run_name="align_ok"), verbose=False
    )
    assert load_manifest(run_dir)["status"] == "completed"


def test_driver_rejects_duplicate_and_extra_source_records(tmp_path):
    import shutil

    from mac_experiment.experiments.mx_alignment_sweep import run_from_file as align_run
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)

    duplicate = tmp_path / "dup_src"
    shutil.copytree(source, duplicate)
    records = _read_gz_jsonl(duplicate / "samples.jsonl.gz")
    _write_gz_jsonl(
        duplicate / "samples.jsonl.gz", [records[0], records[0]] + records[1:]
    )
    with pytest.raises(ValueError, match="duplicate source snapshot"):
        align_run(
            _write_align_config(tmp_path, duplicate, run_name="a_dup"), verbose=False
        )

    conflicting = tmp_path / "conflict_src"
    shutil.copytree(source, conflicting)
    records = _read_gz_jsonl(conflicting / "samples.jsonl.gz")
    payload = dict(records[0]["S"][0])
    payload["m"] = str(int(payload["m"]) + 7)
    altered = {**records[0], "S": [payload] + records[0]["S"][1:]}
    _write_gz_jsonl(
        conflicting / "samples.jsonl.gz", [records[0], altered] + records[1:]
    )
    with pytest.raises(ValueError, match="duplicate source snapshot"):
        align_run(
            _write_align_config(tmp_path, conflicting, run_name="a_conflict"), verbose=False
        )

    extra = tmp_path / "extra_src"
    shutil.copytree(source, extra)
    inputs = _read_gz_jsonl(extra / "inputs.jsonl.gz")
    _write_gz_jsonl(extra / "inputs.jsonl.gz", inputs + [inputs[0]])
    with pytest.raises(ValueError, match="unexpected extra source input record"):
        align_run(
            _write_align_config(tmp_path, extra, run_name="a_extra"), verbose=False
        )


def test_analyzer_source_digest_and_engine_isolation(tmp_path):
    from mac_experiment.results.schema import source_digest

    run_dir = _fresh_align_run(tmp_path)
    out = analyze_mx_alignment_run(run_dir, out_dir=tmp_path / "analysis", write_plots=False)
    summary = load_summary(out)
    manifest = json.loads((out / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert summary["analyzer_source_digest"] == source_digest()
    assert summary["source_digest"] == load_manifest(run_dir).get("source_digest")
    assert manifest["analyzer_source_digest"] == summary["analyzer_source_digest"]
    assert manifest["numeric_run_source_digest"] == summary["source_digest"]

    # offline analysis must not import the execution engine or samplers
    import subprocess
    import sys

    code = (
        "import sys; import mac_experiment.results.mx_alignment_analysis as m; "
        "bad=[x for x in ('mac_experiment.matrix.engine','mac_experiment.matrix.aligned_engine',"
        "'mac_experiment.matrix.inputs','mac_experiment.experiments.common') if x in sys.modules]; "
        "print(bad)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_per_pair_merge_matches_cell_metrics(tmp_path):
    from fractions import Fraction

    from mac_experiment.arithmetic.dyadic import Dyadic as _D

    run_dir = _fresh_align_run(tmp_path)
    out = analyze_mx_alignment_run(run_dir, out_dir=tmp_path / "analysis", write_plots=False)
    summary = load_summary(out)
    for cell in summary["cells"]:
        pairs = cell["alignment_diagnostics"]["by_pair"]
        assert len(pairs) == 2
        # numeric: merge exactly from per-pair sum_err2/sum_ref2 (never average ratios)
        for name, payload in cell["comparisons"].items():
            if payload["status"] != "ok":
                continue
            sum_err2 = sum(
                (Fraction(*_dyadic_fraction(pair["comparisons"][name]["sum_err2"])) for pair in pairs),
                Fraction(0),
            )
            sum_ref2 = sum(
                (Fraction(*_dyadic_fraction(pair["comparisons"][name]["sum_ref2"])) for pair in pairs),
                Fraction(0),
            )
            expected = float((sum_err2 / sum_ref2) ** Fraction(1, 2)) if sum_ref2 else 0.0
            assert payload["metrics"]["relative_l2"]["value"] == pytest.approx(expected, rel=1e-12)
            merged_total = sum(
                pair["comparisons"][name]["coverage"]["n_total"] for pair in pairs
            )
            assert payload["coverage"]["n_total"] == merged_total
        # MX counts merge
        mx = cell["mx_comparisons"]["mx_R_vs_S_independent"]
        assert mx["coverage"]["groups"] == sum(
            pair["mx_comparisons"]["mx_R_vs_S_independent"]["coverage"]["groups"] for pair in pairs
        )
        assert mx["counts"]["n_element_representation_match"] == sum(
            pair["mx_comparisons"]["mx_R_vs_S_independent"]["counts"][
                "n_element_representation_match"
            ]
            for pair in pairs
        )
        # transitions merge with explicit valid/invalid/total denominators
        transitions = cell["transitions"]
        assert transitions["coverage"]["valid_groups"] == sum(
            pair["transitions"]["groups"]["valid_denominator"] for pair in pairs
        )
        assert transitions["coverage"]["invalid_groups"] == sum(
            pair["transitions"]["groups"]["invalid"] for pair in pairs
        )


def _dyadic_fraction(payload: dict):
    value = Dyadic.from_dict(payload)
    fraction = value.to_fraction()
    return fraction.numerator, fraction.denominator


def test_trace_entry_points_for_reuse_runs(tmp_path):
    import argparse
    import shutil

    from mac_experiment.cli import _cmd_mx_align_trace
    from mac_experiment.experiments.mx_alignment_sweep import run_from_file as align_run
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)
    align_dir = align_run(_write_align_config(tmp_path, source, K_list=[32]), verbose=False)
    sample_id = "mx|gaussian|gaussian|seed1|00000"

    def namespace(**overrides):
        base = dict(
            run=str(align_dir),
            sample_id=sample_id,
            row=0,
            col=0,
            block_start=0,
            block_end=1,
            w_align=None,
            out=None,
            source_run=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    # reuse entry point without --source-run: recorded source directory is used
    code = _cmd_mx_align_trace(namespace(out=str(tmp_path / "trace1.jsonl")))
    assert code == 0
    trace = json.loads((tmp_path / "trace1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert trace["run_w_align"] == W and trace["w_align_override"] is False
    assert trace["prefix_Kmax"] == 32
    assert trace["sample_id"] == sample_id

    # moved source directory: explicit override with identity verification
    moved = tmp_path / "moved_source"
    shutil.copytree(source, moved)
    code = _cmd_mx_align_trace(
        namespace(source_run=str(moved), out=str(tmp_path / "trace2.jsonl"))
    )
    assert code == 0

    # mismatching override is rejected
    tampered = tmp_path / "tampered_source"
    shutil.copytree(source, tampered)
    samples_path = tampered / "samples.jsonl.gz"
    data = bytearray(samples_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    samples_path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match the identity"):
        _cmd_mx_align_trace(
            namespace(source_run=str(tampered), out=str(tmp_path / "trace3.jsonl"))
        )

    # W_align override is explicit in the output
    code = _cmd_mx_align_trace(
        namespace(w_align=20, out=str(tmp_path / "trace4.jsonl"))
    )
    trace = json.loads((tmp_path / "trace4.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert trace["w_align"] == 20 and trace["w_align_override"] is True

    # block beyond the executed prefix is rejected
    with pytest.raises(ValueError, match="exceeds this run's executed K prefix"):
        _cmd_mx_align_trace(namespace(block_start=2, block_end=2))


def test_driver_requires_complete_source_k_set_per_pair(tmp_path):
    """A source run declaring K=[32,64] must really contain every pair's both K.

    Regression: with a subset request (K=32) only the requested K was checked,
    so deleting all K=64 snapshots still reported completed.
    """
    import shutil

    from mac_experiment.experiments.mx_alignment_sweep import run_from_file as align_run
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file as mx_run

    source = mx_run(_write_mx_config(tmp_path), verbose=False)
    subset_config = dict(K_list=[32], run_name="align_prefix")

    # A) delete every K=64 snapshot, then request only K=32
    dropped = tmp_path / "dropped_all_k64"
    shutil.copytree(source, dropped)
    records = [r for r in _read_gz_jsonl(dropped / "samples.jsonl.gz") if int(r["K"]) != 64]
    _write_gz_jsonl(dropped / "samples.jsonl.gz", records)
    with pytest.raises(ValueError, match="source snapshots incomplete"):
        align_run(
            _write_align_config(tmp_path, dropped, **subset_config), verbose=False
        )

    # B) delete K=64 for a single pair only
    dropped_one = tmp_path / "dropped_one_pair_k64"
    shutil.copytree(source, dropped_one)
    records = _read_gz_jsonl(dropped_one / "samples.jsonl.gz")
    victim = "mx|gaussian|gaussian|seed1|00001"
    kept = [r for r in records if not (r["sample_id"] == victim and int(r["K"]) == 64)]
    assert len(kept) == len(records) - 1
    _write_gz_jsonl(dropped_one / "samples.jsonl.gz", kept)
    with pytest.raises(ValueError, match="source snapshots incomplete"):
        align_run(
            _write_align_config(tmp_path, dropped_one, run_name="align_prefix_one", K_list=[32]),
            verbose=False,
        )

    # C) a per-pair K not declared by the source manifest is rejected as well
    extra_k = tmp_path / "extra_k"
    shutil.copytree(source, extra_k)
    records = _read_gz_jsonl(extra_k / "samples.jsonl.gz")
    rewritten = []
    for record in records:
        if record["sample_id"].endswith("00001") and int(record["K"]) == 64:
            rewritten.append({**record, "K": 512, "B": 32})
        else:
            rewritten.append(record)
    _write_gz_jsonl(extra_k / "samples.jsonl.gz", rewritten)
    with pytest.raises(ValueError, match="unexpected K=\\[512\\]"):
        align_run(
            _write_align_config(tmp_path, extra_k, run_name="align_extra_k", K_list=[32]),
            verbose=False,
        )

    # D) the legal short-prefix reuse still passes
    ok = align_run(
        _write_align_config(tmp_path, source, run_name="align_prefix_ok", K_list=[32]),
        verbose=False,
    )
    assert load_manifest(ok)["status"] == "completed"
    assert int(load_manifest(ok)["counts"]["snapshot_records"]) == 2
