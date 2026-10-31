"""MX analysis tests: comparisons, ULP conditions, denominators, offline runs."""

from __future__ import annotations

import pytest
import yaml

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.formats.e4m3_element import decode_element
from mac_experiment.matrix.quantize import quantize_matrix_independent, quantize_matrix_shared
from mac_experiment.metrics.mx_quant import main_accumulator, shared_accumulator
from mac_experiment.results.mx_analysis import analyze_mx_run, write_mx_csv
from mac_experiment.results.mx_io import MX_INPUTS_NAME, MX_SAMPLES_NAME, iter_mx_inputs, iter_mx_samples
from mac_experiment.results.reader import load_manifest, load_summary, require_schema
from mac_experiment.results.schema import loads_strict
from mac_experiment.version import MX_SCHEMA_VERSION

DIST = "uniform"
SCENARIO = "unit"
GROUP = 32
CODE_ONE = 0x38  # E4M3 1.0


def _decoded(codes, exponent):
    values = []
    for code in codes:
        element = decode_element(code)
        values.append(element if element.is_zero() else Dyadic(element.m, element.e + exponent))
    return values


def _independent(values, rows=1):
    return quantize_matrix_independent(values, None, rows, GROUP)


def _feed(acc, values_s, values_r, rows=1):
    q_s = _independent(values_s, rows)
    q_r = _independent(values_r, rows)
    for group_index in range(len(q_s.groups)):
        g_s = q_s.groups[group_index]
        g_r = q_r.groups[group_index]
        from mac_experiment.matrix.quantize import decoded_group_values

        acc.add_group(
            "sample",
            1,
            g_s.scale_code,
            g_r.scale_code,
            g_s.element_codes,
            g_r.element_codes,
            decoded_group_values(g_s),
            decoded_group_values(g_r),
        )
    return q_s, q_r


def test_code_mismatch_but_decoded_match_signed_zero():
    acc = main_accumulator()
    values = [Dyadic.zero()] * GROUP
    signs_r = [1] * GROUP
    q_s = quantize_matrix_independent(values, None, 1, GROUP)
    q_r = quantize_matrix_independent(values, signs_r, 1, GROUP)
    acc.add_group(
        "sample",
        1,
        q_s.groups[0].scale_code,
        q_r.groups[0].scale_code,
        q_s.groups[0].element_codes,
        q_r.groups[0].element_codes,
        [Dyadic.zero()] * GROUP,
        [Dyadic.zero()] * GROUP,
    )
    payload = acc.finalize()
    assert payload["status"] == "ok"
    assert payload["counts"]["n_element_code_only_match"] == 0
    assert payload["counts"]["n_element_representation_match"] == 0
    assert payload["counts"]["n_decoded_match"] == GROUP
    representation = payload["distributions"]["group_representation_mismatch_elements"]
    code_only = payload["distributions"]["group_code_only_mismatch_elements"]
    assert representation["histogram"]["32"] == 1
    assert code_only["histogram"]["32"] == 1
    assert payload["counts"]["n_group_bit_match"] == 0
    assert payload["counts"]["n_group_code_only_match"] == 0
    ulp = payload["distributions"]["ulp"]
    assert ulp["n_mismatch"] == GROUP
    assert ulp["n_signed_zero_mismatch"] == GROUP
    assert ulp["p50"] == 0 and ulp["max"] == 0


def test_scale_mismatch_excludes_main_ulp_but_aux_keeps_it():
    values_s = [Dyadic(1, -1)] * GROUP  # max 0.5
    values_r = [Dyadic(1, 0)] * GROUP  # max 1.0
    q_s = _independent(values_s)
    q_r = _independent(values_r)
    assert q_s.groups[0].scale_code != q_r.groups[0].scale_code

    main = main_accumulator()
    main.add_group(
        "sample",
        1,
        q_s.groups[0].scale_code,
        q_r.groups[0].scale_code,
        q_s.groups[0].element_codes,
        q_r.groups[0].element_codes,
        [Dyadic(1, -1)] * GROUP,
        [Dyadic(1, 0)] * GROUP,
    )
    payload = main.finalize()
    assert payload["counts"]["n_scale_mismatch"] == 1
    assert payload["counts"]["n_element_code_only_match"] == GROUP  # same relative code
    assert payload["counts"]["n_element_representation_match"] == 0
    representation = payload["distributions"]["group_representation_mismatch_elements"]
    code_only = payload["distributions"]["group_code_only_mismatch_elements"]
    assert representation["histogram"]["32"] == 1
    assert code_only["histogram"]["0"] == 1
    assert payload["counts"]["n_group_bit_match"] == 0
    assert payload["counts"]["n_group_code_only_match"] == 1
    ulp = payload["distributions"]["ulp"]
    assert ulp["n_condition_elements"] == 0
    assert ulp["status"] == "no_mismatch_samples"
    assert ulp["p50"] is None
    assert payload["metrics"]["scale_mismatch_rate"]["value"] == 1.0
    assert payload["metrics"]["group_representation_mismatch_rate"]["value"] == 1.0
    assert payload["metrics"]["group_code_only_mismatch_rate"]["value"] == 0.0

    aux = shared_accumulator()
    shared = quantize_matrix_shared(values_r, None, 1, GROUP, [q_s.groups[0].scale_code])
    from mac_experiment.matrix.quantize import decoded_group_values

    aux.add_group(
        "sample",
        1,
        q_s.groups[0].scale_code,
        shared.groups[0].scale_code,
        q_s.groups[0].element_codes,
        shared.groups[0].element_codes,
        decoded_group_values(q_s.groups[0]),
        decoded_group_values(shared.groups[0]),
    )
    aux_payload = aux.finalize()
    assert aux_payload["status"] == "ok"
    assert aux_payload["counts"]["n_scale_match"] == 1
    assert aux_payload["counts"]["n_element_code_only_match"] == 0
    assert aux_payload["distributions"]["ulp"]["status"] == "ok"
    assert aux_payload["distributions"]["ulp"]["n_mismatch"] == GROUP


def test_no_mismatch_ulp_is_null_not_zero():
    values = [Dyadic(3, -2)] * GROUP
    acc = main_accumulator()
    _feed(acc, values, values)
    payload = acc.finalize()
    ulp = payload["distributions"]["ulp"]
    assert ulp["status"] == "no_mismatch_samples"
    assert ulp["p50"] is None and ulp["p99"] is None and ulp["max"] is None
    assert payload["metrics"]["decoded_match_rate"]["value"] == 1.0


def test_invalid_group_partial_policy_and_denominators():
    acc = main_accumulator()
    values = [Dyadic(1, 0)] * GROUP
    _feed(acc, values, values)
    acc.add_invalid_group("sample", 1, GROUP, "nonfinite_value")
    payload = acc.finalize()
    assert payload["status"] == "partial_invalid_groups"
    assert payload["coverage"]["groups"] == 2
    assert payload["coverage"]["valid_groups"] == 1
    assert payload["coverage"]["invalid_groups"] == 1
    assert payload["coverage"]["invalid_elements"] == GROUP
    assert payload["events"]["invalid_group_reasons"] == {"nonfinite_value": 1}
    assert all(entry["value"] is None for entry in payload["metrics"].values())
    subset = payload["finite_subset"]
    assert subset["metrics"]["decoded_match_rate"]["value"] == 1.0
    assert subset["metrics"]["scale_mismatch_rate"]["value"] == 0.0


def test_no_valid_groups_status():
    acc = main_accumulator()
    acc.add_invalid_group("sample", 1, GROUP, "reserved_scale_code")
    payload = acc.finalize()
    assert payload["status"] == "no_valid_groups"
    assert all(entry["value"] is None for entry in payload["metrics"].values())


def test_same_codes_different_scale_is_representation_mismatch_only():
    """code-only mismatch 0, representation mismatch 32, decoded values differ."""
    codes = [CODE_ONE] * GROUP
    scale_s, scale_r = 118, 119  # exponents -9 and -8
    acc = main_accumulator()
    acc.add_group(
        "sample",
        1,
        scale_s,
        scale_r,
        codes,
        list(codes),
        _decoded(codes, -9),
        _decoded(codes, -8),
    )
    payload = acc.finalize()
    assert payload["counts"]["n_element_code_only_match"] == GROUP
    assert payload["counts"]["n_element_representation_match"] == 0
    assert payload["counts"]["n_decoded_match"] == 0
    representation = payload["distributions"]["group_representation_mismatch_elements"]
    code_only = payload["distributions"]["group_code_only_mismatch_elements"]
    assert representation["histogram"]["32"] == 1
    assert code_only["histogram"]["0"] == 1
    assert payload["counts"]["n_group_bit_match"] == 0
    assert payload["counts"]["n_group_code_only_match"] == 1
    assert payload["metrics"]["group_representation_mismatch_rate"]["value"] == 1.0
    assert payload["metrics"]["group_code_only_mismatch_rate"]["value"] == 0.0
    assert payload["metrics"]["decoded_match_rate"]["value"] == 0.0
    # ULP is only defined on a common scale grid: here the scales differ
    assert payload["distributions"]["ulp"]["n_condition_elements"] == 0
    assert payload["distributions"]["ulp"]["status"] == "no_mismatch_samples"


def test_same_scale_one_code_different_is_one_representation_mismatch():
    codes_s = [CODE_ONE] * GROUP
    codes_r = list(codes_s)
    codes_r[7] = 0x39  # 1.125
    acc = main_accumulator()
    acc.add_group(
        "sample",
        1,
        127,
        127,
        codes_s,
        codes_r,
        _decoded(codes_s, 0),
        _decoded(codes_r, 0),
    )
    payload = acc.finalize()
    representation = payload["distributions"]["group_representation_mismatch_elements"]
    assert representation["histogram"]["1"] == 1
    assert payload["counts"]["n_element_representation_match"] == GROUP - 1
    assert payload["counts"]["n_element_code_only_match"] == GROUP - 1
    assert payload["counts"]["n_decoded_match"] == GROUP - 1
    assert payload["counts"]["n_group_bit_match"] == 0
    assert payload["distributions"]["ulp"]["n_mismatch"] == 1
    assert payload["distributions"]["ulp"]["p50"] == 1


def test_representation_mismatch_histogram_and_by_seed_consistency():
    """Mixed groups over two seeds; histogram, totals and by-seed must agree."""
    acc = main_accumulator()
    codes = [CODE_ONE] * GROUP

    # seed 1: identical group (0 mismatches), then scale-only mismatch (32)
    acc.add_group("s1", 1, 127, 127, codes, list(codes), _decoded(codes, 0), _decoded(codes, 0))
    acc.add_group("s1", 1, 118, 119, codes, list(codes), _decoded(codes, -9), _decoded(codes, -8))

    # seed 2: one code-only difference (1), then a signed-zero code difference (32)
    codes_r = list(codes)
    codes_r[3] = 0x39
    acc.add_group("s2", 2, 127, 127, codes, codes_r, _decoded(codes, 0), _decoded(codes_r, 0))
    zero_s = [0x00] * GROUP
    zero_r = [0x80] * GROUP
    acc.add_group(
        "s2", 2, 127, 127, zero_s, zero_r, [Dyadic.zero()] * GROUP, [Dyadic.zero()] * GROUP
    )

    payload = acc.finalize()
    representation = payload["distributions"]["group_representation_mismatch_elements"]
    histogram = {int(key): value for key, value in representation["histogram"].items()}
    weighted = sum(key * value for key, value in histogram.items())
    total = payload["coverage"]["valid_elements"] - payload["counts"]["n_element_representation_match"]
    assert weighted == total
    # histogram zero bin is exactly the group bit-match count
    assert histogram[0] == payload["counts"]["n_group_bit_match"] == 1
    assert {key: value for key, value in histogram.items() if value} == {0: 1, 1: 1, 32: 2}
    # by-seed counts must add up to the summary counts
    by_seed = payload["by_seed"]
    assert sum(entry["representation_mismatch_elements"] for entry in by_seed.values()) == total
    assert (
        sum(entry["code_only_mismatch_elements"] for entry in by_seed.values())
        == payload["coverage"]["valid_elements"] - payload["counts"]["n_element_code_only_match"]
    )
    assert (
        sum(entry["decoded_mismatch_elements"] for entry in by_seed.values())
        == payload["coverage"]["valid_elements"] - payload["counts"]["n_decoded_match"]
    )
    assert sum(entry["ulp_mismatch_elements"] for entry in by_seed.values()) == payload["counts"]["n_ulp_mismatch"]
    # seed 1: one scale-only group -> 32 representation mismatches, 0 code-only
    assert by_seed["1"]["representation_mismatch_elements"] == 32
    assert by_seed["1"]["code_only_mismatch_elements"] == 0
    # the ULP-condition counter is not presented as the total mismatch count
    assert by_seed["1"]["ulp_mismatch_elements"] == 0


def test_error_and_decoded_match_rates_are_not_confused():
    codes_s = [CODE_ONE] * GROUP
    codes_r = list(codes_s)
    codes_r[0] = 0x39
    acc = main_accumulator()
    acc.add_group("s", 1, 127, 127, codes_s, codes_r, _decoded(codes_s, 0), _decoded(codes_r, 0))
    payload = acc.finalize()
    assert payload["metrics"]["element_code_only_match_rate"]["value"] == pytest.approx((GROUP - 1) / GROUP)
    assert payload["metrics"]["element_representation_match_rate"]["value"] == pytest.approx(
        (GROUP - 1) / GROUP
    )
    assert payload["metrics"]["decoded_match_rate"]["value"] == pytest.approx((GROUP - 1) / GROUP)


def write_config(tmp_path, **overrides):
    config = {
        "mode": "mx_matrix_sweep",
        "M": 2,
        "N": 32,
        "K_list": [32, 64],
        "distribution_pairs": [
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"},
            {"activation_distribution": "laplace", "weight_distribution": "laplace"},
        ],
        "seeds": [1],
        "pairs_per_seed": 2,
        "output_root": str(tmp_path / "runs"),
        "run_name": "mx_test",
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_end_to_end_run_analyze_and_offline_determinism(tmp_path):
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    run_dir = run_from_file(write_config(tmp_path), verbose=False)
    manifest = load_manifest(run_dir)
    require_schema(manifest, (MX_SCHEMA_VERSION,))
    assert manifest["status"] == "completed"
    assert manifest["counts"]["matrix_pairs"] == 4  # 2 distribution pairs x 2 pairs_per_seed
    assert manifest["counts"]["snapshot_records"] == 8
    assert (run_dir / MX_INPUTS_NAME).exists() and (run_dir / MX_SAMPLES_NAME).exists()
    inputs = list(iter_mx_inputs(run_dir))
    samples = list(iter_mx_samples(run_dir))
    assert len(inputs) == 4 and len(samples) == 8
    assert inputs[0]["x"]["shape"] == [2, 64]
    assert inputs[0]["w"]["shape"] == [64, 32]
    assert inputs[0]["input_sampling_version"] == "mx-fp32-input-v1"
    assert inputs[0]["x_fp32_bits"] and inputs[0]["w_fp32_bits"]
    assert {entry["pair_index"] for entry in inputs} == {0, 1}
    by_pair = {
        (entry["activation_distribution"], entry["seed"], entry["pair_index"]): entry
        for entry in inputs
    }
    same_distribution = [
        entry for entry in inputs if entry["activation_distribution"] == "gaussian"
    ]
    assert same_distribution[0]["x_fp32_bits"] != same_distribution[1]["x_fp32_bits"]

    out_a = analyze_mx_run(run_dir, out_dir=tmp_path / "analysis_a", write_plots=False)
    out_b = analyze_mx_run(run_dir, out_dir=tmp_path / "analysis_b", write_plots=False)
    summary_a = load_summary(out_a)
    summary_b = load_summary(out_b)
    assert summary_a["x_values"] == [32, 64]
    assert len(summary_a["cells"]) == 4
    assert (out_a / "summary.csv").exists()
    assert (out_a / "analysis_manifest.json").exists()
    assert summary_a["summary_version"] == "mac-mx-summary-v3"
    assert len(summary_a["input_diagnostics"]) == 2
    for cell in summary_a["cells"]:
        assert "scale_scenario" not in cell
        assert "distribution" not in cell
        assert cell["activation_distribution"] in {
            "gaussian",
            "laplace",
            "student_t3",
            "outlier",
        }
        assert set(cell["comparisons"]) == {
            "main_independent_scale",
            "aux_shared_reference_scale",
            "accum_error_r_vs_s",
            "total_error_decode_qr_vs_s",
        }
        assert cell["comparisons"]["main_independent_scale"]["coverage"]["groups"] == 4
    for key in ("created_utc",):
        summary_a.pop(key, None)
        summary_b.pop(key, None)
    summary_a["run_id"] = summary_b["run_id"] = "same"
    assert summary_a == summary_b

    from mac_experiment.results.reanalyze import compare_summaries

    assert compare_summaries(summary_a, summary_b) == []


def test_analysis_output_is_strict_json_and_csv(tmp_path):
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    run_dir = run_from_file(write_config(tmp_path), verbose=False)
    out = analyze_mx_run(run_dir, out_dir=tmp_path / "analysis", write_plots=False)
    text = (out / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    parsed = loads_strict(text)
    csv_text = (out / "summary.csv").read_text(encoding="utf-8")
    assert "comparison" in csv_text.splitlines()[0]
    assert "main_independent_scale" in csv_text
    assert "event_" in csv_text


def test_analysis_rejects_dot_product_run(tmp_path):
    from mac_experiment.experiments import width_sweep

    dot_config = {
        "mode": "width_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": 1,
        "B": 2,
        "output_root": str(tmp_path / "dot"),
    }
    path = tmp_path / "dot.yaml"
    path.write_text(yaml.safe_dump(dot_config), encoding="utf-8")
    run_dir = width_sweep.run_from_file(path, verbose=False)
    with pytest.raises(ValueError):
        analyze_mx_run(run_dir, out_dir=tmp_path / "bad", write_plots=False)


def test_analysis_rejects_legacy_v1_input_runs(tmp_path):
    """Historical v1 runs are excluded explicitly, never silently re-interpreted."""
    import gzip
    import json
    import shutil

    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    run_dir = run_from_file(write_config(tmp_path), verbose=False)

    legacy = tmp_path / "legacy_run"
    shutil.copytree(run_dir, legacy)
    manifest_path = legacy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("input_sampling_version", None)
    manifest.pop("input_stream_version", None)
    manifest.pop("input_quantization_policy", None)
    manifest["schema_version"] = "mac-mx-exp-v2"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="retired v1 MX input scheme"):
        analyze_mx_run(legacy, out_dir=tmp_path / "legacy_analysis", write_plots=False)

    old_schema = tmp_path / "old_schema_run"
    shutil.copytree(run_dir, old_schema)
    manifest = json.loads((old_schema / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "mac-mx-exp-v1"
    (old_schema / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema version mismatch"):
        analyze_mx_run(old_schema, out_dir=tmp_path / "old_schema_analysis", write_plots=False)


def test_source_run_is_read_only_and_json_is_strict(tmp_path):
    import hashlib
    import json

    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    run_dir = run_from_file(write_config(tmp_path), verbose=False)

    def digest(name):
        return hashlib.sha256((run_dir / name).read_bytes()).hexdigest()

    before = {name: digest(name) for name in (MX_INPUTS_NAME, MX_SAMPLES_NAME)}
    out = analyze_mx_run(run_dir, out_dir=tmp_path / "analysis", write_plots=False)
    after = {name: digest(name) for name in (MX_INPUTS_NAME, MX_SAMPLES_NAME)}
    assert before == after
    summary = load_summary(out)
    manifest = json.loads((out / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["input_sampling_version"] == "mx-fp32-input-v1"
    for name, identity in manifest["source"]["files"].items():
        assert identity["sha256"].startswith("sha256:")
    json.dumps(summary, allow_nan=False, sort_keys=True)


def test_saved_records_match_independent_fraction_oracle(tmp_path):
    """Independent codec + Fraction oracle over the saved input/snapshot files."""
    from fractions import Fraction

    from mac_experiment.arithmetic.fp32 import Fp32
    from mac_experiment.arithmetic.dyadic import Dyadic
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    from oracle import oracle_fp32_bits

    run_dir = run_from_file(write_config(tmp_path), verbose=False)
    inputs = {record["sample_id"]: record for record in iter_mx_inputs(run_dir)}

    def element(code):
        sign = (code >> 7) & 1
        exponent = (code >> 3) & 0xF
        fraction = code & 0x7
        if exponent == 15 and fraction == 7:
            raise ValueError("NaN code in saved input")
        if exponent == 0:
            magnitude = Fraction(fraction, 512)
        else:
            magnitude = Fraction(8 + fraction) * Fraction(2) ** (exponent - 10)
        return -magnitude if sign else magnitude

    def scale(code):
        return Fraction(2) ** (code - 127)

    checked = 0
    for record in iter_mx_samples(run_dir):
        if record["K"] != 32:
            continue
        payload = inputs[record["sample_id"]]
        x, w = payload["x"], payload["w"]
        M, N = record["M"], record["N"]
        x_rows, x_cols = x["shape"]
        for m in range(M):
            for n in range(N):
                total = Fraction(0)
                l1 = Fraction(0)
                for k in range(32):
                    xv = element(x["element_codes"][m * x_cols + k]) * scale(x["scale_codes"][m * (x_cols // 32) + k // 32])
                    wv = element(w["element_codes"][k * N + n]) * scale(w["scale_codes"][(k // 32) * N + n])
                    total += xv * wv
                    l1 += abs(xv * wv)
                index = m * N + n
                assert Dyadic.from_dict(record["S"][index]).to_fraction() == total
                assert Dyadic.from_dict(record["L1"][index]).to_fraction() == l1
                # independent dot16 FP32 recurrence
                state = Fraction(0)
                bits = 0
                for block in range(2):
                    partial = Fraction(0)
                    for i in range(16):
                        k = block * 16 + i
                        xv = element(x["element_codes"][m * x_cols + k]) * scale(x["scale_codes"][m * (x_cols // 32) + k // 32])
                        wv = element(w["element_codes"][k * N + n]) * scale(w["scale_codes"][(k // 32) * N + n])
                        partial += xv * wv
                    bits = oracle_fp32_bits(state + partial)
                    state = Fp32.from_bits(bits).exact_dyadic().to_fraction()
                assert record["r_bits"][index] == bits
                checked += 1
    assert checked > 0


def test_analysis_refuses_existing_output(tmp_path):
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    run_dir = run_from_file(write_config(tmp_path), verbose=False)
    out = tmp_path / "analysis"
    analyze_mx_run(run_dir, out_dir=out, write_plots=False)
    with pytest.raises(FileExistsError):
        analyze_mx_run(run_dir, out_dir=out, write_plots=False)


def _rewrite_samples(run_dir, destination, transform):
    import gzip
    import json
    import shutil

    shutil.copytree(run_dir, destination)
    path = destination / MX_SAMPLES_NAME
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records.append(transform(record))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def test_analysis_detects_injected_code_change(tmp_path):
    """Independent check: tampering with one snapshot's R must show up in the metrics."""
    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    run_dir = run_from_file(write_config(tmp_path), verbose=False)
    clean_out = analyze_mx_run(run_dir, out_dir=tmp_path / "clean", write_plots=False)
    clean = load_summary(clean_out)

    tampered = tmp_path / "tampered_run"

    def bump_exponent(record):
        if record["record_type"] == "matrix_snapshot" and record["K"] == 32:
            bits = int(record["r_bits"][0])
            exponent = (bits >> 23) & 0xFF
            if exponent < 250:
                bits = (bits & 0x807FFFFF) | ((exponent + 3) << 23)
            record["r_bits"][0] = bits
        return record

    _rewrite_samples(run_dir, tampered, bump_exponent)
    tampered_out = analyze_mx_run(tampered, out_dir=tmp_path / "tampered", write_plots=False)
    summary = load_summary(tampered_out)

    clean_cells = {
        (c["activation_distribution"], c["weight_distribution"], c["K"]): c for c in clean["cells"]
    }
    for cell in summary["cells"]:
        base = clean_cells[
            (cell["activation_distribution"], cell["weight_distribution"], cell["K"])
        ]
        base_main = base["comparisons"]["main_independent_scale"]
        main = cell["comparisons"]["main_independent_scale"]
        aux = cell["comparisons"]["aux_shared_reference_scale"]
        if cell["K"] != 32:
            assert main == base_main
            continue
        assert main["metrics"]["element_representation_match_rate"]["value"] < 1.0
        assert aux["distributions"]["ulp"]["n_mismatch"] > 0
        assert aux["distributions"]["ulp"]["p50"] is not None
