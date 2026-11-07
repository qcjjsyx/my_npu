import yaml

from mac_experiment.cli import main
from mac_experiment.results.reader import load_summary


def test_cli_version_and_distributions(tmp_path, capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "schema" in out and "stream" in out

    config = {
        "mode": "width_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["laplace"],
        "seeds": [1],
        "samples_per_seed": 1,
        "B": 1,
        "output_root": str(tmp_path / "results"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["distributions", "--config", str(path), "--out", str(tmp_path / "pmf.json")]) == 0
    assert (tmp_path / "pmf.json").exists()


def test_cli_run_analyze_plot(tmp_path, capsys):
    config = {
        "mode": "block_sweep",
        "formats": ["fp8_e4m3"],
        "distributions": ["positive"],
        "seeds": [1],
        "samples_per_seed": 1,
        "W": 24,
        "B_list": [1, 2],
        "output_root": str(tmp_path / "results"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "--config", str(path), "--quiet"]) == 0
    run_dir = capsys.readouterr().out.strip()
    assert main(["analyze", "--run", run_dir]) == 0
    assert "matches the original summary" in capsys.readouterr().out
    out_dir = tmp_path / "plots"
    assert main(["plot", "--run", run_dir, "--out", str(out_dir)]) == 0
    summary = load_summary(run_dir)
    assert summary["x_values"] == [1, 2]
    assert (out_dir / "b_sweep_fp8_e4m3_p99_abs_error.png").exists()


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["analyze", "--run", str(tmp_path / "missing")]) == 2
    assert "error" in capsys.readouterr().err
