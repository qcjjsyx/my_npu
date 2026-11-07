import yaml

from mac_experiment.experiments import width_sweep
from mac_experiment.plotting import plot_all
from mac_experiment.results.reader import load_summary


def test_plot_all_creates_png_and_svg(tmp_path):
    config = {
        "mode": "width_sweep",
        "formats": ["fp8_e4m3", "tf32"],
        "distributions": ["positive", "laplace"],
        "seeds": [1],
        "samples_per_seed": 1,
        "B": 2,
        "output_root": str(tmp_path / "results"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = width_sweep.run_from_file(path, verbose=False)
    summary = load_summary(run_dir)
    out = tmp_path / "plots"
    written = plot_all(summary, out)
    pngs = [p for p in written if p.suffix == ".png"]
    svgs = [p for p in written if p.suffix == ".svg"]
    assert len(pngs) == 2 * (7 + 2)
    assert len(svgs) == len(pngs)
    for path in written:
        assert path.exists() and path.stat().st_size > 1000


def test_plot_all_handles_undefined_values(tmp_path):
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
    from mac_experiment.experiments import block_sweep

    run_dir = block_sweep.run_from_file(path, verbose=False)
    summary = load_summary(run_dir)
    for cell in summary["cells"]:
        cell["comparisons"]["total"]["metrics"]["p99_abs_error"]["value"] = None
        cell["comparisons"]["total"]["metrics"]["p99_abs_error"]["status"] = "undefined_zero_l1"
    out = tmp_path / "plots"
    written = plot_all(summary, out)
    assert written and all(p.exists() for p in written)
