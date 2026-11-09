"""MX run config validation and legacy-entry guard tests."""

from __future__ import annotations

import pytest
import yaml

from mac_experiment.experiments.config import MX_MAC_PARALLELISM, load_config
from mac_experiment.matrix.inputs import INPUT_DISTRIBUTIONS


def write_config(tmp_path, **overrides):
    config = {
        "mode": "mx_matrix_sweep",
        "M": 4,
        "N": 64,
        "K_list": [32, 128],
        "distribution_pairs": [
            {"activation_distribution": "gaussian", "weight_distribution": "gaussian"},
            {"activation_distribution": "laplace", "weight_distribution": "outlier"},
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


def test_mx_config_effective_payload(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.mode == "mx_matrix_sweep"
    assert config.is_mx
    assert config.x_name == "K"
    assert config.x_values == [32, 128]
    assert config.snapshots == [32, 128]
    assert config.windows == []
    assert config.fixed == {
        "M": 4,
        "N": 64,
        "group_length": 32,
        "mac_parallelism": MX_MAC_PARALLELISM,
    }
    assert config.distribution_pairs[0] == {
        "activation_distribution": "gaussian",
        "weight_distribution": "gaussian",
    }
    assert config.distribution_pairs[1]["weight_distribution"] == "outlier"
    effective = config.effective()
    assert effective["K_list"] == [32, 128]
    assert effective["pairs_per_seed"] == 2
    assert effective["input_sampling_version"] == "mx-fp32-input-v1"
    assert effective["input_stream_version"] == 1
    assert "scale_scenarios" not in effective


def test_mx_config_rejects_invalid_shapes(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, M=0))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, N=48))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, K_list=[32, 48]))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, K_list=[]))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, M=None))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, K_list=None))


def test_mx_config_rejects_legacy_input_scheme_explicitly(tmp_path):
    with pytest.raises(ValueError, match="retired"):
        load_config(write_config(tmp_path, scale_scenarios=["unit", "narrow"]))
    with pytest.raises(ValueError, match="retired"):
        load_config(write_config(tmp_path, distributions=["uniform"]))
    # B/W/B_list and old element distributions are rejected, not converted
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, B=8))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, W=24))
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, B_list=[1, 2]))
    with pytest.raises(ValueError, match="unknown input distribution"):
        load_config(
            write_config(
                tmp_path,
                distribution_pairs=[
                    {"activation_distribution": "uniform", "weight_distribution": "uniform"}
                ],
            )
        )
    with pytest.raises(ValueError, match="unknown input distribution"):
        load_config(
            write_config(
                tmp_path,
                distribution_pairs=[
                    {"activation_distribution": "gaussian", "weight_distribution": "narrow"}
                ],
            )
        )


def test_mx_config_rejects_custom_distribution_parameters(tmp_path):
    """Custom parameters must be rejected, not silently ignored by the sampler."""
    with pytest.raises(ValueError, match="custom distribution parameters"):
        load_config(
            write_config(
                tmp_path,
                distribution_pairs=[
                    {
                        "activation_distribution": "gaussian",
                        "weight_distribution": "gaussian",
                        "activation_parameters": {"mean": 100.0, "std": 0.0},
                    }
                ],
            )
        )
    with pytest.raises(ValueError, match="custom distribution parameters"):
        load_config(
            write_config(
                tmp_path,
                distribution_pairs=[
                    {
                        "activation_distribution": "laplace",
                        "weight_distribution": "laplace",
                        "weight_parameters": {"scale": 2.0},
                    }
                ],
            )
        )
    # a valid config without parameters still works and keeps only the two keys
    config = load_config(write_config(tmp_path))
    for pair in config.distribution_pairs:
        assert set(pair) == {"activation_distribution", "weight_distribution"}


def test_mx_config_requires_distribution_pairs_and_fields(tmp_path):
    with pytest.raises(ValueError, match="distribution_pairs"):
        load_config(write_config(tmp_path, distribution_pairs=None))
    with pytest.raises(ValueError, match="distribution_pairs"):
        load_config(write_config(tmp_path, distribution_pairs=[]))
    with pytest.raises(ValueError, match="weight_distribution"):
        load_config(
            write_config(
                tmp_path, distribution_pairs=[{"activation_distribution": "gaussian"}]
            )
        )
    with pytest.raises(ValueError, match="unknown distribution_pairs keys"):
        load_config(
            write_config(
                tmp_path,
                distribution_pairs=[
                    {
                        "activation_distribution": "gaussian",
                        "weight_distribution": "gaussian",
                        "scale_scenario": "unit",
                    }
                ],
            )
        )


def test_all_input_distributions_accepted(tmp_path):
    for name in INPUT_DISTRIBUTIONS:
        config = load_config(
            write_config(
                tmp_path,
                distribution_pairs=[
                    {"activation_distribution": name, "weight_distribution": name}
                ],
            )
        )
        assert config.distribution_pairs[0]["activation_distribution"] == name


def test_legacy_element_scale_samplers_are_not_called(tmp_path, monkeypatch):
    """The retired v1 sampler entry points must never run in the active path."""
    import mac_experiment.matrix.sampling as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("retired v1 element/scale sampler was called")

    monkeypatch.setattr(legacy.MxElementSampler, "__init__", forbidden)
    monkeypatch.setattr(legacy.MxScaleSampler, "__init__", forbidden)
    monkeypatch.setattr(legacy, "mx_stream_rng", forbidden)

    from mac_experiment.experiments.mx_matrix_sweep import run_from_file

    run_dir = run_from_file(write_config(tmp_path, pairs_per_seed=1, seeds=[1]), verbose=False)
    assert (run_dir / "manifest.json").exists()


def test_dot_product_config_still_requires_formats(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"mode": "width_sweep", "distributions": ["laplace"], "B": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(path)
