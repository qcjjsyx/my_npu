"""Shared plotting style and helpers.

Plotting only consumes saved summary dictionaries; it never imports the
sampling or execution engines.
"""

from __future__ import annotations

import math

COMPARISONS = (
    ("total", "total (a_B vs S_B)"),
    ("window_effect", "window_effect (a_B vs r_B)"),
    ("fp32_baseline", "fp32_baseline (r_B vs S_B)"),
    ("alignment_only", "alignment_only (Q_B vs S_B)"),
)

COMPARISON_COLORS = {
    "total": "#1f77b4",
    "window_effect": "#d62728",
    "fp32_baseline": "#2ca02c",
    "alignment_only": "#9467bd",
}

COMPARISON_MARKERS = {
    "total": "o",
    "window_effect": "o",
    "fp32_baseline": "o",
    "alignment_only": "^",
}

METRICS = (
    "p99_abs_error",
    "p99_eta_l1",
    "relative_l2",
    "rmse_abs_error",
    "max_abs_error",
    "mean_signed_error",
    "match_rate",
)

METRIC_TITLES = {
    "p99_abs_error": "p99 absolute error",
    "p99_eta_l1": "p99 eta_l1 (abs(error)/L1)",
    "relative_l2": "relative L2",
    "rmse_abs_error": "RMSE absolute error",
    "max_abs_error": "max absolute error",
    "mean_signed_error": "mean signed error",
    "match_rate": "exact match rate",
}

STAGE_COLORS = {"mean": "#1f77b4", "rms": "#d62728"}

ABNORMAL_EVENT_KEYS = ("window_overflow", "non_finite_fp32", "unexpected_zero")


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
        }
    )
    return plt


CANONICAL_DISTRIBUTION_ORDER = (
    "positive",
    "uniform",
    "exponent_gaussian",
    "cancellation",
    "laplace",
    "student_t3",
    "outlier",
)


def distribution_names(summary: dict) -> list[str]:
    present = []
    for cell in summary["cells"]:
        if cell["distribution"] not in present:
            present.append(cell["distribution"])
    ordered = [name for name in CANONICAL_DISTRIBUTION_ORDER if name in present]
    ordered.extend(name for name in present if name not in ordered)
    return ordered


def format_names(summary: dict) -> list[str]:
    names = []
    for cell in summary["cells"]:
        if cell["format"] not in names:
            names.append(cell["format"])
    return names


def _coverage_of(payload: dict) -> dict:
    """Coverage of a comparison payload; derives v1 summaries without coverage."""
    coverage = payload.get("coverage")
    if coverage is None:
        counts = payload.get("counts", {})
        coverage = {
            "n_total": counts.get("n_total"),
            "n_valid": counts.get("n_defined", counts.get("n_valid")),
            "n_invalid": counts.get("n_undefined", counts.get("n_invalid")),
        }
    coverage = dict(coverage)
    if coverage.get("invalid_rate") is None and coverage.get("n_total"):
        coverage["invalid_rate"] = (coverage.get("n_invalid") or 0) / coverage["n_total"]
    return coverage


def comparison_status_of(payload: dict) -> str:
    status = payload.get("status")
    if status is not None:
        return status
    value = payload.get("metrics", {}).get("p99_abs_error", {}).get("value")
    return "ok" if value is not None else "failed"


def collect_series(summary: dict, format_name: str, distribution: str, comparison: str, metric: str):
    xs: list[int] = []
    ys: list[float | None] = []
    statuses: list[str] = []
    abnormal: list[bool] = []
    coverages: list[dict] = []
    for cell in summary["cells"]:
        if cell["format"] != format_name or cell["distribution"] != distribution:
            continue
        comparison_payload = cell["comparisons"][comparison]
        payload = comparison_payload["metrics"].get(metric)
        xs.append(int(cell["x"]))
        ys.append(None if payload is None else payload["value"])
        statuses.append("missing" if payload is None else payload["status"])
        abnormal.append(any(cell["events"].get(key, 0) > 0 for key in ABNORMAL_EVENT_KEYS) or cell["counts"]["aborted"] > 0)
        coverages.append(_coverage_of(comparison_payload))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return (
        [xs[i] for i in order],
        [ys[i] for i in order],
        [statuses[i] for i in order],
        [abnormal[i] for i in order],
        [coverages[i] for i in order],
    )


def collect_subset_series(summary: dict, format_name: str, distribution: str, comparison: str, metric: str):
    """Finite-subset diagnostic values; empty when no subset payload exists."""
    xs: list[int] = []
    ys: list[float] = []
    for cell in summary["cells"]:
        if cell["format"] != format_name or cell["distribution"] != distribution:
            continue
        subset = cell["comparisons"][comparison].get("finite_subset")
        if not subset:
            continue
        payload = subset["metrics"].get(metric)
        if payload is None or payload["value"] is None:
            continue
        xs.append(int(cell["x"]))
        ys.append(payload["value"])
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def comparison_failure_counts(summary: dict, format_name: str, distribution: str, comparison: str) -> dict:
    n_x = 0
    n_ok = 0
    n_invalid = 0
    for cell in summary["cells"]:
        if cell["format"] != format_name or cell["distribution"] != distribution:
            continue
        n_x += 1
        payload = cell["comparisons"][comparison]
        if comparison_status_of(payload) == "ok":
            n_ok += 1
        else:
            n_invalid += int(_coverage_of(payload).get("n_invalid") or 0)
    return {"n_x": n_x, "n_ok": n_ok, "n_failed_x": n_x - n_ok, "n_invalid_samples": n_invalid}


def apply_metric_scale(ax, metric: str, all_values: list[float | None]) -> str:
    if metric == "match_rate":
        ax.set_ylim(-0.05, 1.05)
        return "linear 0..1"
    positives = [v for v in all_values if v is not None and math.isfinite(v) and v > 0.0]
    if not positives:
        ax.set_ylim(-0.05, 1.0)
        return "linear (no positive values)"
    linthresh = min(positives)
    if metric == "mean_signed_error":
        ax.set_yscale("symlog", linthresh=linthresh)
        return f"symlog (linear below {linthresh:.3g}; signed)"
    ax.set_yscale("symlog", linthresh=linthresh)
    return f"symlog (linear below {linthresh:.3g})"


def set_x_axis(ax, summary: dict) -> None:
    x_name = summary["x_name"]
    if x_name == "W":
        xs = summary["x_values"]
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=90, fontsize=6)
        ax.set_xlabel("W")
    else:
        xs = summary["x_values"]
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=90, fontsize=6)
        ax.set_xlim(min(xs) * 0.8, max(xs) * 1.25)
        ax.set_xlabel("B (log2 axis)")


def run_subtitle(summary: dict) -> str:
    fixed = summary["fixed"]
    if summary["x_name"] == "W":
        fixed_text = f"fixed B={fixed['B']}"
    else:
        fixed_text = f"fixed W={fixed['W']}"
    config = summary["config"]
    n_samples = len(config["seeds"]) * config["samples_per_seed"]
    return (
        f"{fixed_text} | samples per cell={n_samples} | "
        f"alignment={summary.get('alignment', 'rne_ties_even')} | single FP32 RNE per update"
    )
