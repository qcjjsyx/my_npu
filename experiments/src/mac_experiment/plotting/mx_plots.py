"""Plots for the MX output-quantisation analysis summary.

Reads only the saved analysis summary (never the run records or the engine).
Figures are grouped by activation distribution (subplots) and weight
distribution (lines); there is no scale-scenario grouping in the current input
scheme.  The K axis states the B = K/16 relation, zero/null values are
distinguished from missing data, and scan points with invalid groups are marked
in red.  Input scale/exponent plots are diagnostics only.
"""

from __future__ import annotations

from pathlib import Path

from .style import apply_metric_scale, configure_matplotlib


def _metric_value(cell: dict, comparison: str, metric: str):
    payload = cell["comparisons"].get(comparison)
    if payload is None:
        return None
    entry = payload.get("metrics", {}).get(metric)
    if entry is None:
        return None
    return entry["value"]


def _ulp_value(cell: dict, comparison: str, key: str):
    payload = cell["comparisons"].get(comparison)
    if payload is None:
        return None
    ulp = (payload.get("distributions") or {}).get("ulp")
    if ulp is None:
        return None
    return ulp.get(key)


def _distribution_value(cell: dict, comparison: str, section: str, key: str):
    payload = cell["comparisons"].get(comparison)
    if payload is None:
        return None
    data = (payload.get("distributions") or {}).get(section)
    if data is None:
        return None
    return data.get(key)


def _invalid(cell: dict, comparison: str) -> int:
    payload = cell["comparisons"].get(comparison)
    if payload is None:
        return 0
    return int((payload.get("coverage") or {}).get("invalid_groups", 0))


def _orders(summary: dict):
    activations: list[str] = []
    weights: list[str] = []
    for cell in summary["cells"]:
        if cell["activation_distribution"] not in activations:
            activations.append(cell["activation_distribution"])
        if cell["weight_distribution"] not in weights:
            weights.append(cell["weight_distribution"])
    return activations, weights


def _collect(summary, activation, weight, comparison, getter):
    xs, ys, markers = [], [], []
    for cell in summary["cells"]:
        if (
            cell["activation_distribution"] != activation
            or cell["weight_distribution"] != weight
        ):
            continue
        xs.append(int(cell["x"]))
        value = getter(cell)
        if value is None:
            ys.append(None)
            markers.append("null")
        else:
            ys.append(float(value))
            markers.append("invalid" if _invalid(cell, comparison) > 0 else "ok")
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return (
        [xs[i] for i in order],
        [ys[i] for i in order],
        [markers[i] for i in order],
    )


def _draw_series(ax, summary, activation, comparison, getter):
    """Draw one line per weight distribution; returns all finite values drawn."""
    _, weights = _orders(summary)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]
    drawn: list[float] = []
    for index, weight in enumerate(weights):
        xs, ys, flags = _collect(summary, activation, weight, comparison, getter)
        if not xs:
            continue
        xs_ok = [x for x, y, flag in zip(xs, ys, flags) if y is not None and flag == "ok"]
        ys_ok = [y for y, flag in zip(ys, flags) if y is not None and flag == "ok"]
        xs_bad = [x for x, y, flag in zip(xs, ys, flags) if y is not None and flag == "invalid"]
        ys_bad = [y for y, flag in zip(ys, flags) if y is not None and flag == "invalid"]
        xs_null = [x for x, y in zip(xs, ys) if y is None]
        drawn.extend(ys_ok)
        drawn.extend(ys_bad)
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        ax.plot(
            xs_ok,
            ys_ok,
            marker=marker,
            color=color,
            linestyle="-",
            markersize=3.5,
            linewidth=1.0,
            label=f"W: {weight}",
        )
        if xs_bad:
            ax.plot(xs_bad, ys_bad, marker="x", color="red", linestyle="none", markersize=5)
        if xs_null:
            ax.plot(
                xs_null,
                [0.0] * len(xs_null),
                marker="o",
                mfc="none",
                mec="black",
                linestyle="none",
                markersize=4,
            )
    ax.grid(True, which="both", alpha=0.25)
    return drawn


def _finish(ax, summary) -> None:
    xs = summary["x_values"]
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}\nB={k // 16}" for k in xs], fontsize=6, rotation=0)
    ax.set_xlabel("K (log2 axis; B = K/16)")
    ax.legend(fontsize=6, loc="best")


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.svg")
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_mx_summary(summary: dict, out_dir: str | Path) -> list[Path]:
    plt = configure_matplotlib()
    out = Path(out_dir)
    activations, weights = _orders(summary)
    written: list[Path] = []

    def figure(title: str, rows: int | None = None, height: float = 2.6):
        n_rows = rows or max(1, len(activations))
        fig, axes = plt.subplots(n_rows, 1, figsize=(7.4, height * max(1, n_rows)), squeeze=False)
        fig.suptitle(title, fontsize=9)
        return fig, axes[:, 0]

    for metric, title in (
        ("scale_mismatch_rate", "independent-scale output scale mismatch rate"),
        (
            "group_representation_mismatch_rate",
            "MX group representation mismatch rate (scale_code, element_code)",
        ),
    ):
        fig, axes = figure(
            f"{title} | subplot: activation distribution, line: weight distribution\n"
            "red x: point has invalid groups; hollow circle: null (no valid groups)"
        )
        for ax, activation in zip(axes, activations):
            values = _draw_series(
                ax,
                summary,
                activation,
                "main_independent_scale",
                lambda c, m=metric: _metric_value(c, "main_independent_scale", m),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            apply_metric_scale(ax, "rate", values)
            _finish(ax, summary)
        _save(fig, out, f"mx_rate_{metric}")
        written.append(out / f"mx_rate_{metric}.png")

    fig, axes = figure(
        "independent-scale element representation (scale_code, element_code) mismatch rate\n"
        "subplot: activation distribution, line: weight distribution"
    )
    for ax, activation in zip(axes, activations):
        values = _draw_series(
            ax,
            summary,
            activation,
            "main_independent_scale",
            lambda c: _element_mismatch(c),
        )
        ax.set_title(f"activation: {activation}", fontsize=8)
        apply_metric_scale(ax, "rate", values)
        _finish(ax, summary)
    _save(fig, out, "mx_rate_element_representation_mismatch")
    written.append(out / "mx_rate_element_representation_mismatch.png")

    for key, title in (
        ("p50", "shared reference scale: conditional E4M3 ULP distance p50"),
        ("p90", "shared reference scale: conditional E4M3 ULP distance p90"),
        ("p99", "shared reference scale: conditional E4M3 ULP distance p99"),
        ("max", "shared reference scale: conditional E4M3 ULP distance max"),
    ):
        fig, axes = figure(
            title + "\nnearest-rank over mismatching elements in valid groups"
        )
        for ax, activation in zip(axes, activations):
            values = _draw_series(
                ax,
                summary,
                activation,
                "aux_shared_reference_scale",
                lambda c, k=key: _ulp_value(c, "aux_shared_reference_scale", k),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            apply_metric_scale(ax, "value", values)
            _finish(ax, summary)
        _save(fig, out, f"mx_shared_scale_ulp_{key}")
        written.append(out / f"mx_shared_scale_ulp_{key}.png")

    for section, key, name, title in (
        (
            "group_representation_mismatch_elements",
            "mean",
            "mx_group_representation_mismatch_mean",
            "per-group representation mismatch element count (scale_code, element_code), mean over valid groups",
        ),
        (
            "group_code_only_mismatch_elements",
            "mean",
            "mx_group_code_only_mismatch_mean",
            "explicit code-only diagnostic: per-group element-code mismatch count (scale ignored)",
        ),
    ):
        fig, axes = figure(title)
        for ax, activation in zip(axes, activations):
            values = _draw_series(
                ax,
                summary,
                activation,
                "main_independent_scale",
                lambda c, s=section, k=key: _distribution_value(c, "main_independent_scale", s, k),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            _finish(ax, summary)
        _save(fig, out, name)
        written.append(out / f"{name}.png")

    for section, name, title in (
        (
            "group_representation_mismatch_elements",
            "mx_group_representation_mismatch_histogram",
            "per-group representation mismatch histogram at K={k} (B={b})",
        ),
        (
            "group_code_only_mismatch_elements",
            "mx_group_code_only_mismatch_histogram",
            "explicit code-only diagnostic: per-group element-code mismatch histogram at K={k} (B={b})",
        ),
    ):
        max_k = max(summary["x_values"])
        fig, axes = plt.subplots(
            len(activations), 1, figsize=(7.6, 3.0 * max(1, len(activations))), squeeze=False
        )
        fig.suptitle(title.format(k=max_k, b=max_k // 16))
        for ax, activation in zip(axes[:, 0], activations):
            width = 0.8 / max(1, len(weights))
            for index, weight in enumerate(weights):
                cell = next(
                    (
                        c
                        for c in summary["cells"]
                        if c["activation_distribution"] == activation
                        and c["weight_distribution"] == weight
                        and c["K"] == max_k
                    ),
                    None,
                )
                if cell is None:
                    continue
                payload = cell["comparisons"]["main_independent_scale"]
                mismatch = (payload.get("distributions") or {}).get(section)
                if mismatch is None:
                    continue
                counts = mismatch["histogram"]
                bins = list(range(33))
                values = [counts.get(str(b), 0) for b in bins]
                offset = (index - (len(weights) - 1) / 2) * width
                ax.bar([b + offset for b in bins], values, width=width, label=f"W: {weight}")
            ax.set_title(f"activation: {activation}", fontsize=8)
            ax.set_xlabel("mismatching elements per group (of 32)")
            ax.set_ylabel("groups")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.25)
        _save(fig, out, name)
        written.append(out / f"{name}.png")

    for metric, title in (
        ("decoded_abs_error_p50", "decoded |Q_R - Q_S| p50"),
        ("decoded_abs_error_p99", "decoded |Q_R - Q_S| p99"),
        ("decoded_abs_error_max", "decoded |Q_R - Q_S| max"),
        ("decoded_relative_l2", "decoded relative L2 (reference: decoded Q_S)"),
    ):
        fig, axes = figure(f"independent scale: {title}")
        for ax, activation in zip(axes, activations):
            values = _draw_series(
                ax,
                summary,
                activation,
                "main_independent_scale",
                lambda c, m=metric: _metric_value(c, "main_independent_scale", m),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            _finish(ax, summary)
        _save(fig, out, f"mx_decoded_{metric}")
        written.append(out / f"mx_decoded_{metric}.png")

    for metric, title in (
        ("p50_abs_error", "total error decode(Q_R) - S, p50"),
        ("p99_abs_error", "total error decode(Q_R) - S, p99"),
        ("max_abs_error", "total error decode(Q_R) - S, max"),
        ("relative_l2", "total error decode(Q_R) - S, relative L2 (reference S)"),
    ):
        fig, axes = figure(title)
        for ax, activation in zip(axes, activations):
            values = _draw_series(
                ax,
                summary,
                activation,
                "total_error_decode_qr_vs_s",
                lambda c, m=metric: _metric_value(c, "total_error_decode_qr_vs_s", m),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            _finish(ax, summary)
        _save(fig, out, f"mx_total_error_{metric}")
        written.append(out / f"mx_total_error_{metric}.png")

    for metric, title in (
        ("p99_abs_error", "accumulation error R - S, p99"),
        ("relative_l2", "accumulation error R - S, relative L2 (reference S)"),
    ):
        fig, axes = figure(title)
        for ax, activation in zip(axes, activations):
            values = _draw_series(
                ax,
                summary,
                activation,
                "accum_error_r_vs_s",
                lambda c, m=metric: _metric_value(c, "accum_error_r_vs_s", m),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            _finish(ax, summary)
        _save(fig, out, f"mx_accum_error_{metric}")
        written.append(out / f"mx_accum_error_{metric}.png")

    written.extend(_plot_input_diagnostics(plt, summary, out, activations, weights))
    return written


def _plot_input_diagnostics(plt, summary: dict, out: Path, activations, weights) -> list[Path]:
    """Diagnostic-only figures for the actual generated inputs (not error metrics)."""
    diagnostics = summary.get("input_diagnostics") or []
    if not diagnostics:
        return []
    written: list[Path] = []
    by_key = {
        (entry["activation_distribution"], entry["weight_distribution"]): entry
        for entry in diagnostics
    }
    fig, axes = plt.subplots(
        len(activations), 1, figsize=(7.4, 2.6 * max(1, len(activations))), squeeze=False
    )
    fig.suptitle(
        "input diagnostics (not error metrics): actual RMS of FP32 draws and decoded MX inputs"
    )
    for ax, activation in zip(axes[:, 0], activations):
        labels, fp32_values, mx_values = [], [], []
        for weight in weights:
            entry = by_key.get((activation, weight))
            if entry is None:
                continue
            labels.append(f"W: {weight}")
            fp32_values.append(entry.get("fp32_rms_mean"))
            mx_values.append(entry.get("mx_decoded_rms_mean"))
        positions = range(len(labels))
        ax.bar([p - 0.2 for p in positions], fp32_values, width=0.4, label="FP32 RMS")
        ax.bar([p + 0.2 for p in positions], mx_values, width=0.4, label="decoded MX RMS")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(f"activation: {activation}", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.25)
    _save(fig, out, "mx_input_rms_diagnostic")
    written.append(out / "mx_input_rms_diagnostic.png")

    fig, axes = plt.subplots(
        len(activations), 1, figsize=(7.4, 2.6 * max(1, len(activations))), squeeze=False
    )
    fig.suptitle(
        "input diagnostic: E8M0 scale exponent histogram of actual input groups (X and W merged)"
    )
    for ax, activation in zip(axes[:, 0], activations):
        merged: dict[int, int] = {}
        for weight in weights:
            entry = by_key.get((activation, weight))
            if entry is None:
                continue
            for exponent, count in (entry.get("scale_exponent_histogram") or {}).items():
                merged[int(exponent)] = merged.get(int(exponent), 0) + int(count)
        if merged:
            exponents = sorted(merged)
            ax.bar([str(e) for e in exponents], [merged[e] for e in exponents])
        ax.set_title(f"activation: {activation}", fontsize=8)
        ax.set_xlabel("E8M0 exponent (code - 127)")
        ax.set_ylabel("groups (X + W, all pairs)")
        ax.grid(True, alpha=0.25)
    _save(fig, out, "mx_input_scale_exponent_histogram")
    written.append(out / "mx_input_scale_exponent_histogram.png")
    return written


def _element_mismatch(cell: dict):
    value = _metric_value(cell, "main_independent_scale", "element_representation_match_rate")
    if value is None:
        return None
    return 1.0 - value
