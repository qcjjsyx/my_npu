"""Plots for the fixed-W_align K/B sweep analysis summary.

Figures group by activation distribution (subplots) and weight distribution
(lines); the K axis states B = K/16.  Error figures use the saved summary only
(no engine, no records).  Alignment event rates use explicit denominators
(products_nonzero / dot16_total) and are marked as cumulative up to K.
"""

from __future__ import annotations

from pathlib import Path

from .style import apply_metric_scale, configure_matplotlib


def _orders(summary: dict):
    activations: list[str] = []
    weights: list[str] = []
    for cell in summary["cells"]:
        if cell["activation_distribution"] not in activations:
            activations.append(cell["activation_distribution"])
        if cell["weight_distribution"] not in weights:
            weights.append(cell["weight_distribution"])
    return activations, weights


def _cell(summary, activation, weight, K):
    for cell in summary["cells"]:
        if (
            cell["activation_distribution"] == activation
            and cell["weight_distribution"] == weight
            and cell["K"] == K
        ):
            return cell
    return None


def _draw(ax, summary, activation, getter):
    _, weights = _orders(summary)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]
    drawn: list[float] = []
    for index, weight in enumerate(weights):
        xs, ys, nulls = [], [], []
        for cell in summary["cells"]:
            if cell["activation_distribution"] != activation or cell["weight_distribution"] != weight:
                continue
            value = getter(cell)
            xs.append(int(cell["x"]))
            ys.append(value)
            if value is None:
                nulls.append(int(cell["x"]))
        if not xs:
            continue  # no cells for this activation/weight combination: no empty line
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        nulls = sorted(nulls)
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        drawn.extend(y for _, y in pairs)
        ax.plot(
            [x for x, _ in pairs],
            [y for _, y in pairs],
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            linestyle="-",
            markersize=3.5,
            linewidth=1.0,
            label=f"W: {weight}",
        )
        if nulls:
            ax.plot(nulls, [0.0] * len(nulls), marker="o", mfc="none", mec="black", linestyle="none", markersize=4)
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


def _comparison_metric(cell, comparison, metric):
    payload = cell["comparisons"].get(comparison)
    if payload is None:
        return None
    entry = (payload.get("metrics") or {}).get(metric)
    return None if entry is None else entry["value"]


def _mx_metric(cell, comparison, metric):
    payload = cell["mx_comparisons"].get(comparison)
    if payload is None:
        return None
    entry = (payload.get("metrics") or {}).get(metric)
    return None if entry is None else entry["value"]


def _alignment_rate(cell, key, denominator_key):
    diagnostics = cell["alignment_diagnostics"]["events_cumulative"]
    denominator = diagnostics.get(denominator_key, 0)
    if not denominator:
        return None
    return diagnostics.get(key, 0) / denominator


def _transition_rate(cell, key):
    transitions = cell["transitions"]["groups"]
    denominator = transitions.get("denominator", 0)
    if not denominator:
        return None
    return transitions.get(key, 0) / denominator


def plot_mx_alignment_summary(summary: dict, out_dir: str | Path) -> list[Path]:
    plt = configure_matplotlib()
    out = Path(out_dir)
    activations, weights = _orders(summary)
    written: list[Path] = []

    def figure(title: str):
        n_rows = max(1, len(activations))
        fig, axes = plt.subplots(n_rows, 1, figsize=(7.4, 2.6 * n_rows), squeeze=False)
        fig.suptitle(title, fontsize=9)
        return fig, axes[:, 0]

    error_series = (
        ("align_T_minus_S", "relative_l2", "alignment T-S relative L2 (reference S)"),
        ("align_T_minus_S", "p99_abs_error", "alignment T-S p99 absolute error"),
        ("baseline_R_minus_S", "relative_l2", "legacy R-S relative L2 (reference S)"),
        ("candidate_A_minus_T", "relative_l2", "candidate A-T relative L2 (reference T)"),
        ("candidate_A_minus_S", "relative_l2", "candidate A-S relative L2 (reference S)"),
        ("candidate_A_minus_R", "relative_l2", "candidate A-R relative L2 (reference R)"),
    )
    for comparison, metric, title in error_series:
        fig, axes = figure(title + "\nsubplot: activation distribution, line: weight distribution")
        for ax, activation in zip(axes, activations):
            values = _draw(
                ax,
                summary,
                activation,
                lambda cell, c=comparison, m=metric: _comparison_metric(cell, c, m),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            apply_metric_scale(ax, "value", values)
            _finish(ax, summary)
        name = f"mx_align_error_{comparison}_{metric}"
        _save(fig, out, name)
        written.append(out / f"{name}.png")

    for comparison, metric, title in (
        ("mx_A_vs_S_independent", "group_representation_mismatch_rate", "MX group representation mismatch: Q_A vs Q_S"),
        ("mx_A_vs_R_independent", "group_representation_mismatch_rate", "MX group representation mismatch: Q_A vs Q_R"),
        ("mx_R_vs_S_independent", "group_representation_mismatch_rate", "MX group representation mismatch: Q_R vs Q_S (baseline)"),
    ):
        fig, axes = figure(title + "\nsubplot: activation distribution, line: weight distribution")
        for ax, activation in zip(axes, activations):
            values = _draw(
                ax,
                summary,
                activation,
                lambda cell, c=comparison, m=metric: _mx_metric(cell, c, m),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            apply_metric_scale(ax, "rate", values)
            _finish(ax, summary)
        name = f"mx_align_{comparison}_{metric}"
        _save(fig, out, name)
        written.append(out / f"{name}.png")

    fig, axes = figure(
        "shared reference scale: conditional E4M3 ULP distance p99 (Q_A vs Q_S)\n"
        "nearest-rank over mismatching elements in valid groups"
    )
    for ax, activation in zip(axes, activations):
        values = _draw(
            ax,
            summary,
            activation,
            lambda cell: (
                ((cell["mx_comparisons"].get("mx_A_vs_S_shared") or {}).get("distributions") or {})
                .get("ulp", {})
                .get("p99")
            ),
        )
        ax.set_title(f"activation: {activation}", fontsize=8)
        apply_metric_scale(ax, "value", values)
        _finish(ax, summary)
    _save(fig, out, "mx_align_shared_ulp_p99_A_vs_S")
    written.append(out / "mx_align_shared_ulp_p99_A_vs_S.png")

    for rate_key, denominator_key, name, title in (
        ("dot16_qb_diff", "dot16_total", "mx_align_rate_qb_diff", "rate of dot16 blocks with q_b != s_b (cumulative up to K)"),
        ("products_changed", "products_nonzero", "mx_align_rate_products_changed", "rate of aligned products whose integer value changed"),
        ("products_zeroed", "products_nonzero", "mx_align_rate_products_zeroed", "rate of non-zero products aligned to zero"),
        ("dot16_with_changed_product", "dot16_total", "mx_align_rate_dot16_changed", "rate of dot16 blocks with at least one changed product"),
    ):
        fig, axes = figure(title + "\nsubplot: activation distribution, line: weight distribution")
        for ax, activation in zip(axes, activations):
            values = _draw(
                ax,
                summary,
                activation,
                lambda cell, k=rate_key, d=denominator_key: _alignment_rate(cell, k, d),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            apply_metric_scale(ax, "rate", values)
            _finish(ax, summary)
        _save(fig, out, name)
        written.append(out / f"{name}.png")

    transition_keys = (
        ("baseline_match_candidate_mismatch", "baseline match -> candidate representation mismatch"),
        ("baseline_mismatch_candidate_match", "baseline mismatch -> candidate match"),
        ("both_mismatch_candidate_differs", "both mismatch, Q_A != Q_R"),
        ("both_mismatch_candidate_same", "both mismatch, Q_A == Q_R"),
    )
    for key, title in transition_keys:
        fig, axes = figure(
            f"representation transitions (groups): {title}\nsubplot: activation distribution, line: weight distribution"
        )
        for ax, activation in zip(axes, activations):
            values = _draw(ax, summary, activation, lambda cell, k=key: _transition_rate(cell, k))
            ax.set_title(f"activation: {activation}", fontsize=8)
            apply_metric_scale(ax, "rate", values)
            _finish(ax, summary)
        name = f"mx_align_transition_{key}"
        _save(fig, out, name)
        written.append(out / f"{name}.png")

    for error_name, title in (
        ("decode_QA_minus_S", "total decode(Q_A)-S relative L2 (reference S)"),
        ("decode_QR_minus_S", "total decode(Q_R)-S relative L2 (reference S)"),
    ):
        fig, axes = figure(title)
        for ax, activation in zip(axes, activations):
            values = _draw(
                ax,
                summary,
                activation,
                lambda cell, n=error_name: (
                    ((cell["total_errors"].get(n) or {}).get("metrics") or {})
                    .get("relative_l2", {})
                    .get("value")
                ),
            )
            ax.set_title(f"activation: {activation}", fontsize=8)
            apply_metric_scale(ax, "value", values)
            _finish(ax, summary)
        _save(fig, out, f"mx_align_total_{error_name}_relative_l2")
        written.append(out / f"mx_align_total_{error_name}_relative_l2.png")

    return written
