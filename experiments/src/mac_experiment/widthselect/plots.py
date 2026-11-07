"""Offline figures for the width-select analysis.

Layout (one figure per synthetic distribution pair):

* rows = metrics: ``bit_match_rate``, conditional ULP ``p99``, conditional ULP
  ``max`` (all against ``O = RN32(S)``);
* columns = input formats (``mx_fp8``, ``tf32``); a single-format run degrades
  to one column;
* x-axis = ``B = K // 16`` on a log2 scale, with a top secondary axis showing
  the corresponding ``K = 16B``;
* one curve per candidate width (``W_align`` for the align scan, ``W_acc`` for
  the accumulator scan, ``W_align/W_acc`` pairs for the joint verification).

Empty conditional-ULP cells (no finite bit-mismatch sample) are **plotted at
y = 0 with hollow markers** so every width curve stays continuous (the user's
display convention); the underlying ``summary.json`` keeps ``null`` for these
quantiles.  The panels annotate the convention and the finite/non-finite pair
counts; bit-match curves whose outputs are entirely non-finite are drawn dashed
and named in the panel note, so a trivial ``Inf == Inf`` match rate is not
misread.  matplotlib is imported lazily so the numeric analysis path does not
require it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

METRIC_ROWS = (
    ("bit match rate", "bit_match_rate", (-0.03, 1.03)),
    ("conditional ULP p99 (steps)", "conditional_ulp_p99", None),
    ("conditional ULP max (steps)", "conditional_ulp_max", None),
)

MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">")


def _panel_totals(cells: list[dict]) -> tuple[int, int, int]:
    """``(finite_pairs, finite_mismatch, nonfinite_candidate)`` over panel cells."""
    finite = sum(cell["metrics"]["counts"]["n_finite_pairs"] for cell in cells)
    mismatch = sum(cell["metrics"]["counts"]["n_finite_mismatch"] for cell in cells)
    nonfinite = sum(cell["metrics"]["counts"]["n_nonfinite_candidate"] for cell in cells)
    return finite, mismatch, nonfinite


def series_points(metric_values: list) -> tuple[list[float], list[bool]]:
    """Plot convention: ``None`` (no mismatch samples) is drawn at 0, hollow.

    Returns ``(ys, hollow)``; the summary data keeps ``null`` and only the
    rendering layer applies this convention.
    """
    ys: list[float] = []
    hollow: list[bool] = []
    for value in metric_values:
        if value is None:
            ys.append(0.0)
            hollow.append(True)
        else:
            ys.append(float(value))
            hollow.append(False)
    return ys, hollow


def hollow_points_note(cells: list[dict]) -> str:
    """Panel annotation explaining hollow 0-points and the pair counts."""
    finite, _mismatch, nonfinite = _panel_totals(cells)
    return (
        "hollow = no mismatch samples (drawn at 0)\n"
        f"finite pairs = {finite:,}  nonfinite = {nonfinite:,}"
    )


def all_nonfinite_curves(
    mode: str,
    fmt: str,
    widths: list[tuple],
    b_values: list[int],
    by_key: dict,
) -> list[str]:
    """Width labels whose cells contain no finite pair at all (only non-finite)."""
    labels = []
    for width in widths:
        cells = [
            by_key[(fmt, width, b)]
            for b in b_values
            if (fmt, width, b) in by_key
        ]
        if cells and all(
            cell["metrics"]["counts"]["n_finite_pairs"] == 0 for cell in cells
        ):
            labels.append(_width_label(mode, width))
    return labels


def _import_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "plotting requires matplotlib; install the optional 'plots' extra"
        ) from exc
    return plt


def _mode_of(summary: dict) -> str:
    kind = summary.get("analysis_kind", "")
    if "acc" in kind:
        return "acc"
    if "joint" in kind:
        return "joint"
    return "align"


def _width_label(mode: str, key: tuple) -> str:
    if mode == "acc":
        return f"W_acc={key[0]}"
    if mode == "joint":
        return f"W_align={key[0]}, W_acc={key[1]}"
    return f"W_align={key[0]}"


def _width_key(mode: str, cell: dict) -> tuple:
    if mode == "acc":
        return (cell["w_acc"],)
    if mode == "joint":
        return (cell["w_align"], cell["w_acc"])
    return (cell["w_align"],)


def _metric_value(mode: str, cell: dict, metric: str):
    metrics = cell["metrics"]
    if metric == "bit_match_rate":
        return metrics["bit_match_rate"]
    if metric == "conditional_ulp_p99":
        return metrics["conditional_ulp"]["p99"]
    if metric == "conditional_ulp_max":
        return metrics["conditional_ulp"]["max"]
    raise ValueError(f"unknown metric {metric!r}")


def _pair_file_label(pair: str) -> str:
    parts = pair.split("|")
    label = parts[0] if len(set(parts)) == 1 else "_".join(parts)
    return label.replace("/", "_")


def _mode_title(mode: str) -> str:
    return {
        "align": "W_align scan",
        "acc": "W_acc scan",
        "joint": "joint (W_align, W_acc)",
    }[mode]


def _distribution_figure(plt, summary: dict, cells: list[dict], pair: str, out_dir: Path, mode: str) -> Path:
    formats = summary["formats"]
    b_values = sorted({int(cell["B"]) for cell in cells})
    widths = sorted({_width_key(mode, cell) for cell in cells})
    by_key = {
        (cell["format"], _width_key(mode, cell), int(cell["B"])): cell
        for cell in cells
        if cell["distribution_pair"] == pair
    }

    figure, axes = plt.subplots(
        len(METRIC_ROWS),
        len(formats),
        squeeze=False,
        figsize=(4.6 * len(formats), 3.1 * len(METRIC_ROWS)),
    )
    legend_entries: dict[tuple, object] = {}
    figure_has_hollow = False
    fmt_cells: dict[str, list] = {fmt: [] for fmt in formats}
    for cell in cells:
        if cell["distribution_pair"] == pair:
            fmt_cells[cell["format"]].append(cell)
    for row, (_metric_label, metric, ylim) in enumerate(METRIC_ROWS):
        for col, fmt in enumerate(formats):
            axis = axes[row][col]
            hollow_points: list[tuple] = []
            nonfinite_labels = (
                all_nonfinite_curves(mode, fmt, widths, b_values, by_key)
                if metric == "bit_match_rate"
                else []
            )
            for width_index, width in enumerate(widths):
                values = []
                for b_value in b_values:
                    cell = by_key.get((fmt, width, b_value))
                    values.append(
                        _metric_value(mode, cell, metric) if cell is not None else None
                    )
                ys, hollow = series_points(values)
                marker = MARKERS[width_index % len(MARKERS)]
                line = axis.plot(
                    b_values,
                    ys,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1.2,
                    linestyle="--"
                    if metric == "bit_match_rate" and _width_label(mode, width) in nonfinite_labels
                    else "-",
                    label=_width_label(mode, width),
                )[0]
                legend_entries.setdefault(width, line)
                if metric != "bit_match_rate":
                    for b_value, y_value, is_hollow in zip(b_values, ys, hollow):
                        if is_hollow:
                            hollow_points.append((b_value, y_value, marker, line.get_color()))
            if len(b_values) > 1:
                axis.set_xscale("log", base=2)
            axis.set_xticks(b_values)
            axis.set_xticklabels([str(b) for b in b_values], fontsize=7)
            if row == len(METRIC_ROWS) - 1:
                axis.set_xlabel("B = K / 16", fontsize=8)
            if col == 0:
                axis.set_ylabel(_metric_label, fontsize=8)
            if row == 0:
                axis.set_title(fmt, fontsize=9)
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.tick_params(labelsize=7)
            axis.grid(True, linewidth=0.3, alpha=0.5)
            if hollow_points:
                figure_has_hollow = True
                for b_value, y_value, marker, color in hollow_points:
                    axis.scatter(
                        [b_value],
                        [y_value],
                        facecolors="white",
                        edgecolors=color,
                        marker=marker,
                        s=22,
                        linewidths=1.0,
                        zorder=4,
                    )
                note = hollow_points_note(fmt_cells[fmt])
                all_hollow = len(hollow_points) == len(widths) * len(b_values)
                if all_hollow:
                    axis.text(
                        0.5,
                        0.5,
                        note,
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )
                else:
                    axis.text(
                        0.02,
                        0.96,
                        textwrap.fill(note.replace("\n", "; "), width=72),
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6,
                        color="gray",
                    )
            if nonfinite_labels:
                note = textwrap.fill(
                    "all outputs non-finite (dashed): " + ", ".join(nonfinite_labels),
                    width=72,
                )
                axis.text(
                    0.02,
                    0.96,
                    note,
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="gray",
                )

    # top secondary axis showing K = 16B on every top-row panel
    if len(b_values) > 1:
        for col in range(len(formats)):
            axis = axes[0][col]
            try:
                secondary = axis.secondary_xaxis(
                    "top", functions=(lambda b: b * 16.0, lambda k: k / 16.0)
                )
                secondary.set_xticks([16 * b for b in b_values])
                secondary.set_xticklabels([str(16 * b) for b in b_values], fontsize=6)
                secondary.set_xlabel("K", fontsize=8)
            except Exception:  # pragma: no cover - older matplotlib fallback
                pass

    if legend_entries:
        from matplotlib.lines import Line2D

        handles = list(legend_entries.values())
        labels = [_width_label(mode, key) for key in legend_entries]
        if figure_has_hollow:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="gray",
                    markerfacecolor="white",
                    markeredgecolor="gray",
                    linestyle="None",
                    markersize=5,
                )
            )
            labels.append("no mismatch samples (drawn at 0)")
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=7,
        )
    figure.suptitle(f"{pair}  —  {_mode_title(mode)}", fontsize=11)
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    path = out_dir / f"{_pair_file_label(pair)}_metrics.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def render_plots(summary: dict, out_dir) -> list[Path]:
    cells = summary.get("cells") or []
    if not cells:
        return []
    plt = _import_plt()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mode = _mode_of(summary)
    pairs = sorted({cell["distribution_pair"] for cell in cells})
    paths = [
        _distribution_figure(plt, summary, cells, pair, out_path, mode) for pair in pairs
    ]
    return paths
