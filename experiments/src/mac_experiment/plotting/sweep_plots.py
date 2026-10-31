"""Sweep figures: one figure per (format, metric) with distribution panels.

Reads only the summary dictionary produced by a run.  Whole-batch metrics are
plotted as the primary curves; when a comparison has invalid samples the
whole-batch value is ``null`` (marked with a red x) and the finite-subset
diagnostic is drawn as a faint dotted line, never as a normal batch curve.
Every undefined point is marked, abnormal-event x positions are shaded, and
zero values are shown at zero (symlog or linear, documented in the figure
note).
"""

from __future__ import annotations

import math
from pathlib import Path

from .style import (
    COMPARISONS,
    COMPARISON_COLORS,
    COMPARISON_MARKERS,
    METRIC_TITLES,
    METRICS,
    STAGE_COLORS,
    apply_metric_scale,
    collect_series,
    collect_subset_series,
    comparison_failure_counts,
    configure_matplotlib,
    distribution_names,
    format_names,
    run_subtitle,
    set_x_axis,
)

COMPARISON_ABBREV = {
    "total": "total",
    "window_effect": "window",
    "fp32_baseline": "baseline",
    "alignment_only": "align",
}


def _prefix(summary: dict) -> str:
    return "w_sweep" if summary["x_name"] == "W" else "b_sweep"


def _as_plot_values(values):
    return [math.nan if v is None else v for v in values]


def _failure_note(summary: dict, fmt: str, dist: str) -> str:
    parts = []
    for comparison, _label in COMPARISONS:
        counts = comparison_failure_counts(summary, fmt, dist, comparison)
        if counts["n_failed_x"]:
            parts.append(
                f"{COMPARISON_ABBREV[comparison]} {counts['n_failed_x']}/{counts['n_x']} x-points "
                f"({counts['n_invalid_samples']} invalid samples)"
            )
    return "; ".join(parts)


def plot_metric_grids(summary: dict, out_dir: str | Path) -> list[Path]:
    plt = configure_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = format_names(summary)
    dists = distribution_names(summary)
    written: list[Path] = []
    for fmt in formats:
        for metric in METRICS:
            fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2))
            handles: dict[str, object] = {}
            plotted_undefined = False
            plotted_subset = False
            scale_note = "linear 0..1"
            for panel, dist in enumerate(dists):
                ax = axes.flat[panel]
                all_values: list[float | None] = []
                for comparison, label in COMPARISONS:
                    xs, ys, statuses, abnormal, coverages = collect_series(
                        summary, fmt, dist, comparison, metric
                    )
                    finite_ys = [y if y is not None and math.isfinite(y) else None for y in ys]
                    all_values.extend(finite_ys)
                    (line,) = ax.plot(
                        xs,
                        _as_plot_values(finite_ys),
                        marker=COMPARISON_MARKERS.get(comparison, "o"),
                        ms=3,
                        lw=1.2,
                        color=COMPARISON_COLORS[comparison],
                        label=label,
                    )
                    handles.setdefault(label, line)
                    subset_xs, subset_ys = collect_subset_series(summary, fmt, dist, comparison, metric)
                    if subset_xs:
                        plotted_subset = True
                        (subset_line,) = ax.plot(
                            subset_xs,
                            subset_ys,
                            ls=":",
                            lw=1.0,
                            alpha=0.7,
                            color=COMPARISON_COLORS[comparison],
                            label=f"{label} [finite subset]",
                        )
                        handles.setdefault(f"{label} [finite subset]", subset_line)
                    undefined_x = [x for x, y in zip(xs, finite_ys) if y is None]
                    if undefined_x:
                        plotted_undefined = True
                        ax.plot(undefined_x, [0.0] * len(undefined_x), marker="x", ls="none", color="red", ms=6)
                    for x, is_abnormal in zip(xs, abnormal):
                        if is_abnormal:
                            ax.axvspan(x * 0.97, x * 1.03, color="red", alpha=0.07, lw=0)
                note = _failure_note(summary, fmt, dist)
                ax.set_title(dist if not note else f"{dist}\ninvalid: {note}", fontsize=8)
                set_x_axis(ax, summary)
                scale = apply_metric_scale(ax, metric, all_values)
                if metric != "match_rate":
                    scale_note = scale
                ax.axhline(0.0, color="black", lw=0.5, alpha=0.6)
                ax.grid(True, which="both", alpha=0.25)
                ax.set_ylabel(METRIC_TITLES[metric])
            for panel in range(len(dists), axes.size):
                axes.flat[panel].axis("off")
            if plotted_undefined:
                handles["undefined / failed whole-batch value"] = plt.Line2D(
                    [], [], marker="x", ls="none", color="red", label="undefined / failed whole-batch value"
                )
            handles["abnormal events"] = plt.Line2D(
                [], [], marker="s", ls="none", color="red", alpha=0.25, label="abnormal events (shaded)"
            )
            fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, frameon=False)
            note_line = (
                "y scale: " + scale_note + "; zero drawn at 0; red x = undefined/failed whole-batch value; "
                "alignment_only uses exact E_align (Q_B = S_B + E_align, never rounded to FP32)"
            )
            if plotted_subset:
                note_line += "; dotted = finite subset only (diagnostic, not whole-batch)"
            fig.suptitle(
                f"{fmt} | {METRIC_TITLES[metric]}\n{run_subtitle(summary)}\n{note_line}",
                y=0.995,
            )
            fig.tight_layout(rect=(0, 0.06, 1, 0.90))
            stem = f"{_prefix(summary)}_{fmt}_{metric}"
            for suffix in (".png", ".svg"):
                path = out / f"{stem}{suffix}"
                fig.savefig(path, bbox_inches="tight")
                written.append(path)
            plt.close(fig)
    return written


def plot_stage_diagnostics(summary: dict, out_dir: str | Path) -> list[Path]:
    plt = configure_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = format_names(summary)
    dists = distribution_names(summary)
    written: list[Path] = []
    for fmt in formats:
        for stage in ("E_align", "E_acc"):
            fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2))
            handles: dict[str, object] = {}
            scale_note = ""
            for panel, dist in enumerate(dists):
                ax = axes.flat[panel]
                all_values: list[float | None] = []
                failed_points = 0
                for stat in ("mean", "rms"):
                    xs: list[int] = []
                    ys: list[float | None] = []
                    subset_xs: list[int] = []
                    subset_ys: list[float] = []
                    for cell in summary["cells"]:
                        if cell["format"] != fmt or cell["distribution"] != dist:
                            continue
                        payload = cell["stage"][stage]
                        xs.append(int(cell["x"]))
                        ys.append(payload.get(stat))
                        subset = payload.get("finite_subset")
                        if subset and subset.get(stat) is not None:
                            subset_xs.append(int(cell["x"]))
                            subset_ys.append(subset[stat])
                        if payload.get("status") != "ok":
                            failed_points += 1
                    order = sorted(range(len(xs)), key=lambda i: xs[i])
                    xs = [xs[i] for i in order]
                    ys = [ys[i] for i in order]
                    finite_ys = [y if y is not None and math.isfinite(y) else None for y in ys]
                    all_values.extend(finite_ys)
                    (line,) = ax.plot(
                        xs,
                        _as_plot_values(finite_ys),
                        marker="o",
                        ms=3,
                        lw=1.2,
                        color=STAGE_COLORS[stat],
                        label=f"{stage} {stat}",
                    )
                    handles.setdefault(f"{stage} {stat}", line)
                    if subset_xs:
                        (subset_line,) = ax.plot(
                            subset_xs,
                            subset_ys,
                            ls=":",
                            lw=1.0,
                            alpha=0.7,
                            color=STAGE_COLORS[stat],
                            label=f"{stage} {stat} [finite subset]",
                        )
                        handles.setdefault(f"{stage} {stat} [finite subset]", subset_line)
                    undefined_x = [x for x, y in zip(xs, finite_ys) if y is None]
                    if undefined_x:
                        ax.plot(undefined_x, [0.0] * len(undefined_x), marker="x", ls="none", color="red", ms=6)
                title = dist if not failed_points else f"{dist}\ninvalid/partial: {failed_points} x-points"
                ax.set_title(title, fontsize=8)
                set_x_axis(ax, summary)
                scale_note = apply_metric_scale(ax, "mean_signed_error", all_values)
                ax.axhline(0.0, color="black", lw=0.5, alpha=0.6)
                ax.grid(True, which="both", alpha=0.25)
                ax.set_ylabel(f"{stage} ({scale_note})")
            for panel in range(len(dists), axes.size):
                axes.flat[panel].axis("off")
            fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, frameon=False)
            fig.suptitle(
                f"{fmt} | stage diagnostic {stage}\n{run_subtitle(summary)}\n"
                "E_align = sum(q_b - s_b); E_acc = sum(delta_b); "
                "E_acc undefined after non-finite accumulation; dotted = finite subset only",
                y=0.995,
            )
            fig.tight_layout(rect=(0, 0.06, 1, 0.90))
            stem = f"{_prefix(summary)}_{fmt}_stage_{stage}"
            for suffix in (".png", ".svg"):
                path = out / f"{stem}{suffix}"
                fig.savefig(path, bbox_inches="tight")
                written.append(path)
            plt.close(fig)
    return written


def plot_all(summary: dict, out_dir: str | Path) -> list[Path]:
    written = plot_metric_grids(summary, out_dir)
    written.extend(plot_stage_diagnostics(summary, out_dir))
    return written
