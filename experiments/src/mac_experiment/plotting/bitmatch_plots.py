"""Bit-match and ULP-distance figures for the offline bitmatch analysis.

Only the primary pairs are plotted (FP32 and BF16 separately).  Each figure
family has one subplot per distribution:

* bit match / mismatch rate vs B (or W);
* conditional ULP distance p50/p90/p99/max for finite bit-mismatch samples;
* distance bucket rates over finite pairs.

Missing conditional statistics (no mismatch samples) are shown as open red
circles at the bottom and labelled; they are never plotted as zero.
"""

from __future__ import annotations

import math
from pathlib import Path

from .style import configure_matplotlib, distribution_names, format_names, set_x_axis

BUCKET_COLORS = {
    "0": "#2ca02c",
    "1": "#9edae5",
    "2": "#1f77b4",
    "3-4": "#ff7f0e",
    ">=5": "#d62728",
}
BUCKET_LABELS = ("0", "1", "2", "3-4", ">=5")
ULP_SERIES = (
    ("p50", "#1f77b4"),
    ("p90", "#2ca02c"),
    ("p99", "#ff7f0e"),
    ("max", "#d62728"),
)
PAIR_TITLES = {"fp32": "FP32: o32 vs r_B", "bf16": "BF16: o16 vs r16"}


def _prefix(summary: dict) -> str:
    return "b_sweep" if summary["x_name"] == "B" else "w_sweep"


def _pair_cells(summary: dict, fmt: str, dist: str, pair: str):
    cells = [
        cell
        for cell in summary["cells"]
        if cell["format"] == fmt and cell["distribution"] == dist
    ]
    return sorted(cells, key=lambda cell: cell["x"])


def _subtitle(summary: dict) -> str:
    fixed = summary["fixed"]
    fixed_text = f"fixed B={fixed['B']}" if summary["x_name"] == "W" else f"fixed W={fixed['W']}"
    config = summary["config"]
    n_samples = len(config["seeds"]) * config["samples_per_seed"]
    return (
        f"{fixed_text} | samples per cell={n_samples} | RNE ties-to-even | subnormals kept (no FTZ) | "
        "overflow -> +/-Inf | bit match compares full encoding (+0 != -0)"
    )


def _event_note(payload: dict) -> str:
    events = payload["events"]
    parts = [f"{key}={count}" for key, count in events.items() if count]
    return "events: " + ", ".join(parts) if parts else ""


def plot_bitmatch(summary: dict, out_dir: str | Path) -> list[Path]:
    plt = configure_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    formats = format_names(summary)
    dists = distribution_names(summary)
    for fmt in formats:
        for pair in ("fp32", "bf16"):
            written.extend(_plot_rates(plt, summary, out, fmt, pair, dists))
            written.extend(_plot_conditional_ulp(plt, summary, out, fmt, pair, dists))
            written.extend(_plot_buckets(plt, summary, out, fmt, pair, dists))
    return written


def _plot_rates(plt, summary, out, fmt, pair, dists):
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2))
    handles = {}
    for panel, dist in enumerate(dists):
        ax = axes.flat[panel]
        cells = _pair_cells(summary, fmt, dist, pair)
        xs = [cell["x"] for cell in cells]
        payloads = [cell["pairs"][pair] for cell in cells]
        match = [payload["bit_match_rate"] for payload in payloads]
        mismatch = [payload["bit_mismatch_rate"] for payload in payloads]
        (line_mismatch,) = ax.plot(
            xs,
            [math.nan if value is None else value for value in mismatch],
            marker="o",
            ms=3,
            lw=1.2,
            color="#d62728",
            label="bit mismatch rate",
        )
        (line_match,) = ax.plot(
            xs,
            [math.nan if value is None else value for value in match],
            marker="o",
            ms=3,
            lw=1.2,
            color="#2ca02c",
            label="bit match rate",
        )
        handles.setdefault("bit mismatch rate", line_mismatch)
        handles.setdefault("bit match rate", line_match)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="black", lw=0.5, alpha=0.6)
        ax.grid(True, which="both", alpha=0.25)
        set_x_axis(ax, summary)
        event_note = _event_note(payloads[-1]) if payloads else ""
        ax.set_title(dist if not event_note else f"{dist}\n{event_note}", fontsize=8)
        ax.set_ylabel("rate (denominator n_total)")
    for panel in range(len(dists), axes.size):
        axes.flat[panel].axis("off")
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"{fmt} | {PAIR_TITLES[pair]} | bit match\n{_subtitle(summary)}\n"
        "denominator = all records in the cell; rates are not conditioned on finiteness",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    stem = f"{_prefix(summary)}_{fmt}_{pair}_bitmatch_rate"
    return _save(fig, plt, out, stem)


def _plot_conditional_ulp(plt, summary, out, fmt, pair, dists):
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2))
    handles = {}
    for panel, dist in enumerate(dists):
        ax = axes.flat[panel]
        cells = _pair_cells(summary, fmt, dist, pair)
        xs = [cell["x"] for cell in cells]
        payloads = [cell["pairs"][pair] for cell in cells]
        for key, color in ULP_SERIES:
            values = [
                None if payload["conditional_ulp"][key] is None else float(payload["conditional_ulp"][key])
                for payload in payloads
            ]
            (line,) = ax.plot(
                xs,
                [math.nan if value is None else value for value in values],
                marker="o",
                ms=3,
                lw=1.2,
                color=color,
                label=f"conditional ULP {key}",
            )
            handles.setdefault(f"conditional ULP {key}", line)
        missing_x = [
            cell["x"] for cell, payload in zip(cells, payloads) if payload["conditional_ulp"]["status"] != "ok"
        ]
        if missing_x:
            ax.plot(
                missing_x,
                [0.0] * len(missing_x),
                marker="o",
                ls="none",
                mfc="none",
                mec="red",
                ms=6,
                label="no mismatch samples",
            )
            handles.setdefault(
                "no mismatch samples",
                plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="red", label="no mismatch samples"),
            )
        finite_values = [
            float(value)
            for payload in payloads
            for value in payload["conditional_ulp"].values()
            if isinstance(value, int)
        ]
        if finite_values:
            ax.set_yscale("symlog", linthresh=1)
        ax.axhline(0.0, color="black", lw=0.5, alpha=0.6)
        ax.grid(True, which="both", alpha=0.25)
        set_x_axis(ax, summary)
        n_mismatch = sum(payload["counts"]["n_finite_mismatch"] for payload in payloads)
        ax.set_title(f"{dist}\nfinite mismatch samples: {n_mismatch}", fontsize=8)
        ax.set_ylabel("ULP distance (conditional)")
    for panel in range(len(dists), axes.size):
        axes.flat[panel].axis("off")
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"{fmt} | {PAIR_TITLES[pair]} | conditional ULP distance\n{_subtitle(summary)}\n"
        "only finite pairs with bit mismatch; denominator n_finite_mismatch; "
        "open red circle = no mismatch samples (value is null, not 0)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    stem = f"{_prefix(summary)}_{fmt}_{pair}_conditional_ulp"
    return _save(fig, plt, out, stem)


def _plot_buckets(plt, summary, out, fmt, pair, dists):
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2))
    for panel, dist in enumerate(dists):
        ax = axes.flat[panel]
        cells = _pair_cells(summary, fmt, dist, pair)
        labels = [str(cell["x"]) for cell in cells]
        positions = range(len(cells))
        bottom = [0.0] * len(cells)
        for bucket in BUCKET_LABELS:
            rates = []
            for cell in cells:
                rate = cell["pairs"][pair]["distance_buckets"]["buckets"][bucket]["rate"]
                rates.append(0.0 if rate is None else rate)
            ax.bar(positions, rates, bottom=bottom, color=BUCKET_COLORS[bucket], label=f"ULP {bucket}")
            bottom = [base + rate for base, rate in zip(bottom, rates)]
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("B" if summary["x_name"] == "B" else "W")
        ax.grid(True, axis="y", alpha=0.25)
        denominators = {cell["pairs"][pair]["counts"]["n_finite_pairs"] for cell in cells}
        ax.set_title(f"{dist}\nfinite pairs per point: {sorted(denominators)}", fontsize=8)
        ax.set_ylabel("fraction of finite pairs")
    for panel in range(len(dists), axes.size):
        axes.flat[panel].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"{fmt} | {PAIR_TITLES[pair]} | ULP distance buckets\n{_subtitle(summary)}\n"
        "denominator = finite pairs; bucket 0 includes exact bit matches and signed-zero mismatches",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    stem = f"{_prefix(summary)}_{fmt}_{pair}_ulp_buckets"
    return _save(fig, plt, out, stem)


def _save(fig, plt, out: Path, stem: str) -> list[Path]:
    written = []
    for suffix in (".png", ".svg"):
        path = out / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
