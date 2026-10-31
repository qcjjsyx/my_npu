#!/usr/bin/env python3
"""Run the three width-select experiments sequentially with live progress.

Each experiment stays an isolated run (one mode per run, fresh inputs, its own
run directory / schema / analysis), so the script only orchestrates:

    align  ->  width-run + width-analyze
    acc    ->  width-run + width-analyze
    joint  ->  width-run + width-analyze

Progress is parsed from the CLI's own ``i/N matrix pairs`` lines and rendered as
a bar when stderr is a TTY (``--plain`` forces plain lines, useful for logs).
Analysis outputs are written to ``<output_root>/../analysis/``,
matching the manual workflow; per-cell metrics and passing candidates are
printed compactly when ``pass_criteria`` is configured.

Examples:
    .venv/bin/python experiments/scripts/run_widthselect_all.py                 # smoke preset
    .venv/bin/python experiments/scripts/run_widthselect_all.py --overwrite     # re-run the smoke preset
    .venv/bin/python experiments/scripts/run_widthselect_all.py --preset formal
    .venv/bin/python experiments/scripts/run_widthselect_all.py --skip-analysis
    .venv/bin/python experiments/scripts/run_widthselect_all.py \
        --align-config experiments/configs/widthselect_smoke.yaml \
        --acc-config experiments/configs/widthselect_acc_smoke.yaml \
        --joint-config experiments/configs/widthselect_joint_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRESS_RE = re.compile(r"\]\s+(\d+)/(\d+)\s+matrix pairs")
BAR_WIDTH = 26
EXPERIMENTS = ("align", "acc", "joint")

PRESETS = {
    "smoke": {
        "align": "experiments/configs/widthselect_smoke.yaml",
        "acc": "experiments/configs/widthselect_acc_smoke.yaml",
        "joint": "experiments/configs/widthselect_joint_smoke.yaml",
    },
    "formal": {
        "align": "experiments/configs/widthselect_align_formal_template.yaml",
        "acc": "experiments/configs/widthselect_acc_formal_template.yaml",
        "joint": "experiments/configs/widthselect_joint_formal_template.yaml",
    },
}


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def parse_progress(line: str) -> tuple[int, int] | None:
    match = PROGRESS_RE.search(line)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _render_bar(done: int, total: int) -> str:
    total = max(total, 1)
    filled = int(BAR_WIDTH * done / total)
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + f"] {done}/{total} pairs"


class Progress:
    """Single-line progress bar on a TTY, plain lines otherwise."""

    def __init__(self, plain: bool) -> None:
        self.interactive = (not plain) and sys.stderr.isatty()
        self._last_width = 0

    def _write(self, text: str, end: str) -> None:
        if self.interactive:
            padded = text + " " * max(0, self._last_width - len(text))
            self._last_width = len(text)
            sys.stderr.write("\r" + padded + end)
        else:
            sys.stderr.write(text + end)
        sys.stderr.flush()

    def start(self, label: str) -> None:
        self._write(f"      {label} starting...", "" if self.interactive else "\n")

    def update(self, done: int, total: int, elapsed: float) -> None:
        text = f"      run  {_render_bar(done, total)}  {format_duration(elapsed)}"
        self._write(text, "" if self.interactive else "\n")

    def note(self, text: str) -> None:
        if self.interactive:
            sys.stderr.write("\r" + " " * self._last_width + "\r")
            self._last_width = 0
        sys.stderr.write(text + "\n")
        sys.stderr.flush()


def _child_env() -> dict:
    env = dict(os.environ)
    src = str(REPO_ROOT / "experiments" / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    return env


def _load_plan(config_path: Path) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "src"))
    from mac_experiment.widthselect.config import load_widthselect_config

    config = load_widthselect_config(config_path)
    candidates = len(config.W_align_list) or len(config.W_acc_list) or len(config.joint_pairs)
    pairs = config.n_pairs()
    rows = pairs * len(config.K_list) * len(config.formats) * (1 + candidates)
    output_root = Path(config.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return {
        "mode": config.mode,
        "pairs": pairs,
        "K_list": list(config.K_list),
        "formats": list(config.formats),
        "candidates": candidates,
        "rows": rows,
        "output_root": output_root,
        "run_name": config.run_name,
    }


def _existing_run_dir(plan: dict) -> Path | None:
    if not plan["run_name"]:
        return None
    return plan["output_root"] / plan["run_name"]


def _purge_existing_run(path: Path | None) -> None:
    if path is None or not path.is_dir():
        return
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {path}: manifest.json is missing")
    schema = json.loads(manifest_path.read_text(encoding="utf-8")).get("schema_version")
    if schema != "mac-width-v1":
        raise SystemExit(f"refusing to overwrite {path}: schema {schema!r} is not mac-width-v1")
    shutil.rmtree(path)
    print(f"      overwrite  removed {path}")


def _stream(cmd: list[str], progress: Progress, label: str) -> tuple[int, str]:
    started = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    progress.start(label)
    last_update = 0.0
    assert process.stderr is not None
    for line in process.stderr:
        parsed = parse_progress(line)
        if parsed is not None:
            done, total = parsed
            now = time.monotonic()
            if now - last_update >= 0.05 or done == total:
                progress.update(done, total, now - started)
                last_update = now
        else:
            progress.note("      | " + line.rstrip())
    process.wait()
    stdout = process.stdout.read() if process.stdout is not None else ""
    if process.returncode == 0:
        progress.note(f"      done in {format_duration(time.monotonic() - started)}")
    return process.returncode, stdout


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _median(values: list[float]) -> float | None:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return None
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def summarize_analysis(summary: dict, limit: int = 4) -> list[str]:
    lines: list[str] = []
    cells = summary.get("cells", [])
    rates = [
        cell["metrics"]["bit_match_rate"]
        for cell in cells
        if cell["metrics"]["bit_match_rate"] is not None
    ]
    nonfinite = sum(cell["metrics"]["counts"]["n_nonfinite_candidate"] for cell in cells)
    markers = sum(cell.get("state_marker_count", 0) for cell in cells)
    if rates:
        lines.append(
            f"metrics  cells={len(cells)} bit_match min/med/max = "
            f"{min(rates):.3f}/{_median(rates):.3f}/{max(rates):.3f}"
        )
    else:
        lines.append(f"metrics  cells={len(cells)} (no bit-match denominator)")
    if nonfinite or markers:
        lines.append(f"events   nonfinite candidates={nonfinite}, acc overflow markers={markers}")

    recommendation = summary.get("recommendation")
    if not recommendation:
        lines.append("criteria none (report-only; no acceptance claim)")
        return lines
    group_total = len(recommendation["by_group"])
    passing = []
    for group in recommendation["by_group"]:
        if group.get("passing_candidates"):
            passing.append(group)
        elif group.get("minimal_passing_candidate") is not None:
            passing.append(group)
    lines.append(
        f"criteria {recommendation['criteria']} -> {len(passing)}/{group_total} groups with a "
        "passing candidate"
    )
    for group in passing[:limit]:
        if "passing_candidates" in group:
            detail = ", ".join(
                f"(w_align={c['w_align']},w_acc={c['w_acc']})" for c in group["passing_candidates"][:3]
            )
        else:
            candidate = group["tested"][0]["candidate"] if group.get("tested") else {}
            key = "w_acc" if candidate.get("w_align") is None else "w_align"
            detail = f"{key}={group['minimal_passing_candidate']}"
        lines.append(
            f"         {group['format']} {group['distribution_pair']} K={group['K']} -> {detail}"
        )
    if len(passing) > limit:
        lines.append(f"         ... and {len(passing) - limit} more groups")
    return lines


def run_experiment(
    name: str,
    config_path: Path,
    index: int,
    total: int,
    plain: bool,
    skip_analysis: bool,
    overwrite: bool,
    plots: bool,
) -> dict:
    progress = Progress(plain)
    plan = _load_plan(config_path)
    print(f"\n[{index}/{total}] {plan['mode']}  config={config_path}")
    print(
        f"      plan  pairs={plan['pairs']} K={plan['K_list']} formats={len(plan['formats'])} "
        f"candidates={plan['candidates']} -> ~{plan['rows']} snapshot rows"
    )
    if overwrite:
        _purge_existing_run(_existing_run_dir(plan))
    cmd = [sys.executable, "-m", "mac_experiment", "width-run", "--config", str(config_path)]
    returncode, stdout = _stream(cmd, progress, "run ")
    if returncode != 0:
        return {"name": name, "status": "failed", "stage": "run", "config": str(config_path)}
    run_dir = Path(_last_line(stdout))
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    print(f"      run dir  {run_dir}")
    result = {"name": name, "status": "ok", "run_dir": run_dir, "analysis_dir": None}

    if skip_analysis:
        return result
    stamp = time.strftime("%Y%m%d_%H%M%S")
    analysis_out = (
        run_dir.parent.parent / "analysis" / f"{run_dir.name}_analysis_{stamp}"
    )
    analyze_cmd = [
        sys.executable,
        "-m",
        "mac_experiment",
        "width-analyze",
        "--run",
        str(run_dir),
        "--out",
        str(analysis_out),
    ]
    if plots:
        analyze_cmd.append("--plots")
    returncode, stdout = _stream(analyze_cmd, progress, "analyze ")
    if returncode != 0:
        result["status"] = "failed"
        result["stage"] = "analyze"
        return result
    analysis_dir = Path(_last_line(stdout))
    if not analysis_dir.is_absolute():
        analysis_dir = REPO_ROOT / analysis_dir
    result["analysis_dir"] = analysis_dir
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    for line in summarize_analysis(summary):
        print("      " + line)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the align / acc / joint width-select experiments with progress.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--align-config", default=None, help="override the align config path")
    parser.add_argument("--acc-config", default=None, help="override the acc config path")
    parser.add_argument("--joint-config", default=None, help="override the joint config path")
    parser.add_argument("--skip-analysis", action="store_true", help="run experiments only")
    parser.add_argument("--plots", action="store_true", help="render bit-match / ULP figures during analysis")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove the configured run directories first (only mac-width-v1 runs with a fixed run_name)",
    )
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failure")
    parser.add_argument("--plain", action="store_true", help="force plain progress lines (no TTY bar)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "align": args.align_config,
        "acc": args.acc_config,
        "joint": args.joint_config,
    }
    configs = {
        name: Path(overrides[name]) if overrides[name] else REPO_ROOT / PRESETS[args.preset][name]
        for name in EXPERIMENTS
    }
    print(f"=== width-select batch: preset={args.preset}, {len(EXPERIMENTS)} experiments ===")
    print("(each experiment is an isolated run with fresh inputs)")
    results = []
    started = time.monotonic()
    for index, name in enumerate(EXPERIMENTS, start=1):
        try:
            result = run_experiment(
                name,
                configs[name],
                index,
                len(EXPERIMENTS),
                plain=args.plain,
                skip_analysis=args.skip_analysis,
                overwrite=args.overwrite,
                plots=args.plots,
            )
        except KeyboardInterrupt:
            print("\ninterrupted by user")
            return 130
        results.append(result)
        if result["status"] != "ok" and args.fail_fast:
            break
    elapsed = format_duration(time.monotonic() - started)
    ok = [result for result in results if result["status"] == "ok"]
    failed = [result for result in results if result["status"] != "ok"]
    print(f"\n=== summary: {len(ok)}/{len(EXPERIMENTS)} succeeded in {elapsed} ===")
    for result in results:
        state = result["status"]
        detail = ""
        if state != "ok":
            detail = f" (stage={result.get('stage')}, config={result.get('config')})"
        elif result.get("analysis_dir"):
            detail = f" -> {result['analysis_dir']}"
        elif result.get("run_dir"):
            detail = f" -> {result['run_dir']}"
        print(f"  {result['name']:>5}: {state}{detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
