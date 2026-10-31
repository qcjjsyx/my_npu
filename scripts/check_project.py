#!/usr/bin/env python3
"""Check migration integrity, local links, regressions, vectors and small experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ("integrity", "links", "tests", "vectors", "smoke")


def identity(path: Path) -> dict:
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"kind": "file", "size": path.stat().st_size, "sha256": digest.hexdigest()}


def check_integrity() -> dict:
    manifest = json.loads((ROOT / "docs/project/migration_manifest.json").read_text())
    errors, adapted, preserved = [], 0, 0
    for entry in manifest["files"]:
        path = ROOT / entry["new_path"]
        if not path.exists() and not path.is_symlink():
            errors.append(f"missing: {entry['new_path']}")
        elif entry.get("intentional_change"):
            adapted += 1
            if identity(path) != entry["after"]:
                errors.append(f"changed since migration: {entry['new_path']}")
        elif identity(path) != entry["before"]:
            errors.append(f"integrity mismatch: {entry['new_path']}")
        else:
            preserved += 1
    if errors:
        raise RuntimeError("\n".join(errors))
    return {"preserved": preserved, "intentionally_adapted": adapted,
            "total": len(manifest["files"])}


def markdown_destinations(text: str):
    # Skip fenced code, and allow balanced parentheses in local PDF filenames.
    text = re.sub(r"```[\s\S]*?```", "", text)
    for match in re.finditer(r"!?\[[^\]\n]*\]\(", text):
        start, depth = match.end(), 1
        index = start
        while index < len(text) and text[index] != "\n":
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    yield text[start:index].strip().strip("<>")
                    break
            index += 1


def check_links() -> dict:
    docs = [ROOT / "README.md", ROOT / "AGENTS.md"]
    for name in ("docs", "thesis", "hardware", "assets", "references", "archive/handoffs"):
        if name == "references":
            docs += list((ROOT / name).glob("*.md"))
            docs += list((ROOT / name / "papers").glob("*.md"))
            docs += list((ROOT / name / "projects").glob("*.md"))
        else:
            docs += list((ROOT / name).rglob("*.md"))
    errors, links = [], 0
    for path in docs:
        for url in markdown_destinations(path.read_text(encoding="utf-8")):
            if url.startswith("#") or re.match(r"^[a-zA-Z][\w+.-]*:", url):
                continue
            links += 1
            target = path.parent / unquote(url.split("#", 1)[0])
            if not target.exists():
                errors.append(f"{path.relative_to(ROOT)} -> {url}")
    if errors:
        raise RuntimeError("Broken local links:\n" + "\n".join(errors))
    return {"documents": len(docs), "local_links": links}


def execute(command: list[str], log: Path, timeout: int = 300) -> str:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    env["MPLCONFIGDIR"] = str(ROOT / "results/verification/.matplotlib")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=timeout)
    output = result.stdout + result.stderr
    log.write_text(output, encoding="utf-8")
    print(output[-5000:], end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {log.relative_to(ROOT)}")
    return result.stdout


def check_smoke(log_dir: Path, stamp: str) -> dict:
    import yaml
    configs = (
        ("smoke_block.yaml", "run", "analyze", "dot_product"),
        ("mx_smoke.yaml", "run", "mx-analyze", "mxfp8"),
        ("mx_align_smoke.yaml", "run", "mx-align-analyze", "mx_alignment"),
        ("widthselect_smoke.yaml", "width-run", "width-analyze", "widthselect"),
    )
    outputs = []
    for filename, run_command, analyze_command, family in configs:
        config = yaml.safe_load((ROOT / "experiments/configs" / filename).read_text())
        config["run_name"] = f"migration_check_{stamp}"
        if family == "dot_product":
            config.update(samples_per_seed=1, formats=["fp8_e4m3"],
                          distributions=["laplace"], B_list=[1, 2])
        cfg_path = log_dir / filename
        cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
        cli = [sys.executable, "-m", "mac_experiment"]
        output = execute([*cli, run_command, "--config", str(cfg_path), "--quiet"],
                         log_dir / f"{family}_run.log")
        run_dir = ROOT / output.strip().splitlines()[-1]
        expected = ROOT / f"results/numerical/{family}/runs"
        if run_dir.parent.resolve() != expected.resolve():
            raise RuntimeError(f"unexpected output directory: {run_dir}")
        args = [*cli, analyze_command, "--run", str(run_dir)]
        if family == "widthselect":
            args.append("--plots")
        analyzed = execute(args, log_dir / f"{family}_analysis.log")
        if family == "dot_product":
            execute([*cli, "plot", "--run", str(run_dir)], log_dir / "dot_product_plot.log")
            plot_dir = run_dir / "plots"
        else:
            analysis_dir = ROOT / analyzed.strip().splitlines()[-1]
            if analysis_dir.parent.resolve() != (expected.parent / "analysis").resolve():
                raise RuntimeError(f"unexpected analysis directory: {analysis_dir}")
            plot_dir = analysis_dir / "plots"
        if not list(plot_dir.glob("*.png")):
            raise RuntimeError(f"missing plot output: {plot_dir}")
        outputs.append(str(run_dir.relative_to(ROOT)))
    return {"runs": outputs, "families": len(outputs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    log_dir = ROOT / "results/verification" / f"check_{stamp}"
    log_dir.mkdir(parents=True)
    report = {"timestamp_asia_shanghai": stamp, "python": sys.executable, "checks": {}}
    for name in args.checks:
        print(f"[{name}]", flush=True)
        try:
            if name == "integrity":
                detail = check_integrity()
            elif name == "links":
                detail = check_links()
            elif name == "tests":
                execute([sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider"],
                        log_dir / "pytest.log")
                detail = {"log": str((log_dir / "pytest.log").relative_to(ROOT))}
            elif name == "vectors":
                execute([sys.executable, "hardware/verification/tools/run_txt_vectors.py",
                         "--build", str(log_dir / "txt")], log_dir / "txt.log")
                detail = {"log": str((log_dir / "txt.log").relative_to(ROOT))}
            else:
                detail = check_smoke(log_dir, stamp)
            report["checks"][name] = {"status": "passed", **detail}
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            report["checks"][name] = {"status": "failed", "error": str(exc)}
            print(str(exc), file=sys.stderr)
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["checks"].values()) else "failed"
    (log_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['status']}: {(log_dir / 'report.json').relative_to(ROOT)}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
