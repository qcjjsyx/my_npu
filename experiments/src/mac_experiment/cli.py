"""Command line interface.

Heavy imports (experiments, sampling, plotting) are performed inside the
individual command handlers so that offline commands such as ``enhance`` do
not import the sampler or the numerical execution engine at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .version import PACKAGE_VERSION, SCHEMA_VERSION, STREAM_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mac-experiment",
        description="Fixed-window MAC accumulation precision experiments (FP8 E4M3 / TF32)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="execute a configured sweep (width_sweep or block_sweep)")
    run_p.add_argument("--config", required=True, help="path to a YAML run config")
    run_p.add_argument("--quiet", action="store_true", help="suppress progress output")

    analyze_p = sub.add_parser("analyze", help="recompute metrics from saved sample records")
    analyze_p.add_argument("--run", required=True, help="run directory")
    analyze_p.add_argument("--no-write", action="store_true", help="do not write summary_recomputed.json")

    enhance_p = sub.add_parser(
        "enhance",
        help="offline alignment_only enhancement of a saved run (no engine, no re-sampling)",
    )
    enhance_p.add_argument("--run", required=True, help="source run directory")
    enhance_p.add_argument(
        "--out",
        default=None,
        help="output directory (default: results/numerical/dot_product/analysis/alignment_only/<run_id>_alignment_only_<timestamp>)",
    )
    enhance_p.add_argument("--no-plots", action="store_true", help="skip figure generation")

    bitmatch_p = sub.add_parser(
        "bitmatch",
        help="offline bit-match / ULP-distance analysis of a saved run (no engine, no re-sampling)",
    )
    bitmatch_p.add_argument("--run", required=True, help="source run directory")
    bitmatch_p.add_argument(
        "--out",
        default=None,
        help="output directory (default: results/numerical/dot_product/analysis/bitmatch/<run_id>_bitmatch_ulp_<timestamp>)",
    )
    bitmatch_p.add_argument("--no-plots", action="store_true", help="skip figure generation")

    mx_p = sub.add_parser(
        "mx-analyze",
        help="offline output-quantisation analysis of a saved MX matrix run (no engine, no re-sampling)",
    )
    mx_p.add_argument("--run", required=True, help="source MX run directory")
    mx_p.add_argument(
        "--out",
        default=None,
        help="output directory (default: results/numerical/mxfp8/analysis/<run_id>_quant_<timestamp>)",
    )
    mx_p.add_argument("--no-plots", action="store_true", help="skip figure generation")

    align_p = sub.add_parser(
        "mx-align-analyze",
        help="offline analysis of a fixed-W_align K/B sweep run (no engine, no re-sampling)",
    )
    align_p.add_argument("--run", required=True, help="source alignment run directory")
    align_p.add_argument(
        "--out",
        default=None,
        help="output directory (default: results/numerical/mx_alignment/analysis/<run_id>_analysis_<timestamp>)",
    )
    align_p.add_argument("--no-plots", action="store_true", help="skip figure generation")

    trace_p = sub.add_parser(
        "mx-align-trace",
        help="export filtered per-dot16 alignment traces from a run that stores MX codes",
    )
    trace_p.add_argument("--run", required=True, help="alignment or MX run directory")
    trace_p.add_argument(
        "--source-run",
        default=None,
        help=(
            "override the recorded source MX input directory (reused alignment runs); "
            "the recorded file identities are still verified"
        ),
    )
    trace_p.add_argument("--sample-id", default=None, help="only this sample_id")
    trace_p.add_argument("--row", type=int, required=True, help="output row coordinate")
    trace_p.add_argument("--col", type=int, required=True, help="output column coordinate")
    trace_p.add_argument("--block-start", type=int, required=True, help="first dot16 block (inclusive)")
    trace_p.add_argument("--block-end", type=int, default=None, help="last dot16 block (inclusive; default start)")
    trace_p.add_argument("--w-align", type=int, default=None, help="override W_align (default: manifest/21)")
    trace_p.add_argument("--out", default=None, help="write JSONL here instead of stdout")

    plot_p = sub.add_parser("plot", help="render figures from a saved summary")
    plot_p.add_argument("--run", required=True, help="run directory")
    plot_p.add_argument("--out", default=None, help="output directory (default: <run>/plots)")

    width_run_p = sub.add_parser(
        "width-run",
        help="execute a W_align / W_acc bit-width selection run (fresh inputs only)",
    )
    width_run_p.add_argument("--config", required=True, help="path to a YAML run config")
    width_run_p.add_argument("--quiet", action="store_true", help="suppress progress output")

    width_analyze_p = sub.add_parser(
        "width-analyze",
        help="offline bit-match / ULP analysis of a saved width-select run",
    )
    width_analyze_p.add_argument("--run", required=True, help="width-select run directory")
    width_analyze_p.add_argument(
        "--out",
        default=None,
        help="output directory (default: results/numerical/widthselect/analysis/<run_id>_analysis_<timestamp>)",
    )
    width_analyze_p.add_argument(
        "--plots", action="store_true", help="render bit-match / ULP figures under <out>/plots/ (needs matplotlib)"
    )

    width_replay_p = sub.add_parser(
        "width-replay",
        help="export per-dot16 alignment diagnostics for one output coordinate",
    )
    width_replay_p.add_argument("--run", required=True, help="width-select run directory")
    width_replay_p.add_argument("--sample-id", required=True, help="matrix pair sample_id")
    width_replay_p.add_argument("--row", type=int, required=True, help="output row coordinate")
    width_replay_p.add_argument("--col", type=int, required=True, help="output column coordinate")
    width_replay_p.add_argument("--K", type=int, required=True, help="declared K snapshot")
    width_replay_p.add_argument("--w-align", type=int, default=None, help="W_align value (experiment 1 / joint)")
    width_replay_p.add_argument("--w-acc", type=int, default=None, help="W_acc value (experiment 2 / joint)")
    width_replay_p.add_argument("--format", default=None, help="target format (required when the run has several)")
    width_replay_p.add_argument("--out", default=None, help="write JSON here instead of stdout")

    dist_p = sub.add_parser("distributions", help="dump magnitude PMF metadata for a config")
    dist_p.add_argument("--config", required=True, help="path to a YAML run config")
    dist_p.add_argument("--out", default=None, help="write JSON metadata to this file")

    sub.add_parser("version", help="print package and schema versions")
    return parser


def _cmd_run(args) -> int:
    from .experiments import block_sweep, width_sweep
    from .experiments.config import MX_ALIGN_MODE, MX_MODE, load_config

    config = load_config(args.config)
    verbose = not args.quiet
    if config.mode == MX_MODE:
        from .experiments import mx_matrix_sweep

        run_dir = mx_matrix_sweep.run(config, verbose=verbose)
    elif config.mode == MX_ALIGN_MODE:
        from .experiments import mx_alignment_sweep

        run_dir = mx_alignment_sweep.run(config, verbose=verbose)
    elif config.mode == "width_sweep":
        run_dir = width_sweep.run(config, verbose=verbose)
    else:
        run_dir = block_sweep.run(config, verbose=verbose)
    print(run_dir)
    return 0


def _cmd_analyze(args) -> int:
    from .results.reanalyze import reanalyze

    result = reanalyze(args.run, write=not args.no_write)
    if result["matches_original_summary"]:
        print(f"re-analysis matches the original summary ({result['n_records_read']} records)")
        return 0
    print(f"re-analysis DIFFERS from the original summary ({len(result['differences'])} differences):")
    for line in result["differences"][:50]:
        print(f"  {line}")
    return 1


def _cmd_enhance(args) -> int:
    from .results.enhance import enhance_run

    out_dir = enhance_run(args.run, out_dir=args.out, write_plots=not args.no_plots)
    print(out_dir)
    return 0


def _cmd_bitmatch(args) -> int:
    from .results.bitmatch_analysis import analyze_bitmatch

    out_dir = analyze_bitmatch(args.run, out_dir=args.out, write_plots=not args.no_plots)
    print(out_dir)
    return 0


def _cmd_mx_analyze(args) -> int:
    from .results.mx_analysis import analyze_mx_run

    out_dir = analyze_mx_run(args.run, out_dir=args.out, write_plots=not args.no_plots)
    print(out_dir)
    return 0


def _cmd_mx_align_analyze(args) -> int:
    from .results.mx_alignment_analysis import analyze_mx_alignment_run

    out_dir = analyze_mx_alignment_run(args.run, out_dir=args.out, write_plots=not args.no_plots)
    print(out_dir)
    return 0


def _cmd_mx_align_trace(args) -> int:
    from pathlib import Path as _Path

    from .matrix.aligned_engine import trace_dot16
    from .matrix.tensor import MxTensor
    from .results.enhance import _file_identity
    from .results.mx_io import iter_mx_inputs
    import json as _json

    from .results.reader import load_manifest, run_path
    from .version import MX_ALIGN_DEFAULT, MX_ALIGN_SCHEMA_VERSION, MX_SCHEMA_VERSION

    path = run_path(args.run)
    manifest = load_manifest(path)
    schema = manifest.get("schema_version")
    block_end = args.block_start if args.block_end is None else args.block_end
    if block_end < args.block_start:
        raise ValueError("block-end must be >= block-start")

    run_w_align = None
    prefix_kmax = None
    identity_note = None
    if schema == MX_ALIGN_SCHEMA_VERSION:
        run_w_align = manifest.get("w_align")
        prefix_kmax = max(int(k) for k in manifest.get("K_list") or [])
        source_info = manifest.get("source") or {}
        if source_info.get("kind") == "reused":
            recorded = source_info.get("source_file_identity") or {}
            source_dir = _Path(args.source_run) if args.source_run else _Path(
                source_info.get("source_run_dir", "")
            )
            if not source_dir or not source_dir.exists():
                raise ValueError(f"source input directory not found: {source_dir}")
            for name, ident in recorded.items():
                candidate = source_dir / name
                if not candidate.exists():
                    raise ValueError(
                        f"source override {source_dir} is missing {name}; refusing to use a "
                        "same-named run with different files"
                    )
                current = _file_identity(candidate)
                if current["sha256"] != ident.get("sha256") or int(current["size"]) != int(
                    ident.get("size", -1)
                ):
                    raise ValueError(
                        f"source override {source_dir}/{name} does not match the identity recorded "
                        "by this alignment run; refusing to trace against different inputs"
                    )
            codes_iter = iter_mx_inputs(source_dir)
            identity_note = {
                "codes_from": str(source_dir),
                "source_identity_verified": True,
                "recorded_source_run_id": source_info.get("source_run_id"),
            }
        else:
            if args.source_run:
                raise ValueError(
                    "--source-run is only needed for reused alignment runs; this run stores its own codes"
                )
            codes_iter = iter_mx_inputs(path)
            identity_note = {"codes_from": str(path), "source_identity_verified": None}
    elif schema == MX_SCHEMA_VERSION:
        if args.source_run:
            raise ValueError("--source-run is not applicable to plain MX matrix runs")
        prefix_kmax = int(manifest.get("Kmax") or max(manifest.get("K_list") or [0]))
        codes_iter = iter_mx_inputs(path)
        identity_note = {"codes_from": str(path), "source_identity_verified": None}
    else:
        raise ValueError(
            f"mx-align-trace expects schema {MX_ALIGN_SCHEMA_VERSION!r} or {MX_SCHEMA_VERSION!r}, "
            f"got {schema!r}"
        )

    w_align = args.w_align if args.w_align is not None else (run_w_align or MX_ALIGN_DEFAULT)
    w_align_override = (
        args.w_align is not None and run_w_align is not None and int(args.w_align) != int(run_w_align)
    )
    if prefix_kmax is not None and (block_end + 1) * 16 > prefix_kmax:
        raise ValueError(
            f"block range [{args.block_start}, {block_end}] exceeds this run's executed K prefix "
            f"(Kmax={prefix_kmax}); no trajectory exists beyond it"
        )

    lines: list[str] = []
    matched = 0
    for record in codes_iter:
        if record.get("record_type") not in ("align_input_pair", "input_pair"):
            continue
        if args.sample_id and record.get("sample_id") != args.sample_id:
            continue
        x_payload = record.get("x") or {}
        w_payload = record.get("w") or {}
        if not x_payload.get("element_codes") or not w_payload.get("element_codes"):
            raise ValueError(
                f"{path} stores no MX codes (codes_in={record.get('codes_in')!r}); pass "
                "--source-run pointing at the source MX run that stores element/scale codes"
            )
        x = MxTensor.from_dict(x_payload)
        w = MxTensor.from_dict(w_payload)
        for block in range(args.block_start, block_end + 1):
            trace = trace_dot16(x, w, args.row, args.col, block, w_align=w_align)
            trace.update(
                {
                    "sample_id": record.get("sample_id"),
                    "trace_run": manifest.get("run_id"),
                    "run_w_align": run_w_align,
                    "w_align": w_align,
                    "w_align_override": w_align_override,
                    "prefix_Kmax": prefix_kmax,
                    **identity_note,
                }
            )
            lines.append(
                _json.dumps(trace, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
        matched += 1
    if matched == 0:
        raise ValueError("no input record matched the filters")
    output = "\n".join(lines) + "\n"
    if args.out:
        _Path(args.out).write_text(output, encoding="utf-8")
        print(f"wrote {len(lines)} dot16 traces to {args.out}")
    else:
        print(output, end="")
    return 0


def _cmd_width_run(args) -> int:
    from .widthselect.runner import run_from_file

    run_dir = run_from_file(args.config, verbose=not args.quiet)
    print(run_dir)
    return 0


def _cmd_width_analyze(args) -> int:
    from .widthselect.analysis import analyze_width_run

    out_dir = analyze_width_run(args.run, out_dir=args.out, write_plots=args.plots)
    print(out_dir)
    return 0


def _cmd_width_replay(args) -> int:
    from .widthselect.replay import replay_width_run, write_replay

    payload = replay_width_run(
        args.run,
        sample_id=args.sample_id,
        row=args.row,
        col=args.col,
        K=args.K,
        w_align=args.w_align,
        w_acc=args.w_acc,
        format=args.format,
    )
    write_replay(payload, args.out)
    return 0


def _cmd_plot(args) -> int:
    from .plotting import plot_all
    from .results.reader import load_manifest, load_summary, run_path
    from .version import MX_ALIGN_SCHEMA_VERSION, MX_SCHEMA_VERSION

    path = run_path(args.run)
    manifest = load_manifest(path)
    if manifest.get("schema_version") == MX_ALIGN_SCHEMA_VERSION:
        raise ValueError(
            "alignment runs are analysed and plotted with 'mx-align-analyze' (not 'plot')"
        )
    if manifest.get("schema_version") == MX_SCHEMA_VERSION:
        raise ValueError("MX matrix runs are analysed and plotted with 'mx-analyze' (not 'plot')")
    summary = load_summary(path)
    out_dir = Path(args.out) if args.out else path / "plots"
    written = plot_all(summary, out_dir)
    print(f"wrote {len(written)} files to {out_dir}")
    return 0


def _cmd_distributions(args) -> int:
    from .experiments.config import load_config
    from .formats import get_format
    from .results.writer import dumps_strict, write_json
    from .sampling.sampler import Sampler

    config = load_config(args.config)
    payload = {}
    if config.is_mx:
        from .matrix.inputs import Fp32InputSampler

        for pair in config.distribution_pairs:
            activation = Fp32InputSampler(pair["activation_distribution"])
            weight = Fp32InputSampler(pair["weight_distribution"])
            key = f"{pair['activation_distribution']}|{pair['weight_distribution']}"
            payload[key] = {
                "activation": activation.metadata(),
                "weight": weight.metadata(),
            }
    else:
        for fmt_name in config.formats:
            fmt = get_format(fmt_name)
            for dist in config.distributions:
                sampler = Sampler(fmt, dist, config.cancellation_sign_mode)
                payload[f"{fmt_name}|{dist}"] = sampler.metadata()
    text = dumps_strict(payload)
    if args.out:
        write_json(args.out, payload)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def _cmd_version(args) -> int:
    print(
        json.dumps(
            {"package": PACKAGE_VERSION, "schema": SCHEMA_VERSION, "stream": STREAM_VERSION},
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "analyze": _cmd_analyze,
        "enhance": _cmd_enhance,
        "bitmatch": _cmd_bitmatch,
        "mx-analyze": _cmd_mx_analyze,
        "mx-align-analyze": _cmd_mx_align_analyze,
        "mx-align-trace": _cmd_mx_align_trace,
        "width-run": _cmd_width_run,
        "width-analyze": _cmd_width_analyze,
        "width-replay": _cmd_width_replay,
        "plot": _cmd_plot,
        "distributions": _cmd_distributions,
        "version": _cmd_version,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:  # pragma: no cover - CLI error surface
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
