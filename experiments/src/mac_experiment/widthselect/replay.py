"""Small-scale diagnostic replay of one output dot16 sequence.

Given a run, a matrix pair (``sample_id``), an output coordinate, a K snapshot
and the run's candidate widths, re-derive the per-dot16 diagnostics from the
codes stored in the run (no re-sampling, no engine import) and export:

* experiment 1 (``W_align``): every product ``(P_i, C_i)``, its alignment shift,
  the aligned integer, dropped/zeroed flags, the exact integer sum and ``q_b``,
  the cumulative exact ``T`` and the saved-snapshot comparison;
* experiment 2 (``W_acc``): the exact ``s_b``, the state before the update, the
  two aligned operands, the dropped-bit flags, the exact integer sum, the
  normalization shift, the exponent and the state after the update;
* joint verification: both of the above for the ``q_b(W_align)`` input of each
  configured combination.

The final reconstructed state/value must match the saved snapshot exactly;
otherwise the replay aborts.
"""

from __future__ import annotations

from pathlib import Path

from ..arithmetic.dyadic import Dyadic
from ..results.reader import load_manifest, run_path
from ..results.schema import dumps_strict
from ..version import (
    WIDTHSELECT_ACC_MODE,
    WIDTHSELECT_ALIGN_MODE,
    WIDTHSELECT_JOINT_MODE,
    WIDTHSELECT_SCHEMA_VERSION,
)
from .acc_format import decode_state, hw_add, output_bits, zero_state
from .config import MAC_PARALLELISM
from .io import WIDTH_INPUTS_NAME, iter_width_inputs, iter_width_samples
from .reference import align_block_trace, block_products, block_scale_exponent, exact_sums
from .tensors import TargetTensor


def _load_input(source, sample_id: str, format: str) -> tuple[TargetTensor, TargetTensor]:
    for record in iter_width_inputs(source):
        if record.get("sample_id") == sample_id:
            payload = (record.get("formats") or {}).get(format)
            if not payload:
                raise ValueError(f"input record {sample_id} has no {format!r} tensor payload")
            return TargetTensor.from_dict(payload["x"]), TargetTensor.from_dict(payload["w"])
    raise ValueError(f"no input record for sample_id {sample_id!r} in {source / WIDTH_INPUTS_NAME}")


def _load_candidate(source, sample_id: str, format: str, K: int, mode: str, width_fields: dict):
    record_type = {
        WIDTHSELECT_ALIGN_MODE: "widthselect_align_snapshot",
        WIDTHSELECT_ACC_MODE: "widthselect_acc_snapshot",
        WIDTHSELECT_JOINT_MODE: "widthselect_joint_snapshot",
    }[mode]
    for record in iter_width_samples(source):
        if record.get("record_type") != record_type:
            continue
        if record.get("sample_id") != sample_id or record.get("format") != format:
            continue
        if int(record.get("K")) != int(K):
            continue
        if any(int(record.get(key)) != int(value) for key, value in width_fields.items()):
            continue
        return record
    raise ValueError(
        f"no saved {mode} snapshot for {sample_id} K={K} format={format} {width_fields}"
    )


def replay_width_run(
    run_dir,
    sample_id: str,
    row: int,
    col: int,
    K: int,
    w_align: int | None = None,
    w_acc: int | None = None,
    format: str | None = None,
) -> dict:
    source = run_path(run_dir)
    manifest = load_manifest(source)
    if manifest.get("schema_version") != WIDTHSELECT_SCHEMA_VERSION:
        raise ValueError(
            f"width-replay expects schema {WIDTHSELECT_SCHEMA_VERSION!r}, got "
            f"{manifest.get('schema_version')!r}; historical runs are never read"
        )
    mode = manifest.get("run_kind")
    if mode not in (WIDTHSELECT_ALIGN_MODE, WIDTHSELECT_ACC_MODE, WIDTHSELECT_JOINT_MODE):
        raise ValueError(f"unsupported run kind {mode!r}")
    config = manifest["config"]
    formats = list(config["formats"])
    if format is None:
        if len(formats) != 1:
            raise ValueError(f"run has formats {formats}; pass --format explicitly")
        format = formats[0]
    if format not in formats:
        raise ValueError(f"format {format!r} is not part of this run ({formats})")
    M, N = int(config["M"]), int(config["N"])
    if not 0 <= row < M or not 0 <= col < N:
        raise ValueError(f"output coordinate ({row}, {col}) outside {M}x{N}")
    if int(K) not in [int(k) for k in config["K_list"]]:
        raise ValueError(f"K={K} is not a declared snapshot {config['K_list']}")

    x, w = _load_input(source, sample_id, format)
    index = row * N + col
    blocks = []

    if mode == WIDTHSELECT_ALIGN_MODE:
        if w_align is None or w_acc is not None:
            raise ValueError("widthselect_align replay requires --w-align and no --w-acc")
        if int(w_align) not in [int(value) for value in config["W_align_list"]]:
            raise ValueError(f"W_align={w_align} is not part of this run")
        saved = _load_candidate(source, sample_id, format, K, mode, {"w_align": w_align})
        exact_t = Dyadic.zero()
        for block in range(int(K) // MAC_PARALLELISM):
            h = block_scale_exponent(x, w, row, col, block)
            products = block_products(x, w, row, col, block, format)
            trace = align_block_trace(products, h, int(w_align), format)
            exact_t = exact_t + Dyadic.from_dict(trace["q_b"])
            blocks.append(
                {
                    "block": block,
                    "k_begin": block * MAC_PARALLELISM,
                    "k_end": (block + 1) * MAC_PARALLELISM - 1,
                    **trace,
                    "exact_T_after_block": exact_t.to_dict(),
                }
            )
        saved_t = Dyadic.from_dict(saved["T"][index])
        if exact_t != saved_t:
            raise ValueError(
                f"replay mismatch for {sample_id} ({row}, {col}) K={K} W_align={w_align}: "
                "reconstructed T differs from the saved snapshot"
            )
        return {
            "replay_schema": "mac-width-replay-v1",
            "run_id": manifest.get("run_id"),
            "run_kind": mode,
            "sample_id": sample_id,
            "format": format,
            "row": row,
            "col": col,
            "K": int(K),
            "B": int(K) // MAC_PARALLELISM,
            "w_align": int(w_align),
            "container_bits": int(w_align) + 7,
            "blocks": blocks,
            "final_exact_T": exact_t.to_dict(),
            "saved_snapshot_matches": True,
        }

    if mode == WIDTHSELECT_ACC_MODE:
        if w_acc is None or w_align is not None:
            raise ValueError("widthselect_acc replay requires --w-acc and no --w-align")
        width_fields = {"w_acc": w_acc}
    else:
        if w_align is None or w_acc is None:
            raise ValueError("widthselect_joint replay requires both --w-align and --w-acc")
        if [int(w_align), int(w_acc)] not in [
            [int(pair["w_align"]), int(pair["w_acc"])] for pair in config["joint_pairs"]
        ]:
            raise ValueError(f"joint combination ({w_align}, {w_acc}) is not part of this run")
        width_fields = {"w_align": w_align, "w_acc": w_acc}

    if mode == WIDTHSELECT_ACC_MODE and int(w_acc) not in [int(value) for value in config["W_acc_list"]]:
        raise ValueError(f"W_acc={w_acc} is not part of this run")

    saved = _load_candidate(source, sample_id, format, K, mode, width_fields)
    state = zero_state(int(w_acc))
    for block in range(int(K) // MAC_PARALLELISM):
        h = block_scale_exponent(x, w, row, col, block)
        products = block_products(x, w, row, col, block, format)
        if mode == WIDTHSELECT_ACC_MODE:
            s_b, _ = exact_sums(products, h, format)
            incoming = s_b
            alignment = None
        else:
            alignment = align_block_trace(products, h, int(w_align), format)
            incoming = Dyadic.from_dict(alignment["q_b"])
        trace: list = []
        state, flags = hw_add(state, incoming, int(w_acc), trace=trace)
        payload = {
            "block": block,
            "k_begin": block * MAC_PARALLELISM,
            "k_end": (block + 1) * MAC_PARALLELISM - 1,
            "incoming": incoming.to_dict(),
            "hw_add": trace[-1],
            "state_after": state.to_dict(),
            "state_value_after": None if state.is_overflow_marker else decode_state(state).to_dict(),
            "Y_bits_after": output_bits(state),
            "events": flags,
        }
        if alignment is not None:
            payload["alignment"] = alignment
        blocks.append(payload)
    if state.to_dict() != saved["C_state"][index]:
        raise ValueError(
            f"replay mismatch for {sample_id} ({row}, {col}) K={K} {width_fields}: "
            "reconstructed state differs from the saved snapshot"
        )
    if output_bits(state) != int(saved["Y_bits"][index]):
        raise ValueError(
            f"replay mismatch for {sample_id} ({row}, {col}) K={K} {width_fields}: "
            "reconstructed output bits differ from the saved snapshot"
        )
    return {
        "replay_schema": "mac-width-replay-v1",
        "run_id": manifest.get("run_id"),
        "run_kind": mode,
        "sample_id": sample_id,
        "format": format,
        "row": row,
        "col": col,
        "K": int(K),
        "B": int(K) // MAC_PARALLELISM,
        "w_acc": int(w_acc),
        "w_align": None if mode == WIDTHSELECT_ACC_MODE else int(w_align),
        "state_bits": int(w_acc) + 9,
        "blocks": blocks,
        "final_state": state.to_dict(),
        "final_state_value": None if state.is_overflow_marker else decode_state(state).to_dict(),
        "saved_snapshot_matches": True,
    }


def write_replay(payload: dict, out: str | None) -> None:
    text = dumps_strict(payload) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote replay to {out}")
    else:
        print(text, end="")
