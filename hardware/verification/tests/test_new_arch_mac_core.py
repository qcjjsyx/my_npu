"""ACC arithmetic and full MAC cycle co-simulation with Icarus."""

from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
import subprocess

import pytest

from reference.new_arch.acc_fixed import update
from reference.new_arch.forward import Mode
from reference.new_arch.input_convert import SourceFormat as Fmt, convert_raw
from reference.new_arch.mac_core import MacCoreModel, MacInput, ZERO_FIELDS, pack_float
from reference.new_arch.oracle import exact_update


ROOT = Path(__file__).resolve().parents[3]
FORMATS = {
    Mode.FP4: Fmt.FP4, Mode.FP8: Fmt.E5M2, Mode.BF16: Fmt.BF16, Mode.FP16: Fmt.FP16,
    Mode.INT4: Fmt.INT4, Mode.INT8: Fmt.INT8, Mode.INT16: Fmt.INT16,
}
MODES = [(mode, False) for mode in Mode] + [
    (Mode.FP4, True), (Mode.FP8, True), (Mode.INT8, True),
]


@dataclass(frozen=True)
class Frame:
    txn: MacInput | None = None
    selected: int = 3
    ready: bool = False
    rst_n: bool = True
    x_unused: bool = False


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    directory = tmp_path_factory.mktemp("new_arch_mac_core")
    images = {}
    for name in ("core", "acc_arithmetic"):
        image = directory / f"{name}.vvp"
        result = subprocess.run(
            ["iverilog", "-g2001", "-s", f"tb_new_arch_mac_{name}", "-o", str(image),
             "-f", "hardware/filelists/new_arch_mac.f",
             "hardware/verification/tb/tb_new_arch_mac_forward.v",
             "hardware/verification/tb/tb_new_arch_mac_core.v"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        images[name] = image
    return images


def run(image, path):
    result = subprocess.run(["vvp", str(image), f"+VEC={path}"], cwd=ROOT,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.splitlines()


def bits(value, width):
    return value % (1 << width)


def compare_trace(lines, expected):
    for line, (kind, index, values) in zip(lines, expected, strict=True):
        event, number, *words = line.split()
        assert (event, number) == (kind, str(index)), line
        assert [word.lstrip("0") or "0" for word in words] == [f"{v:x}" for v in values], (
            f"{kind} at {index}", line, [f"{v:x}" for v in values]
        )


def write_frames(path, frames):
    def fields(side, mode, x_unused):
        lanes = [f"{word:022b}" for word in side]
        if x_unused:
            valid_width = {Mode.FP16: 18, Mode.BF16: 17, Mode.INT16: 16}[mode]
            for lane, word in enumerate(lanes):
                unused = 22 if mode != Mode.BF16 and lane % 2 else 22 - valid_width
                lanes[lane] = "x" * unused + word[unused:]
        return "".join(reversed(lanes))

    with path.open("w", encoding="ascii") as out:
        for frame in frames:
            txn = frame.txn
            prefix = f"{int(frame.rst_n)} {int(txn is not None)}"
            suffix = f"{frame.selected:x} {int(frame.ready)}"
            if txn is None:
                out.write(f"{prefix} ff {suffix} xxxx xxxx {'x' * 352} {'x' * 352}\n")
            else:
                sa = f"{txn.scale_a:04x}" if not frame.x_unused or txn.mx else "xxxx"
                sb = f"{txn.scale_b:04x}" if not frame.x_unused or txn.mx else "xxxx"
                out.write(f"{prefix} {txn.meta:02x} {suffix} {sa} {sb} "
                          f"{fields(txn.a, txn.mode, frame.x_unused)} "
                          f"{fields(txn.b, txn.mode, frame.x_unused)}\n")


def check_core(image, tmp_path, frames):
    frames = [Frame(rst_n=False), *frames]
    path = tmp_path / "core.txt"
    write_frames(path, frames)
    model = MacCoreModel()
    expected = []
    commits = []
    for cycle, frame in enumerate(frames):
        if cycle:
            expected.append(("B", cycle, model.outputs(frame.selected)))
        writeback = model.step(frame.txn, frame.selected, frame.ready, frame.rst_n)
        if writeback is not None:
            txn = writeback.txn
            expected.append(("W", cycle, (txn.meta, int(txn.group_count == 2), *writeback.states)))
            commits.append(writeback)
        valid = sum(1 << j for j, txn in enumerate(model.pending) if txn is not None)
        tags = tuple(txn.meta if txn is not None else 0 for txn in list(model.pending)[8:])
        expected.append(("P", cycle, (valid, *tags, model.mode * 2 + model.mx)))
        expected.append(("A", cycle, model.outputs(frame.selected)))

    lines = run(image, path)
    assert f"PASS: {len(frames)} cycles" in lines, lines[-10:]
    compare_trace([line for line in lines if line.startswith(("B ", "W ", "P ", "A "))], expected)
    return commits


def raw_input(txn, a, b, formats=None):
    fa, fb = formats or (FORMATS[txn.mode], FORMATS[txn.mode])
    return replace(txn, a=tuple(convert_raw(word, fa) for word in a),
                   b=tuple(convert_raw(word, fb) for word in b))


def constant_input(txn, a, b):
    return raw_input(txn, (a,) * 16, (b,) * 16)


def random_input(txn, rng, formats=None):
    def word():
        width = {Mode.INT4: 4, Mode.INT8: 8, Mode.INT16: 16}.get(txn.mode)
        if width is None:
            return rng.getrandbits(16)
        limit = (1 << (width - 1)) - 1
        return sum(bits(rng.randint(-limit, limit), width) << bit for bit in range(0, 16, width))

    return raw_input(txn, [word() for _ in range(16)], [word() for _ in range(16)], formats)


@pytest.mark.parametrize("mode,mx", MODES)
def test_core_interleaved_all_modes(images, tmp_path, mode, mx):
    rng = Random(0xACC40 + 2 * mode + int(mx))
    formats = [(a, b) for a in (Fmt.E4M3FN, Fmt.E5M2) for b in (Fmt.E4M3FN, Fmt.E5M2)]
    frames = []
    for step in range(12):
        for ctx in range(3):
            # Each pair of dot16 contributions reuses the scale of its MX block.
            block = step if mode == Mode.FP4 else step // 2
            sa = ((254 - 11 * block - ctx) << 8) | (21 * block + ctx)
            sb = ((17 * block + 3 * ctx) << 8) | (254 - 19 * block - ctx)
            txn = MacInput(mode, ctx, first=(step == 0), last=(step == 11),
                           mx=mx, scale_a=sa if mx else 0xFFFF, scale_b=sb if mx else 0xFFFF)
            source = formats[(step + ctx) % 4] if mode == Mode.FP8 else None
            frames.append(Frame(random_input(txn, rng, source), selected=ctx))
    frames += [Frame(selected=2)] * 16
    frames += [Frame(selected=3, ready=True), Frame(selected=2, ready=True),
               Frame(selected=0, ready=True), Frame(selected=1, ready=True)]
    commits = check_core(images["core"], tmp_path, frames)
    assert len(commits) == 36


def test_core_release_accept_and_commit_in_same_cycle(images, tmp_path):
    mode = Mode.INT8
    start = constant_input(MacInput(mode, 0), 0x8102, 0x7F03)
    second = replace(start, ctx=1)
    active = replace(start, ctx=2, last=False)
    finish = MacInput(mode, 2, first=False)
    frames = [Frame(start), Frame(second), *([Frame(selected=0, ready=True)] * 10)]
    frames += [
        Frame(active, selected=0, ready=True),  # release ctx0, accept ctx2, finish ctx1
        Frame(start, selected=1, ready=True),  # ctx0 can be used now
        *([Frame(selected=0)] * 45),
        Frame(finish, selected=3, ready=True),  # zero last after a long pause
        *([Frame(selected=2)] * 12),
        Frame(selected=2, ready=True), Frame(selected=0, ready=True),
    ]
    # Change mode after all results were accepted, reusing a previous G=2 slot.
    fp16 = constant_input(MacInput(Mode.FP16, 0), 0xECFF, 0x6FFF)
    frames += [Frame(fp16, x_unused=True), *([Frame(selected=0)] * 12),
               Frame(selected=0, ready=True)]
    commits = check_core(images["core"], tmp_path, frames)
    assert len(commits) == 6
    assert commits[2].txn.ctx == commits[4].txn.ctx == 2
    assert commits[2].states == commits[4].states


@pytest.mark.parametrize("mode", (Mode.FP16, Mode.BF16))
def test_core_reset_cancels_work_and_isolates_unwritten_state(images, tmp_path, mode):
    word = 0x6FFF if mode == Mode.FP16 else 0x7F7F
    first = constant_input(MacInput(mode, 0), word, word)
    aborted = replace(first, ctx=1, last=False)
    frames = [Frame(first, x_unused=True), Frame(), Frame(aborted, x_unused=True),
              *([Frame(selected=0)] * 10)]
    # The pending ctx1 write, ready for ctx0, and a new input coincide with reset.
    frames += [Frame(replace(first, ctx=2), selected=0, ready=True, rst_n=False, x_unused=True)]
    next_value = constant_input(MacInput(mode, 0, last=False), word ^ 0x8000, word)
    frames += [Frame(next_value, x_unused=True), Frame(), Frame(),
               Frame(replace(next_value, first=False, last=True), x_unused=True),
               *([Frame(selected=0)] * 13), Frame(selected=0, ready=True)]
    commits = check_core(images["core"], tmp_path, frames)
    assert [writeback.txn.ctx for writeback in commits] == [0, 0, 0]


@pytest.mark.parametrize("mode,mx", MODES)
def test_core_dot_length_16384(images, tmp_path, mode, mx):
    terms, a, b = {
        Mode.FP4: (32, 0xFF77, 0x7777),
        Mode.FP8: (16, 0xFB7B, 0x7B7B),
        Mode.BF16: (16, 0xFF7F, 0x7F7F),
        Mode.FP16: (8, 0xFBFF, 0x7BFF),
        Mode.INT4: (32, 0x9977, 0x7777),
        Mode.INT8: (16, 0x817F, 0x7F7F),
        Mode.INT16: (8, 0x8001, 0x7FFF),
    }[mode]
    base = MacInput(mode, 0, mx=mx, scale_a=0xFEFE, scale_b=0xFEFE)
    base = constant_input(base, a, b)
    count = 16384 // terms
    frames = []
    for j in range(count):
        frames.append(Frame(replace(base, first=(j == 0), last=(j == count - 1)), selected=0))
        frames.extend((Frame(), Frame()))
    frames.extend([Frame(selected=0)] * 11)
    frames.append(Frame(selected=0, ready=True))
    commits = check_core(images["core"], tmp_path, frames)
    assert len(commits) == count
    if not base.is_float:
        limit = {Mode.INT4: 7, Mode.INT8: 127, Mode.INT16: 32767}[mode]
        magnitude = 16384 * limit**2
        exact = (magnitude, bits(-magnitude, 51)) if base.group_count == 2 else (
            bits(-magnitude, 51), 0)
        assert commits[-1].states == exact


def test_acc_align_and_rne_against_fixed_model_and_exact_oracle(images, tmp_path):
    base, limit = 1 << 39, 1 << 40
    magnitudes = [base, base + 1, base + 2, base + 3, base + 7,
                  3 * (1 << 38), limit - 3, limit - 2, limit - 1]
    values = [sign * m for m in magnitudes for sign in (-1, 1)]
    cases = [(a, ha, b, hb) for a in values for b in values
             for ha, hb in [(5, 5 - d) for d in (0, 1, 2, 3, 4, 5, 40, 41, 42, 43, 44)]
             + [(511, -512), (-512, 511), (-30, -27)]]
    cases += [(0, 0, 0, 0), (0, 0, -base, -200), (base + 1, 12, 0, 0)]
    path = tmp_path / "acc.txt"
    expected = []
    ties, renorm_signs, grs, d_sigma = set(), set(), set(), set()
    saw_t_minus_2 = False
    with path.open("w", encoding="ascii") as out:
        for row, (m0, h0, m1, h1) in enumerate(cases):
            old = pack_float(m0, h0)
            out.write(f"1 0 {old:x} {bits(m1, 41):x} {bits(h1, 10):x} 0\n")
            debug = {}
            result = exact_update(m0, h0, m1, h1)
            assert update(m0, h0, m1, h1, debug=debug) == result
            state = pack_float(*result)
            if not m0 or not m1:
                expected.append(("U", row, (0, 0, 0, 0, 1, state, 0, state)))
                continue
            expected.append(("U", row, (bits(debug["A"], 45), bits(debug["B"], 45),
                                        bits(debug["H_a"], 10), int(debug["tau"]), 0, 0,
                                        bits(debug["T"], 45), state)))
            if "sigma" not in debug:
                continue
            g, r, s = (int(debug[key]) for key in ("G", "R", "S"))
            grs.add((g, r, s))
            d_sigma.add((debug["d"], debug["sigma"]))
            if g and not (r or s):
                ties.add((debug["M_pre"] < 0, debug["M_pre"] & 1))
            if debug["c"]:
                renorm_signs.add(result[0] < 0)
            saw_t_minus_2 |= debug["T"] == -2 and debug["sigma"] == 41

        int_cases = ((-1, 0), (0, -1), ((1 << 44) - 2, 1), (-(1 << 44) + 1, -1))
        for row, (old, contribution) in enumerate(int_cases, start=len(cases)):
            out.write(f"0 0 {bits(old, 51):x} 0 0 {bits(contribution, 45):x}\n")
            result = old + contribution
            expected.append(("U", row, (bits(old, 45), bits(contribution, 45), 0, 0, 0, 0,
                                        bits(result, 45), bits(result, 51))))
    assert grs == {(g, r, s) for g in (0, 1) for r in (0, 1) for s in (0, 1)}
    assert ties == {(negative, parity) for negative in (False, True) for parity in (0, 1)}
    assert renorm_signs == {False, True}
    assert (3, 1) in d_sigma and saw_t_minus_2
    lines = run(images["acc_arithmetic"], path)
    assert f"PASS: {len(expected)} cycles" in lines, lines[-8:]
    compare_trace([line for line in lines if line.startswith("U ")], expected)


def test_acc_float_t_minus_one_is_rejected(images, tmp_path):
    path = tmp_path / "bad_acc.txt"
    path.write_text(f"1 1 {pack_float(1 << 39, 0):x} {1 << 39:x} 0 0\n", encoding="ascii")
    lines = run(images["acc_arithmetic"], path)
    assert "REJECT float_t_minus_one" in lines, lines
    assert not any(line.startswith("PASS:") for line in lines), lines


_OPEN = Frame(MacInput(Mode.INT8, 0, last=False))
_CLOSE = Frame(MacInput(Mode.INT8, 0))
_CONTINUE = Frame(MacInput(Mode.INT8, 0, first=False))
_DRAIN = (Frame(),) * 11


@pytest.mark.parametrize("frames,reason", [
    ((Frame(MacInput(Mode.INT8, 3)),), "ctx"),
    ((Frame(MacInput(7, 0)),), "mode"),
    ((Frame(MacInput(Mode.FP16, 0, mx=True)),), "mx_mode"),
    ((Frame(MacInput(Mode.FP8, 0, mx=True, scale_b=0xFF00)),), "scale"),
    ((Frame(MacInput(Mode.INT16, 0, a=(0x8000,) + ZERO_FIELDS[1:])),), "int16"),
    ((_OPEN, Frame(), Frame(), _OPEN), "repeated_first"),
    ((_OPEN, Frame(MacInput(Mode.INT8, 1, first=False))), "missing_first"),
    ((_CLOSE, *_DRAIN, Frame(selected=0, ready=True),
      Frame(MacInput(Mode.FP16, 0, first=False))), "missing_first_batch_held"),
    ((_CLOSE, Frame(), Frame(), _CONTINUE), "appended_after_last"),
    ((_OPEN, _CONTINUE), "spacing"),
    ((_OPEN, Frame(), _CONTINUE), "spacing"),
    ((_CLOSE, *_DRAIN, _CLOSE), "occupied"),
    ((_CLOSE, *_DRAIN, replace(_CLOSE, selected=0, ready=True)), "reuse_on_release"),
    ((_OPEN, *_DRAIN, Frame(MacInput(Mode.FP16, 1))), "batch_mode"),
    ((_OPEN, *_DRAIN, Frame(MacInput(Mode.INT8, 1, mx=True))), "batch_mode"),
], ids=("ctx", "mode", "mx-mode", "scale", "int-min", "repeated-first", "missing-first",
        "empty-nonfirst", "appended-after-last", "d1", "d2", "occupied", "same-edge-reuse",
        "batch-mode", "batch-mx"))
def test_core_monitor_rejects_illegal_input(images, tmp_path, frames, reason):
    path = tmp_path / "bad_core.txt"
    write_frames(path, [Frame(rst_n=False), *frames])
    lines = run(images["core"], path)
    assert f"REJECT {reason}" in lines, lines[-8:]
    assert not any(line.startswith("PASS:") for line in lines), lines[-8:]
