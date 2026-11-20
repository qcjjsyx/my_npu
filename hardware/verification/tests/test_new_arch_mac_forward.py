"""Bit and cycle checks for the Verilog-2001 R_IN through R_MX pipeline."""

from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
import subprocess

import pytest

from reference.new_arch.acc_fixed import normalize_contribution
from reference.new_arch.forward import ALIGNMENT, Mode, decode_groups, forward_groups
from reference.new_arch.input_convert import SourceFormat as Fmt, convert_raw


ROOT = Path(__file__).resolve().parents[3]
BENCH = "hardware/verification/tb/tb_new_arch_mac_forward.v"
FORMAT = {
    Mode.FP4: Fmt.FP4, Mode.FP8: Fmt.E5M2, Mode.BF16: Fmt.BF16, Mode.FP16: Fmt.FP16,
    Mode.INT4: Fmt.INT4, Mode.INT8: Fmt.INT8, Mode.INT16: Fmt.INT16,
}
FIELD_WIDTH = {
    Mode.FP4: 20, Mode.FP8: 22, Mode.BF16: 17, Mode.FP16: 18,
    Mode.INT4: 20, Mode.INT8: 16, Mode.INT16: 16,
}
PAIRED = (Mode.FP16, Mode.INT16)
PACKED = (Mode.FP4, Mode.INT4)
ZERO_SLOTS = (0,) * 16


@dataclass(frozen=True)
class Input:
    mode: Mode | int
    a: tuple[int, ...] = ZERO_SLOTS
    b: tuple[int, ...] = ZERO_SLOTS
    mx: bool = False
    scale_a: int = 0
    scale_b: int = 0
    x_unused: bool = False


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    directory = tmp_path_factory.mktemp("new_arch_mac_forward")
    compiled = {}
    for name in ("forward", "mx_lane"):
        image = directory / f"{name}.vvp"
        result = subprocess.run(
            ["iverilog", "-g2001", "-s", f"tb_new_arch_mac_{name}", "-o", str(image),
             "-f", "hardware/filelists/new_arch_mac.f", BENCH],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        compiled[name] = image
    return compiled


def _run(image, path):
    result = subprocess.run(["vvp", str(image), f"+VEC={path}"], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.splitlines()


def _bits(value, width):
    return value % (1 << width)


def _pack(values, width):
    return sum(_bits(value, width) << (idx * width) for idx, value in enumerate(values))


def _field_bits(fields, inp):
    slots = [f"{word:022b}" for word in fields]
    if inp.x_unused:
        for lane, slot in enumerate(slots):
            used = 0 if inp.mode in PAIRED and lane % 2 else FIELD_WIDTH[inp.mode]
            slots[lane] = "x" * (22 - used) + slot[22 - used:]
    return "".join(reversed(slots))


def _raw_input(mode, a, b, *, formats=None, **kwargs):
    fmt_a, fmt_b = formats or (FORMAT[mode], FORMAT[mode])
    return Input(mode, tuple(convert_raw(word, fmt_a) for word in a),
                 tuple(convert_raw(word, fmt_b) for word in b), **kwargs)


def _set_unused_bits(inp):
    slot_mask = (1 << 22) - 1
    unused_mask = slot_mask ^ ((1 << FIELD_WIDTH[inp.mode]) - 1)

    def disturb(side):
        return tuple(slot_mask if inp.mode in PAIRED and lane % 2 else word | unused_mask
                     for lane, word in enumerate(side))

    return replace(inp, a=disturb(inp.a), b=disturb(inp.b))


def _mul_record(inp):
    """Projection of the element products onto the five R_MUL fields."""
    mode = inp.mode
    a, b = decode_groups(mode, inp.a), decode_groups(mode, inp.b)
    per_lane = []
    for j in range(16):
        w1 = s0 = s1 = t = 0
        if mode in PAIRED:
            (ac, ae), (bc, be) = a[0][j // 2], b[0][j // 2]
            ah, al = divmod(ac, 256)
            bh, bl = divmod(bc, 256)
            piece = ah if j % 2 else al
            w0, w1 = piece * bl, piece * bh
            t = ae + be if mode == Mode.FP16 and j % 2 == 0 else 0
        elif mode in PACKED:
            w0, s0, w1, s1 = (a[group][2*j+part][0] * b[group][2*j+part][0]
                               for group in range(2) for part in range(2))
        else:
            (ac, ae), (bc, be) = a[0][j], b[0][j]
            w0 = ac * bc
            t = ae + be
            if len(a) == 2:
                (ac, ae), (bc, be) = a[1][j], b[1][j]
                w1 = ac * bc
                t += 16 * (ae + be)
        per_lane.append((w0, w1, s0, s1, t))
    return tuple(_pack(values, width)
                 for values, width in zip(zip(*per_lane), (18, 18, 10, 10, 9), strict=True))


def _prod_record(mode, g0, g1):
    p0, q0 = zip(*g0.products)
    p1 = p3 = p2 = q1 = ()
    if g1.products:
        p2, q1 = zip(*g1.products)
    if mode in PACKED:
        p0, p1 = p0[::2], p0[1::2]
        p2, p3 = p2[::2], p2[1::2]
    return (_pack(p0, 32), _pack(p1, 9), _pack(p2, 16), _pack(p3, 9),
            _pack(q0, 7), _pack(q1, 7))


def _tree_leaves(mode, a, b):
    """Expected physical routing. The reference final sum uses a/b directly."""
    if mode == Mode.FP16:
        high, low = zip(*(divmod(term, 2**19) for term in a))
        a, b = low, high
    elif mode in PACKED:
        l = 9 if mode == Mode.FP4 else 8
        packed = tuple(hi * (2 ** (l + 5)) + lo + 2 ** (l - 1)
                       for lo, hi in zip(a, b, strict=True))
        a, b = packed[::2], packed[1::2]
    return a + (0,) * (16 - len(a)), b + (0,) * (16 - len(b))


def _stage_records(inp, meta):
    mode = inp.mode
    g0, g1 = forward_groups(mode, inp.mx, inp.a, inp.b, inp.scale_a, inp.scale_b)
    a, b = _tree_leaves(mode, g0.terms, g1.terms)
    common = (meta, _bits(g0.scale, 9), _bits(g1.scale, 9))
    e = (_bits(g0.align_e, 10), _bits(g1.align_e, 10))
    is_float = mode < Mode.INT4 or inp.mx
    return (
        common,
        (*common, *_mul_record(inp)),
        (*common, *_prod_record(mode, g0, g1)),
        (*common, g0.max_q, g1.max_q, *(int(mode in ALIGNMENT and any(p for p, _ in g.products))
                                        for g in (g0, g1))),
        (*common, _pack(a, 32), _pack(b, 32), *e),
        (*common, *(_pack((sum(leaves[j:j+4]) for j in range(0, 16, 4)), 34)
                    for leaves in (a, b)), *e),
        (*common, _bits(sum(a), 36), _bits(sum(b), 36), *e),
        (meta, *(_bits(v, w) for g in (g0, g1)
                  for v, w in zip(g.sum_record(), (41, 10, 9), strict=True))),
        (meta, *(_bits(v, w) for g in (g0, g1)
                  for v, w in zip(g.mx_record(is_float), (41, 10, 45), strict=True))),
    )


def _write_frame(out, cycle, rst_n, inp):
    if inp is None:
        out.write(f"{rst_n} 0 ff xxxx xxxx {'x' * 352} {'x' * 352}\n")
        return 0
    meta = (int(inp.mode) << 5) | (int(inp.mx) << 4) | ((cycle % 3) << 2) | (cycle % 4)
    scale_a, scale_b = f"{inp.scale_a:04x}", f"{inp.scale_b:04x}"
    if inp.x_unused and not inp.mx:
        scale_a = scale_b = "xxxx"
    out.write(f"{rst_n} 1 {meta:02x} {scale_a} {scale_b} "
              f"{_field_bits(inp.a, inp)} {_field_bits(inp.b, inp)}\n")
    return meta


def _check_chain(image, tmp_path, inputs, *, reset_at=()):
    # reset_at indexes entries of inputs, so a reset can deliberately coincide with valid.
    frames = [(0, None), *((int(j not in reset_at), inp) for j, inp in enumerate(inputs)),
              *([(1, None)] * 9)]
    pipeline = [None] * 9
    expected = []
    path = tmp_path / "forward.txt"
    with path.open("w", encoding="ascii") as out:
        for cycle, (rst_n, inp) in enumerate(frames):
            meta = _write_frame(out, cycle, rst_n, inp)
            if rst_n:
                pipeline = [_stage_records(inp, meta) if inp else None, *pipeline[:-1]]
            else:
                pipeline = [None] * 9
            valid = sum(1 << j for j, record in enumerate(pipeline) if record is not None)
            expected.append(("V", cycle, (valid, int(valid == 0), valid >> 8)))
            expected.extend((f"E{j}", cycle, record[j]) for j, record in enumerate(pipeline)
                            if record is not None)

    lines = _run(image, path)
    assert f"PASS: {len(frames)} cycles" in lines, lines[-6:]
    trace = [line for line in lines if line.startswith(("V ", "E"))]
    for line, (stage, cycle, values) in zip(trace, expected, strict=True):
        tokens = line.split()
        assert tokens[:2] == [stage, str(cycle)], line
        assert [v.lstrip("0") or "0" for v in tokens[2:]] == [f"{v:x}" for v in values], (
            f"{stage} at cycle {cycle}: {line}", [f"{v:x}" for v in values])


def _random_word(mode, rng):
    if mode == Mode.INT4:
        return _pack((rng.randint(-7, 7) for _ in range(4)), 4)
    if mode == Mode.INT8:
        return _pack((rng.randint(-127, 127) for _ in range(2)), 8)
    if mode == Mode.INT16:
        return _bits(rng.randint(-32767, 32767), 16)
    return rng.getrandbits(16)


@pytest.mark.parametrize("mode,mx", [(mode, False) for mode in Mode]
                         + [(Mode.FP4, True), (Mode.FP8, True), (Mode.INT8, True)])
def test_forward_each_mode(images, tmp_path, mode, mx):
    rng = Random(0x92803 + 2 * mode + mx)
    fp8_formats = [(a, b) for a in (Fmt.E4M3FN, Fmt.E5M2) for b in (Fmt.E4M3FN, Fmt.E5M2)]
    inputs = []
    for j in range(64):
        a, b = ([_random_word(mode, rng) for _ in range(16)] for _ in range(2))
        scales = [_pack((rng.randrange(255 if mx else 256) for _ in range(2)), 8)
                  for _ in range(2)]
        inp = _raw_input(mode, a, b, formats=fp8_formats[j % 4] if mode == Mode.FP8 else None,
                         mx=mx, scale_a=scales[0], scale_b=scales[1], x_unused=(j == 1))
        inputs.append(_set_unused_bits(inp) if j == 0 else inp)
    _check_chain(images["forward"], tmp_path, inputs)


def test_paired_product_limb_and_ignored_slots(images, tmp_path):
    raw_a = (0x10FF, 0xFDFF, 0x1001, 0xA103, 0x7BFF, 0x8001, 0x0001, 0x14FF)
    raw_b = (0x90FF, 0x1055, 0x90FF, 0x3203, 0x7BFE, 0xBC00, 0xBC00, 0x9301)
    fp_max = _raw_input(Mode.FP16, [0x6FFF] * 16, [0xEFFF] * 16)
    int_max = _raw_input(Mode.INT16, [0x7FFF] * 16, [0x8001] * 16)
    assert forward_groups(Mode.FP16, False, fp_max.a, fp_max.b)[0].n == -549_219_074_048
    fp_mixed = _raw_input(Mode.FP16, _with_odd_slots(raw_a), _with_odd_slots(raw_b))
    int_mixed = _raw_input(Mode.INT16, _with_odd_slots(raw_a), _with_odd_slots(raw_b))
    inputs = [fp_max, int_max, fp_mixed, int_mixed,
              _set_unused_bits(fp_mixed), _set_unused_bits(int_mixed)]
    _check_chain(images["forward"], tmp_path, inputs)


def _with_odd_slots(words):
    return [slot for word in words for slot in (word, 0x8000)]


def test_alignment_boundaries(images, tmp_path):
    # FP8 g0 reaches Q*=14, g1 reaches Q*=4 and excludes a zero product at Q=7.
    a = [0x287B, 0x7B81, 0x0001] + [0x0081] * 13
    b = [0x287B, 0x0001, 0x0001] + [0x0001] * 13
    fp8 = _raw_input(Mode.FP8, a, b)
    ga, gb = forward_groups(Mode.FP8, False, fp8.a, fp8.b)
    assert (ga.max_q, gb.max_q, ga.terms[1:3], gb.products[1]) == (14, 4, (-1, 0), (0, 7))

    # BF16 put 28/32/36/40-bit shifts and a negative half-LSB on both A widths.
    distances = (0, 12, 28, 32, 36, 40, 32, 40) * 2
    signs = (0, 1, 1, 1, 1, 1, 0, 0) * 2
    bf16 = _raw_input(Mode.BF16, [(sign << 15) | ((76 - d) << 7) | 1
                                  for sign, d in zip(signs, distances, strict=True)],
                      [(4 << 7) | 1] * 16)
    group = forward_groups(Mode.BF16, False, bf16.a, bf16.b)[0]
    assert group.max_q == 20 and group.terms[1] == group.terms[9] == -8321
    assert group.terms[3:8] == (-1, -1, -1, 0, 0)
    bf_max_t = _raw_input(Mode.BF16, [0x7F7F] * 16, [0xFF7F] * 16)

    # FP16 exercises the last reachable distance below 38 and the first above it.
    fp16 = _raw_input(Mode.FP16, _with_odd_slots([0x7BFF, 0xBFFF, 0xAFFF, 0, 0, 0, 0, 0]),
                      _with_odd_slots([0x7BFF, 0x2FFF, 0x2FFF, 0, 0, 0, 0, 0]))
    group = forward_groups(Mode.FP16, False, fp16.a, fp16.b)[0]
    assert group.max_q == 14
    assert tuple(4 * (group.max_q - q) for _, q in group.products[1:3]) == (36, 40)
    _check_chain(images["forward"], tmp_path, [fp8, bf16, bf_max_t, fp16])


@pytest.mark.parametrize("mode", PACKED)
def test_packing_rails_and_cancellation(images, tmp_path, mode):
    pos, neg = 7, 15 if mode == Mode.FP4 else 9
    a = [_pack([pos] * 4, 4)] * 16
    patterns = [(neg, neg, pos, pos), (pos, pos, neg, neg), (pos, neg, neg, pos)]
    inputs = [_raw_input(mode, a, [_pack(pattern, 4)] * 16) for pattern in patterns]
    if mode == Mode.FP4:
        inputs.append(_raw_input(mode, a, [_pack(patterns[-1], 4)] * 16,
                                 mx=True, scale_a=0xFE00, scale_b=0xFE00))
    _check_chain(images["forward"], tmp_path, inputs)


@pytest.mark.parametrize("mode", (Mode.FP8, Mode.INT8))
def test_mx_scales_and_dot16_halves(images, tmp_path, mode):
    raw = (0x3C01, 0x817B) if mode == Mode.FP8 else (0x017F, 0x817F)
    pair = [_raw_input(mode, [word] * 16, [raw[0]] * 16,
                       mx=True, scale_a=0xFE00, scale_b=0xFE00) for word in raw]
    for inp in pair:
        group_a, group_b = forward_groups(mode, True, inp.a, inp.b, inp.scale_a, inp.scale_b)
        assert (group_a.scale, group_b.scale) == (-254, 254)
        assert group_a.n != 0 and group_b.n != 0
    plain = _raw_input(mode, [raw[0]] * 16, [raw[0]] * 16, scale_a=0xFFFF, scale_b=0xFFFF)
    _check_chain(images["forward"], tmp_path, [pair[0], None, None, pair[1], plain])


@pytest.mark.parametrize("first_mode", (Mode.FP16, Mode.BF16))
def test_pipeline_bubbles_reset_and_x_isolation(images, tmp_path, first_mode):
    fp16 = _raw_input(Mode.FP16, [0xECFF] * 16, [0x6FFF] * 16, x_unused=True)
    bf16 = _raw_input(Mode.BF16, [0xBF81] * 16, [0x40FF] * 16, x_unused=True)
    first, second = (fp16, bf16) if first_mode == Mode.FP16 else (bf16, fp16)
    inputs = [first, None, second, *([None] * 9), first, second, first, second, None, first]
    _check_chain(images["forward"], tmp_path, inputs, reset_at=(14,))


def test_all_zero_and_legal_nonzero_q0(images, tmp_path):
    inputs = [Input(mode) for mode in Mode]
    inputs.extend((Input(Mode.FP8, mx=True, scale_a=0xFE00, scale_b=0xFE00),
                   _raw_input(Mode.FP8, [0x8101] * 16, [0x0101] * 16),
                   _raw_input(Mode.FP16, [0x8001] * 16, [0x0001] * 16),
                   _raw_input(Mode.BF16, [0x8001] * 16, [0x0001] * 16)))
    _check_chain(images["forward"], tmp_path, inputs)


def test_mx_lane_powers_neighbors_and_signed_exponents(images, tmp_path):
    magnitudes = {0, 2**40 - 1}
    for k in range(40):
        magnitudes.update((2**k - 1, 2**k, 2**k + 1))
    cases = [(1, sign * n, e, h) for n in sorted(magnitudes) for sign in (-1, 1)
             for e, h in ((-258, -254), (-58, 0), (218, 254))]
    cases.extend(((0, -8 * 32767**2, 0, 0), (0, 8 * 32767**2, 0, 0)))
    path = tmp_path / "mx_lane.txt"
    with path.open("w", encoding="ascii") as out:
        for use_float, n, e, h in cases:
            m, normalized_h = normalize_contribution(n, e + h) if use_float else (0, 0)
            record = (use_float, n, e, h, m, normalized_h, n if not use_float else 0)
            out.write(" ".join(f"{_bits(v, w):x}" for v, w in zip(
                record, (1, 41, 10, 9, 41, 10, 45), strict=True)) + "\n")
    lines = _run(images["mx_lane"], path)
    assert f"PASS: {len(cases)} cycles" in lines, lines[-6:]


@pytest.mark.parametrize("inp,reason", [
    (Input(7), "mode"), (Input(Mode.BF16, mx=True), "mx_mode"),
    (Input(Mode.FP8, mx=True, scale_a=0x00FF), "scale"),
    (Input(Mode.FP8, mx=True, scale_b=0x00FF), "scale"),
    (Input(Mode.FP8, mx=True, scale_a=0xFF00), "scale"),
    (Input(Mode.FP8, mx=True, scale_b=0xFF00), "scale"),
    (Input(Mode.INT4, a=(0x18,) + ZERO_SLOTS[1:]), "int4"),
    (Input(Mode.INT8, b=(0x8000,) + ZERO_SLOTS[1:]), "int8"),
    (Input(Mode.INT16, a=(0x8000,) + ZERO_SLOTS[1:]), "int16"),
], ids=("mode", "mx-mode", "scale-a0", "scale-b0", "scale-a1", "scale-b1",
        "int4-min", "int8-min", "int16-min"))
def test_forward_monitor_rejects_invalid_inputs(images, tmp_path, inp, reason):
    path = tmp_path / "invalid.txt"
    with path.open("w", encoding="ascii") as out:
        _write_frame(out, 0, 0, None)
        _write_frame(out, 1, 1, inp)
    lines = _run(images["forward"], path)
    assert f"REJECT {reason}" in lines, lines
    assert not any(line.startswith("PASS:") for line in lines), lines
