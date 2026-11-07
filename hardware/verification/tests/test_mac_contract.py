from fractions import Fraction
from dataclasses import replace
import random

import pytest

from reference.mac_contract import (
    AccKind,
    FloatAcc36State,
    UnfrozenContract,
    multiply,
    mx,
    trace_float_acc36_update,
    update_float_acc36,
    Aligned,
    ComputeMode,
    ContractViolation,
    DOT_N,
    Entry,
    MPhase,
    ReferenceAccumulator,
    SourceFormat,
    Transaction,
    align,
    enter,
    mode_for,
    partial_sum,
    round_div_pow2,
    signed,
)


def transaction(fmt_a, fmt_b=None, *, a=0, b=0, valid=1, first=1,
                mx=False, scale_a=(127, 127), scale_b=(127, 127), tag=(1, 2)):
    return Transaction(
        raw_a=(a,) + (0,) * (DOT_N - 1),
        raw_b=(b,) + (0,) * (DOT_N - 1),
        fmt_a=fmt_a,
        fmt_b=fmt_a if fmt_b is None else fmt_b,
        group_valid=valid,
        acc_first=first,
        tag=tag,
        mx_enable=mx,
        scale_a=scale_a,
        scale_b=scale_b,
    )


@pytest.mark.parametrize(
    "fmt,raw,exp,sig",
    [
        (SourceFormat.FP16, 0x8000, 0, 0),       # -0 -> +0
        (SourceFormat.FP16, 0x7E01, 0, 0),       # NaN -> +0
        (SourceFormat.FP16, 0x7C00, 30, 2047),  # +Inf -> max finite FP16
        (SourceFormat.FP16, 0xFC00, 30, -2047), # -Inf -> min finite FP16
        (SourceFormat.FP16, 0x0001, 1, 1),      # preserved subnormal
        (SourceFormat.BF16, 0x0001, 1, 8),      # three appended fraction zeros
        (SourceFormat.BF16, 0x7F80, 254, 2040),
        (SourceFormat.E4M3, 0x7F, 0, 0),        # E4M3FN NaN
        (SourceFormat.E4M3, 0x7E, 15, 14),      # 448 is finite
        (SourceFormat.E4M3, 0x01, 1, 1),
        (SourceFormat.E5M2, 0x7C, 30, 14),      # +Inf -> max finite E5M2
        (SourceFormat.E5M2, 0x01, 1, 1 << 1),
    ],
)
def test_source_classification_and_payload(fmt, raw, exp, sig):
    e = enter(transaction(fmt, a=raw), tag_w=4)
    p = e.payload_a[0]
    if fmt in (SourceFormat.FP16, SourceFormat.BF16):
        assert (p >> 16) & 255 == exp
        assert signed(p, 12) == sig
        assert p & ((3 << 24) | (15 << 12)) == 0
    else:
        assert (p >> 16) & 31 == exp
        assert signed(p, 5) == sig
        assert p & (63 << 10) == 0


def test_mixed_fp8_source_bias_and_lane_order():
    tx = transaction(SourceFormat.E4M3, SourceFormat.E5M2,
                     a=0x3C38, b=0x403C, valid=3, first=3)
    trace = ReferenceAccumulator(tag_w=4).step(tx)
    assert trace.entry.mode == ComputeMode.FP9
    assert trace.entry.bias_sum == 22
    assert trace.entry.payload_a[0] & 31 == 8
    assert signed(trace.entry.payload_a[0] >> 5, 5) == 12
    assert trace.phases[0].m.product0[0] == 8 * 8
    assert trace.snapshot.group_valid == 3
    assert trace.snapshot.tag == (1, 2)


@pytest.mark.parametrize(
    "fmt,one",
    [
        (SourceFormat.FP16, 0x3C00),
        (SourceFormat.BF16, 0x3F80),
        (SourceFormat.E4M3, 0x38),
        (SourceFormat.E5M2, 0x3C),
    ],
)
def test_one_times_one_has_exact_state_one(fmt, one):
    trace = ReferenceAccumulator(tag_w=4).step(transaction(fmt, a=one, b=one))
    assert trace.snapshot.kind == AccKind.TC_FLOAT
    assert trace.snapshot.data[0] == 1 << 34
    assert trace.snapshot.exp[0] == 127
    phase = trace.phases[0]
    if trace.entry.mode == ComputeMode.FP19:
        assert phase.m.product0[0] == 1 << 20
        assert phase.aligned.lane0[0] == 1 << 29
        assert phase.psum.contribution0 == 1 << 29
        assert phase.mx.exp[0] == 132
    else:
        assert phase.m.product0[0] == 64
        assert phase.aligned.lane0[0] == 1 << 21
        assert phase.psum.contribution0 == 1 << 21
        assert phase.mx.exp[0] == 132


def test_effective_exponent_addition_retains_carry():
    tx = transaction(SourceFormat.BF16, a=0x7F7F, b=0x7F7F)
    e, m, a, p, result = stages(tx)
    assert m.exp_sum0[0] == 508
    assert result.exp[0] == 386  # Mathematical E, cannot be silently packed.
    with pytest.raises(UnfrozenContract, match="exponent"):
        ReferenceAccumulator(tag_w=4).step(tx)
    assert stages(transaction(SourceFormat.E5M2, a=0x7B, b=0x7B))[1].exp_sum0[0] == 60


def test_fp19_negative_limb_and_low_tree_growth():
    entry = Entry(ComputeMode.FP19, AccKind.TC_FLOAT, (0,) * 16, (0,) * 16, 30, 1)
    m = MPhase(0, (0,) * 16, (0,) * 16, (0,) * 16, (0,) * 16,
               (False,) * 16, (False,) * 16)
    aligned = Aligned((-1,) * 16, (0,) * 16, (0, 0), (True, False), (False, False))
    psum = partial_sum(entry, m, aligned)
    assert psum.contribution0 == -16
    assert signed(psum.packed56, 56) == -16
    assert sum(x & ((1 << 23) - 1) for x in aligned.lane0) == (1 << 27) - 16


def test_fp9_group_maxima_are_independent():
    entry = Entry(ComputeMode.FP9, AccKind.TC_FLOAT, (0,) * 16, (0,) * 16, 14, 3)
    m = MPhase(0, (1,) * 16, (1,) * 16,
               (20,) + (10,) * 15, (4,) + (5,) * 15,
               (True,) * 16, (True,) * 16)
    aligned = align(entry, m)
    assert aligned.emax == (20, 5)
    assert aligned.lane0[0] == 1 << 15
    assert aligned.lane1[0] == 1 << 14


def test_mxfp8_scale_extremes_are_per_group():
    tx = transaction(SourceFormat.E4M3, a=0x3838, b=0x3838, valid=3, first=3,
                     mx=True, scale_a=(0, 254), scale_b=(0, 254))
    result = stages(tx)[-1]
    assert result.exp == (-122, 386)
    with pytest.raises(UnfrozenContract, match="exponent"):
        ReferenceAccumulator(tag_w=4).step(tx)


@pytest.mark.parametrize(
    "value,shift,expected,lost",
    [
        (3, 1, 2, True),
        (1, 1, 0, True),
        (-1, 1, 0, True),
        (-3, 1, -2, True),
        (-5, 1, -2, True),
        (-7, 1, -4, True),
        (-4, 1, -2, False),
    ],
)
def test_integer_rne_positive_and_negative_ties(value, shift, expected, lost):
    assert round_div_pow2(value, shift) == (expected, lost)


def test_align_rounds_each_term_before_tree_and_handles_large_distance():
    product = (1, -1, -3, -1) + (0,) * 12
    exponent = (20, 10, 10, 0) + (0,) * 12
    nz = (True, True, True, True) + (False,) * 12
    m = MPhase(0, product, (0,) * 16, exponent, (0,) * 16, nz, (False,) * 16)
    entry = Entry(ComputeMode.FP19, AccKind.TC_FLOAT, (0,) * 16, (0,) * 16, 30, 1)
    result = align(entry, m)
    assert result.emax == (20, 0)
    assert result.lane0[:4] == (512, 0, -2, 0)
    assert result.inexact == (True, False)
    assert sum(result.lane0) == 510


def test_int16_two_phases_preserve_mid_and_only_saturate_final():
    acc = ReferenceAccumulator(tag_w=4)
    acc.update_integer(0, (1 << 31) - 1, first=True)
    # 32767 * -1: phase0 uses B_L=255 and crosses INT32_MAX;
    # phase1 uses B_H=-1 and returns to the representable range.
    tx = transaction(SourceFormat.INT16, a=0x7FFF, b=0xFFFF, first=0)
    trace = acc.step(tx)
    assert len(trace.phases) == 2
    assert trace.phases[0].m.product0[0] == 255 * 255
    assert trace.phases[0].m.product1[0] == 127 * 255
    assert trace.phases[0].psum.contribution0 == 32767 * 255
    assert trace.phases[1].psum.contribution0 == 32767 * (-256)
    assert trace.snapshot.data[0] == ((1 << 31) - 1 - 32767)
    assert trace.snapshot.group_valid == 1


def test_dual_groups_independent_and_integer_saturates_each_dot():
    acc = ReferenceAccumulator(tag_w=4)
    first = transaction(SourceFormat.INT8, a=0x057F, b=0x037F, valid=3, first=3)
    one = acc.step(first)
    assert one.snapshot.data == (127 * 127, 5 * 3)
    assert one.snapshot.exp == (0, 0)
    second = transaction(SourceFormat.INT8, a=0x7F00, b=0x7F00, valid=2, first=0)
    two = acc.step(second)
    assert two.snapshot.data == (None, 15 + 127 * 127)
    assert two.snapshot.tag == (None, 2)
    acc.update_integer(0, (1 << 31) - 1, first=True)
    assert acc.update_integer(0, 1, first=False) == (1 << 31) - 1
    assert acc.update_integer(0, -1, first=False) == (1 << 31) - 2


def test_mxint8_scale_applied_after_integer_sum():
    tx = transaction(SourceFormat.INT8, a=0x7F, b=0x7F, mx=True,
                     scale_a=(0, 127), scale_b=(254, 127))
    trace = ReferenceAccumulator(tag_w=4).step(tx)
    assert trace.phases[0].psum.contribution0 == 16129
    assert trace.phases[0].mx.exp[0] == 132
    assert trace.snapshot.kind == AccKind.TC_FLOAT


@pytest.mark.parametrize(
    "tx",
    [
        transaction(SourceFormat.FP16, SourceFormat.BF16),
        transaction(SourceFormat.INT8, a=0x80),
        transaction(SourceFormat.INT16, a=0x8000),
        transaction(SourceFormat.E4M3, mx=True, scale_a=(255, 127)),
        transaction(SourceFormat.FP16, mx=True),
        transaction(SourceFormat.INT16, valid=3),
        transaction(SourceFormat.INT4, a=0x0101),
    ],
)
def test_unfrozen_or_illegal_inputs_are_rejected(tx):
    with pytest.raises(ContractViolation):
        enter(tx, tag_w=4)


def test_explicit_tag_width_and_first_are_required():
    with pytest.raises(ContractViolation):
        enter(transaction(SourceFormat.INT4), tag_w=0)
    with pytest.raises(ContractViolation):
        ReferenceAccumulator(tag_w=4).step(transaction(SourceFormat.INT4, first=0))


def test_source_format_encodings_are_explicit():
    assert [int(x) for x in SourceFormat] == list(range(7))
    assert [int(x) for x in ComputeMode] == list(range(5))
    with pytest.raises(ContractViolation):
        enter(transaction(7), tag_w=4)
    assert mode_for(SourceFormat.E4M3, SourceFormat.E5M2) == ComputeMode.FP9


def stages(tx):
    e = enter(tx, tag_w=4)
    m = multiply(e)
    a = align(e, m)
    p = partial_sum(e, m, a)
    return e, m, a, p, mx(e, tx, m, a, p)


def dyadic(coefficient, exponent):
    return Fraction(coefficient) * Fraction(2) ** exponent


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_fp9_q721_to_q234_group_emax_scale_and_exact_value(scaled, sign):
    tx = transaction(SourceFormat.E4M3, valid=3, first=3, mx=scaled,
                     scale_a=(127, 130), scale_b=(127, 125))
    entry = enter(tx, tag_w=4)
    m = MPhase(0, (sign * 64,) * 16, (-sign * 96,) * 16,
               (32,) * 16, (19,) * 16, (True,) * 16, (True,) * 16)
    a = align(entry, m)
    p = partial_sum(entry, m, a)
    result = mx(entry, tx, m, a, p)
    assert a.emax == (32, 19)
    assert result.exp == (150, 137 + int(scaled))
    for g, source in enumerate((p.contribution0, p.contribution1)):
        scale = (tx.scale_a[g] - 127 + tx.scale_b[g] - 127) if scaled else 0
        assert result.contribution[g] == source << 8
        assert result.contribution[g] % 256 == 0
        assert dyadic(result.contribution[g], result.exp[g] - 161) == dyadic(source, a.emax[g] - 14 - 21 + scale)
    assert 32 - 7 - 7 - 21 == -3
    assert -3 + 26 + 127 == 150


@pytest.mark.parametrize("a,b", [(127,127), (-127,127), (127,-127), (-127,-127), (0,127)])
def test_mxint8_signed15_signed19_q217_to_q234_and_decoded_scale(a, b):
    tx = transaction(SourceFormat.INT8, valid=3, first=3, mx=True,
                     scale_a=(129,124), scale_b=(126,128))
    tx = replace(tx, raw_a=(((a & 255) | ((-a & 255) << 8)),) * 16,
                 raw_b=(((b & 255) | ((b & 255) << 8)),) * 16)
    e, m, aligned, p, result = stages(tx)
    assert m.product0 == (a*b,) * 16
    assert -(1 << 14) <= a*b < (1 << 14)
    assert p.contribution0 == 16*a*b
    assert abs(p.contribution0) <= 258064 < (1 << 18)
    assert result.exp == (133, 130)  # decoded scales +1 and -2, not raw bytes
    assert result.kind == AccKind.TC_FLOAT
    for g, source in enumerate((p.contribution0, p.contribution1)):
        assert -(1 << 18) <= source < (1 << 18)
        assert result.contribution[g] == source << 17
        assert -(1 << 35) <= result.contribution[g] < (1 << 35)
        scale = tx.scale_a[g] + tx.scale_b[g] - 254
        assert dyadic(result.contribution[g], result.exp[g]-161) == dyadic(source, scale-12)


def fp19_aligned(products, distances):
    e = Entry(ComputeMode.FP19, AccKind.TC_FLOAT, (0,)*16, (0,)*16, 30, 1)
    m = MPhase(0, tuple(products), (0,)*16, tuple(40-d for d in distances),
               (0,)*16, tuple(p != 0 for p in products), (False,)*16)
    return e, m, align(e,m)


@pytest.mark.parametrize("sign", [-1,1])
def test_fp19_signed32_range_q329_and_signed36_q729_dot16(sign):
    max_product = 2047**2
    e,m,a = fp19_aligned([sign*max_product]*16, [0]*16)
    old_term = sign*max_product << 10
    new_term = sign*max_product << 9
    assert -(1<<32) <= old_term < (1<<32)
    assert -(1<<31) <= new_term < (1<<31)
    assert abs(old_term) > (1<<31)-1  # Cannot merely drop a high register bit.
    assert a.lane0 == (new_term,)*16
    p = partial_sum(e,m,a)
    assert p.contribution0 == 16*new_term
    assert abs(p.contribution0) < 1<<35
    assert dyadic(new_term,-29) == dyadic(old_term,-30)
    tx = transaction(SourceFormat.FP16)
    result = mx(e,tx,m,a,p)
    assert result.contribution[0] == p.contribution0
    assert result.exp[0] == 40-30+132
    assert dyadic(result.contribution[0],result.exp[0]-161) == dyadic(p.contribution0,40-30-29)
    state = update_float_acc36(None,result.contribution[0],result.exp[0],first=True)
    assert dyadic(state.sig,state.exp-161) == dyadic(p.contribution0,40-30-29)


@pytest.mark.parametrize("product,expected", [(1,0),(3,2),(5,2),(7,4),(-1,0),(-3,-2),(-5,-2),(-7,-4)])
def test_fp19_align_signed32_positive_negative_ties(product,expected):
    e,m,a = fp19_aligned([1,product]+[0]*14,[0,10]+[0]*14)
    assert a.lane0[1] == expected
    assert a.inexact == (True,False)
    assert partial_sum(e,m,a).contribution0 == 512+expected


def test_fp19_requantization_is_direct_rne_not_truncation_or_double_rounding():
    # P=11,d=12: old RNE(2.75)=3; RNE(3/2)=2, direct new RNE(1.375)=1.
    e,m,a = fp19_aligned([1,11,-11]+[0]*13,[0,12,12]+[0]*13)
    assert a.lane0[1:3] == (1,-1)
    assert round_div_pow2(round_div_pow2(11<<10,12)[0],1)[0] == 2
    # P=3,d=10 gives +1.5 => +2, whereas simple truncation gives +1.
    assert fp19_aligned([1,3]+[0]*14,[0,10]+[0]*14)[2].lane0[1] == 2


def test_fp19_rescale_quantization_error_bound_and_cancellation():
    rng = random.Random(234)
    for _ in range(1000):
        p = rng.randrange(-2047**2,2047**2+1)
        d = rng.randrange(507)
        new,_ = round_div_pow2(p<<9,d)
        exact = dyadic(p,9-d)
        assert abs(new-exact) <= Fraction(1,2)
        assert -(1<<31) <= new < (1<<31)
    for products in ([2047**2,-2047**2]*8, [0]*16):
        e,m,a = fp19_aligned(products,[0]*16)
        p = partial_sum(e,m,a)
        assert p.contribution0 == 0
        r = mx(e,transaction(SourceFormat.FP16),m,a,p)
        assert update_float_acc36(None,r.contribution[0],r.exp[0],first=True) == FloatAcc36State(0,0)


@pytest.mark.parametrize("sign", [-1,1])
def test_q234_full_36_plus_36_to_37_normalization(sign):
    operand = sign*((1<<35)-2)
    u = trace_float_acc36_update(FloatAcc36State(operand,127),operand,127,first=False)
    assert u.old_operand == u.new_operand == operand
    assert u.full_sum == 2*operand
    assert not -(1<<35) <= u.full_sum < (1<<35)
    assert -(1<<36) <= u.full_sum < (1<<36)
    assert u.state == FloatAcc36State(operand,128)
    assert dyadic(u.full_sum,u.common_exp-161) == dyadic(u.state.sig,u.state.exp-161)


def test_q234_cancellation_zero_first_and_exact_alignment():
    old = FloatAcc36State(1<<34,127)
    u = trace_float_acc36_update(old,-(1<<34)+1,127,first=False)
    assert u.full_sum == 1
    assert u.state == FloatAcc36State(1<<34,93)
    assert update_float_acc36(old,-(1<<34),127,first=False) == FloatAcc36State(0,0)
    assert update_float_acc36(old,0,127,first=True) == FloatAcc36State(0,0)
    assert update_float_acc36(old,-(1<<35),127,first=True) == FloatAcc36State(-(1<<34),128)
    u = trace_float_acc36_update(old,1<<34,126,first=False)
    assert u.new_operand == 1<<33
    assert u.state == FloatAcc36State(3<<33,127)


@pytest.mark.parametrize("sign", [-1,1])
def test_q234_align_tail_participates_in_final_rne(sign):
    old = FloatAcc36State(sign*((1<<34)+1),127)
    u = trace_float_acc36_update(old,sign*(1<<34),92,first=False)
    assert u.state == FloatAcc36State(sign*((1<<34)+2),127)  # odd +/- half tie
    assert u.old_tail + u.new_tail == Fraction(1,2)
    even = FloatAcc36State(sign*(1<<34),127)
    assert update_float_acc36(even,sign*(1<<34),92,first=False) == even
    far = trace_float_acc36_update(even,-sign,1,first=False)
    assert far.state == even  # retain nonzero tail even at a 126-bit distance
    assert far.old_tail + far.new_tail != 0


@pytest.mark.parametrize("sign", [-1,1])
@pytest.mark.parametrize("offset,rounded", [(1,0),(3,2)])
def test_q234_37_to_36_rne_even_odd_ties(sign,offset,rounded):
    u = trace_float_acc36_update(FloatAcc36State(sign*(1<<34),127),
                                sign*((1<<34)+offset),127,first=False)
    assert u.full_sum == sign*((1<<35)+offset)
    assert u.state == FloatAcc36State(sign*((1<<34)+rounded),128)


def test_q234_tail_retained_through_deep_cancellation_and_rounding_carry():
    old = FloatAcc36State(1<<34,127)
    u = trace_float_acc36_update(old,-(1<<35)+1,126,first=False)
    assert u.full_sum == 0 and u.new_tail == Fraction(1,2)
    assert u.state == FloatAcc36State(1<<34,92)
    # RNE reaches 2**35, then exact carry adjustment increments E.
    assert update_float_acc36(FloatAcc36State((1<<35)-1,127),1<<34,92,first=False) == FloatAcc36State(1<<34,128)


@pytest.mark.parametrize("exponent", [-200,0,255,386])
def test_q234_deferred_mx_input_exponent_boundary_never_wraps(exponent):
    with pytest.raises(UnfrozenContract,match="exponent"):
        update_float_acc36(None,1<<34,exponent,first=True)


@pytest.mark.parametrize("sign", [-1,1])
def test_q234_writeback_ftz_inf_zero_and_first(sign):
    assert update_float_acc36(None,sign,1,first=True) == FloatAcc36State(0,0)
    inf = update_float_acc36(FloatAcc36State(sign*(1<<34),254),sign*(1<<34),254,first=False)
    assert inf == FloatAcc36State(0,255,sign)
    assert update_float_acc36(inf,-sign*(1<<34),254,first=False) == inf
    assert update_float_acc36(inf,0,127,first=True) == FloatAcc36State(0,0)
    assert update_float_acc36(inf,sign*(1<<34),127,first=True) == FloatAcc36State(sign*(1<<34),127)


def test_q234_min_normal_max_finite_and_adjacent_underflow():
    assert update_float_acc36(None,1<<34,1,first=True) == FloatAcc36State(1<<34,1)
    assert update_float_acc36(None,(1<<35)-1,254,first=True) == FloatAcc36State((1<<35)-1,254)
    # Exact min normal remains normal.
    assert update_float_acc36(FloatAcc36State(1<<34,1),0,1,first=False) == FloatAcc36State(1<<34,1)
    # Coefficients on the finest valid input grid cannot create a half-LSB
    # below E=1; the adjacent representable result instead flushes.
    assert update_float_acc36(FloatAcc36State(1<<34,1),-1,1,first=False) == FloatAcc36State(0,0)


def test_q234_infinity_snapshot_keeps_sign_group_and_tag():
    acc = ReferenceAccumulator(tag_w=4)
    tx = transaction(SourceFormat.E4M3,a=0x3838,b=0x3838,valid=3,first=3,tag=(7,11))
    acc.step(tx)
    acc.update_float(0,1<<34,254,first=True)
    acc.update_float(0,1<<34,254,first=False)
    s = acc.step(replace(tx,acc_first=0)).snapshot
    assert s.data[0] is None and s.exp[0] == 255 and s.infinity_sign == (1,0)
    assert s.group_valid == 3 and s.tag == (7,11)
    s = acc.step(replace(tx,raw_a=(0,)*16)).snapshot
    assert s.data == (0,0) and s.exp == (0,0) and s.infinity_sign == (0,0)


def test_q234_dual_groups_first_invalid_hold_metadata_and_atomic_block():
    acc = ReferenceAccumulator(tag_w=4)
    tx = transaction(SourceFormat.E4M3,a=0x3838,b=0x3838,valid=3,first=3,
                     mx=True,scale_a=(127,128),tag=(5,9))
    s = acc.step(tx).snapshot
    assert s.data == (1<<34,1<<34) and s.exp == (127,128)
    assert s.tag == (5,9) and s.group_valid == 3
    s = acc.step(replace(tx,group_valid=2,acc_first=0,tag=(6,10))).snapshot
    assert s.data == (None,1<<34) and s.exp == (None,129)
    assert s.tag == (None,10)
    before = acc._float.copy()
    with pytest.raises(UnfrozenContract):
        acc.step(replace(tx,acc_first=0,scale_a=(127,254),scale_b=(127,254)))
    assert acc._float == before
    s = acc.step(replace(tx,group_valid=1,acc_first=0)).snapshot
    assert s.exp[0] == 128  # group0 was neither changed nor cleared by group1.
    zero = acc.step(replace(tx,raw_a=(0,)*16,acc_first=3)).snapshot
    assert zero.data == (0,0) and zero.exp == (0,0)
    assert zero.tag == (5,9) and zero.group_valid == 3


def test_q234_random_exact_fraction_oracle_and_37bit_tail_invariant():
    rng = random.Random(0x234127)
    for _ in range(3000):
        e0, e1 = rng.randrange(40,215), rng.randrange(40,215)
        c0, c1 = rng.randrange(-(1<<35),1<<35), rng.randrange(-(1<<35),1<<35)
        old = FloatAcc36State(c0,e0) if c0 else FloatAcc36State(0,0)
        u = trace_float_acc36_update(old,c1,e1,first=False)
        value = dyadic(c0,e0-161) + dyadic(c1,e1-161)
        assert dyadic(u.full_sum+u.old_tail+u.new_tail,u.common_exp-161) == value
        assert -(1<<35) <= u.old_operand < (1<<35)
        assert -(1<<35) <= u.new_operand < (1<<35)
        assert -(1<<36) <= u.full_sum < (1<<36)
        if not value:
            assert u.state == FloatAcc36State(0,0)
            continue
        # Independent Fraction oracle: bracket the value by adjacent Q2.34
        # grid points, choose nearest, resolving equal distances by evenness.
        magnitude = abs(value)
        exponent = 127
        while magnitude < 1:
            magnitude *= 2
            exponent -= 1
        while magnitude >= 2:
            magnitude /= 2
            exponent += 1
        scaled = magnitude*(1<<34)
        lo = scaled.numerator // scaled.denominator
        chosen = min((lo,lo+1),key=lambda x:(abs(scaled-x),x&1))
        if chosen == 1<<35:
            chosen //= 2
            exponent += 1
        expected = FloatAcc36State(chosen if value>0 else -chosen,exponent)
        assert u.state == expected


def test_q234_zero_bf16_ignores_unencoded_zero_mx_exponent():
    tx = transaction(SourceFormat.BF16)
    assert stages(tx)[-1].exp[0] == -122
    assert ReferenceAccumulator(tag_w=4).step(tx).snapshot.exp[0] == 0


def test_mxint8_exact_zero_sum_and_independent_scale_metadata():
    tx = transaction(SourceFormat.INT8, valid=3, first=3, mx=True,
                     scale_a=(125,130),scale_b=(128,126))
    tx = replace(tx,raw_a=(0x017F,0x0181)*8,raw_b=(0x017F,)*16)
    e,m,a,p,r = stages(tx)
    assert p.contribution0 == 0 and p.contribution1 == 16
    assert r.contribution == (0,16<<17) and r.exp == (131,134)
    snapshot = ReferenceAccumulator(tag_w=4).step(tx).snapshot
    assert snapshot.data[0] == 0 and snapshot.exp[0] == 0
    assert snapshot.tag == tx.tag and snapshot.group_valid == 3
