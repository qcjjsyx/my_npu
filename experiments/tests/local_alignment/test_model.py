"""Independent Fraction encoding oracle; never calls DUT decoding to obtain expectations."""
from bisect import bisect_left
from fractions import Fraction
import random

import pytest

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.arithmetic.fp32 import Fp32
from mac_experiment.local_alignment.model import (FORMATS, LAYOUTS, PAIRS, align, decode,
                                                  element_value, output_fp32, family)
from mac_experiment.local_alignment.quantize import encode, quantize
from mac_experiment.local_alignment.runner import evaluate


def power(e):
    return Fraction(2) ** e


def oracle_value(code, fmt):
    eb, fb, bias, maximum = FORMATS[fmt]
    sign = -1 if code >> (eb + fb) else 1
    mag = code & ((1 << (eb + fb)) - 1)
    e, f = mag >> fb, mag & ((1 << fb) - 1)
    if fmt == "e4m3fn":
        if mag == 127:
            return Fraction(0)
    elif e == (1 << eb) - 1:
        if f:
            return Fraction(0)
        return oracle_value((int(sign < 0) << (eb + fb)) | maximum, fmt)
    return sign * (Fraction(f, 1 << fb) + bool(e)) * power((e or 1) - bias)


@pytest.mark.parametrize('fmt', FORMATS)
def test_decode_all_codes(fmt):
    eb, fb, _, _ = FORMATS[fmt]
    for code in range(1 << (eb + fb + 1)):
        assert element_value(code, fmt).to_fraction() == oracle_value(code, fmt)
    bound = {'e4m3fn':120, 'e5m2':112, 'fp16':16376, 'bf16':255}[fmt]
    assert max(abs(decode(code, fmt)[0]) for code in range(1 << (eb+fb+1))) == bound


@pytest.mark.parametrize('fmt', FORMATS)
def test_encoder_against_nearest_enumerated_codes(fmt):
    eb, fb, bias, maximum = FORMATS[fmt]
    # Enumerate exact finite source format numbers independently of DUT decode/rounding.
    values = [oracle_value(c, fmt) for c in range(maximum+1)]
    rng = random.Random(782)
    trials = [Fraction(0), values[-1]*2, values[1]/2, values[1]*Fraction(3,2)]
    for _ in range(150):
        index = rng.randrange(maximum)
        trials.extend([(values[index]+values[index+1])/2,
                       (values[index]*3+values[index+1])/4])
    for x in trials:
        pos = min(bisect_left(values, x), maximum)
        candidates = {pos, max(0,pos-1)}
        expected = min(candidates, key=lambda c: (abs(values[c]-x), c & 1))
        # Inputs are dyadic; no oracle use of DUT rounding.
        e = -(x.denominator.bit_length()-1)
        for sign in (1,-1):
            code, _ = encode(Dyadic(sign*x.numerator, e), fmt)
            assert code == expected | ((sign < 0 and x != 0) << (eb+fb))


def round_fraction(x, mode):
    lower = x.numerator // x.denominator
    if mode == 'floor':
        return lower
    delta = x-lower
    return lower + (delta > Fraction(1,2) or (delta == Fraction(1,2) and lower % 2 != 0))


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('rounding', ['floor','rne'])
@pytest.mark.parametrize('F', [0,1,4,16])
def test_local_fraction_oracle(layout, rounding, F):
    C,g,b,K = LAYOUTS[layout]
    rng = random.Random(777)
    for _ in range(20):
        products = [(rng.randrange(-(1<<(C-2)),1<<(C-2)),rng.randrange(64)) for i in range(K)]
        products[0] = (0,9999)  # zero must not enter MAX
        result = align(products,layout,F,rounding,h=-254)
        tmax = max(t for p,t in products if p)
        n = [round_fraction(p*power(F-g*(tmax-t)),rounding) for p,t in products]
        expected = sum(n)*power(g*tmax-b-F-254)
        assert result.q.to_fraction() == expected
        exact = [p*power(g*t-b-254) for p,t in products]
        assert result.s.to_fraction() == sum(exact)
        assert result.l1.to_fraction() == sum(map(abs,exact))


@pytest.mark.parametrize('rounding',['floor','rne'])
def test_zeros_maximum_cancellation_and_long_negative_shift(rounding):
    for layout,(C,g,b,K) in LAYOUTS.items():
        assert align([(0,999)]*K,layout,0,rounding).tmax is None
        products=[((1<<(C-1))-1,1)]*K
        result=align(products,layout,16,rounding)
        assert result.integer_sum == K*((1<<(C-1))-1)*(1<<16)
        products=[(3,500),(-3,500)]+[(0,9999)]*(K-2)
        assert align(products,layout,0,rounding).q.m == 0
        products=[(1,500),(-1,0)]+[(0,9999)]*(K-2)
        result=align(products,layout,0,rounding)
        assert result.terms[1]['n'] == (-1 if rounding=='floor' else 0)


@pytest.mark.parametrize('p,expected', [(1,0),(3,2),(5,2),(-1,0),(-3,-2),(-5,-2)])
def test_signed_half_ties(p,expected):
    result=align([(p,0),(1,1)]+[(0,0)]*14,'bf16',0,'rne')
    assert result.terms[0]['n']==expected


def test_output_contract():
    assert output_fp32(Dyadic(0,1000))[0].bits==0
    assert output_fp32(Dyadic(-1,-150))[0].bits==0x80000000
    assert output_fp32(Dyadic(1,-149))[0].bits==1
    assert output_fp32(Dyadic(3,-150))[0].bits==2
    for sign in (-1,1):
        fp,sat=output_fp32(Dyadic(sign,128))
        assert sat and fp.bits == (0x7f7fffff | ((sign<0)<<31))


@pytest.mark.parametrize('mode',PAIRS)
def test_scale_exchange_and_exact_full_precision(mode):
    rng=random.Random(232)
    a=[Fp32.from_float(rng.uniform(-1,1)).bits for _ in range(45)]
    b=[Fp32.from_float(rng.uniform(-100,100)).bits for _ in range(45)]
    af,bf=PAIRS[mode]
    qa=quantize(a,af,mode.startswith('mx'))
    qb=quantize(b,bf,mode.startswith('mx'))
    result=evaluate(qa,qb,mode,45,1024,'rne')
    ref=Fraction(0)
    for i,(ac,bc) in enumerate(zip(qa.codes,qb.codes)):
        h=qa.scales[i//32]+qb.scales[i//32]-254 if mode.startswith('mx') else 0
        ref+=oracle_value(ac,af)*oracle_value(bc,bf)*power(h)
    assert result['S'].to_fraction()==ref
    assert result['T']==result['S']
    swapped=mode if af == bf else next(m for m,pair in PAIRS.items() if pair==(bf,af))
    assert evaluate(qb,qa,swapped,45,1024,'rne')['T']==result['T']


@pytest.mark.parametrize('fmt',['e4m3fn','e5m2'])
def test_mx_scale_policy_independent_oracle(fmt):
    maximum=oracle_value(FORMATS[fmt][3],fmt)
    for value in [0.0,2**-149,1.0,448.0,449.0,57344.0,2**127]:
        bits=[Fp32.from_float(value).bits]*32
        result=quantize(bits,fmt,True)
        x=Fraction(value)
        expected=0
        if x:
            expected=next((e for e in range(-127,128) if x<=maximum*power(e)),127)
        assert result.scales == [expected+127]
        if x:
            assert x<=maximum*power(expected) or expected==127
            assert expected==-127 or x>maximum*power(expected-1)
    zero=quantize([0x80000000]*32,fmt,True)
    assert zero.scales==[127] and all(c==128 for c in zero.codes)


def test_quantization_events():
    bits=[0x7f7fffff,1,0x33800000,0x80000000]
    result=quantize(bits,'fp16')
    assert result.events['saturated']==1
    assert result.events['zeroed']==1
    assert result.events['subnormal']==1
    assert result.codes[-1]==0x8000
