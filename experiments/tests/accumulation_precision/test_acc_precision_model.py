from fractions import Fraction
import random
import pytest
from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.accumulation_precision.model import State, prepare_blocks, evaluate
from mac_experiment.local_alignment.quantize import Quantized


def oracle(x, p):
    if not x: return Fraction(0)
    h=x.numerator.bit_length()-x.denominator.bit_length()
    if abs(x) < Fraction(2)**h: h-=1
    unit=Fraction(2)**(h-p+1)
    return round(x/unit)*unit


@pytest.mark.parametrize('p',[4,40,44,48])
def test_fraction_oracle(p):
    rng=random.Random(5)
    state=State(p)
    for _ in range(300):
        v=Dyadic(rng.randrange(-10000,10001),rng.randrange(-250,250))
        expected=oracle(state.value.to_fraction()+v.to_fraction(),p)
        state=state.update(v)
        assert state.value.to_fraction()==expected
        assert State.from_dict(state.to_dict())==state


def test_ties_cancellation_absorption_bounds():
    for sign in (1,-1):
        for n in (8,9,15):
            s=State(4,sign*n,3)
            assert s.update(Dyadic(sign,-1)).value.to_fraction()==oracle(Fraction(sign*n)+Fraction(sign,2),4)
    s=State(40).update(Dyadic(1,511))
    assert s.update(Dyadic(1,-100))==s
    assert s.update(-s.value)==State(40)
    assert State(40).update(Dyadic(1,-512)).H==-512
    with pytest.raises(OverflowError): State(40).update(Dyadic(1,512))
    with pytest.raises(OverflowError): State(40).update(Dyadic(1,-513))


def test_resume_interleave_zero():
    xs=[Dyadic(123,40),Dyadic(-127,32),Dyadic(1,-80),Dyadic(-3,44)]
    a=b=State(40)
    for v in xs:
        a=a.update(v)
        b=State.from_dict(b.to_dict()).update(v)
        assert a==b
        assert a.update(Dyadic.zero())==a
    contexts=[State(40),State(44)]
    for v in xs:
        for i in range(2): contexts[i]=contexts[i].update(v)
    for i,p in enumerate((40,44)):
        s=State(p)
        for v in xs: s=s.update(v)
        assert s==contexts[i]


def test_tail_padding_updates():
    a=Quantized([0x3c00]*17,[],{})
    r=evaluate(a,a,'fp16',17,40)
    assert r['updates']==3
    assert Dyadic.from_dict(r['S'])==Dyadic(17,0)
    padded=Quantized(a.codes+[0]*7,[],{})
    assert evaluate(padded,padded,'fp16',24,40)['C']==r['C']


@pytest.mark.parametrize('scale',[0,254])
def test_extreme_mx_scale(scale):
    a=Quantized([0x38]*32,[scale],{})
    r=evaluate(a,a,'mxfp8_e4m3fn',32,40)
    expected=Dyadic(32,2*(scale-127))
    assert Dyadic.from_dict(r['S'])==expected
    assert Dyadic.from_dict(r['V'])==expected
    assert r['updates']==2
    assert -512 <= r['H_min'] <= r['H_max'] <= 511
