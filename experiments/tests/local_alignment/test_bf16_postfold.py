"""BF16 multiply-then-fold exploration, independent original-value Fraction oracle."""
from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import yaml

from mac_experiment.local_alignment.model import align, postfold_product
from mac_experiment.local_alignment.quantize import Quantized
from mac_experiment.local_alignment.runner import blocks, evaluate, load_config, replay


def raw(code):
    """Finite BF16 coefficient and effective exponent, independent of DUT decoder."""
    exp=(code>>7)&255
    frac=code&127
    magnitude=(128 if exp else 0)+frac
    return (-magnitude if code>>15 else magnitude),(exp or 1)


def pow2(e):
    return Fraction(2)**e


def quantized(codes):
    return Quantized(codes,[],{})


@pytest.mark.parametrize('r',range(4))
@pytest.mark.parametrize('p',[-65025,-1,0,1,65025])
def test_recoding_exact_all_residues(p,r):
    t=496+r
    coefficient,q=postfold_product(p,t)
    assert coefficient*pow2(4*q-268)==p*pow2(t-268)
    assert (coefficient,q)==((p<<r,t//4) if p else (0,0))
    assert -(1<<20)<=coefficient<(1<<20)


@pytest.mark.parametrize('rounding',['floor','rne'])
def test_original_fraction_oracle(rounding):
    rng=random.Random(634)
    for _ in range(50):
        # All codes finite; normals, subnormals, signs and exact zeros are included.
        a=[rng.randrange(0x7f80)|(rng.getrandbits(1)<<15) for i in range(16)]
        b=[rng.randrange(0x7f80)|(rng.getrandbits(1)<<15) for i in range(16)]
        a[0]=0
        b[0]=0x7f7f  # high exponent zero product must not affect MAX
        original=[(raw(ac)[0]*raw(bc)[0],raw(ac)[1]+raw(bc)[1]) for ac,bc in zip(a,b)]
        qmax=max(t//4 for p,t in original if p)
        expected_n=[]
        for p,t in original:
            x=p*pow2(t-4*qmax+11)
            n=x.numerator//x.denominator
            if rounding=='rne':
                remainder=x-n
                n+= remainder>Fraction(1,2) or (remainder==Fraction(1,2) and n%2)
            expected_n.append(n)
        transformed,h=next(blocks(quantized(a),quantized(b),'bf16_postfold',16))
        result=align(transformed,'bf16_postfold',11,rounding,h)
        assert result.tmax==qmax
        assert result.integer_sum==sum(expected_n)
        assert result.q.to_fraction()==sum(expected_n)*pow2(4*qmax-279)
        assert result.s.to_fraction()==sum(p*pow2(t-268) for p,t in original)
        assert result.l1.to_fraction()==sum(abs(p*pow2(t-268)) for p,t in original)


@pytest.mark.parametrize('r',range(4))
def test_same_32bit_window_equivalence_and_loss_boundary(r):
    a=[124<<7,0x8001]+[0]*14
    b=[(124+r)<<7,1]+[0]*14
    old=evaluate(quantized(a),quantized(b),'bf16',16,14,'floor')
    new=evaluate(quantized(a),quantized(b),'bf16_postfold',16,11,'floor')
    assert old['S']==new['S']
    # Old absolute LSB Tmax-282; new 4*floor(Tmax/4)-279.
    # For r=3 both grids coincide; r=0..2 postfold discards up to 3 more bits.
    old_lsb=pow2(248+r-282)
    new_lsb=pow2(248-279)
    assert new['T'].to_fraction()-old['T'].to_fraction()==old_lsb-new_lsb
    assert (new['T']==old['T'])==(r==3)
    assert new['trace']['multiplication_product_width']==18
    assert new['trace']['alignment_coefficient_width']==21
    assert new['trace']['terms'][0]['original_T']==248+r
    assert new['trace']['terms'][0]['fold_r']==r
    # Equal exponents are exact in either window even if their absolute grids differ.
    a=[(124<<7)|127]*16
    b=[((124+r)<<7)|127]*16
    old=evaluate(quantized(a),quantized(b),'bf16',16,14,'floor')
    new=evaluate(quantized(a),quantized(b),'bf16_postfold',16,11,'floor')
    assert old['T']==new['T']==old['S']==new['S']


def test_nonzero_max_all_zero_cancellation_and_full_precision():
    a=quantized([0]*16)
    b=quantized([0x7f7f]*16)
    products,h=next(blocks(a,b,'bf16_postfold',16))
    assert align(products,'bf16_postfold',11,'floor').tmax is None
    a=quantized([0x7f7f,0xff7f]+[0]*14)
    b=quantized([0x7eff,0x7eff]+[0]*14)
    result=evaluate(a,b,'bf16_postfold',16,11,'floor')
    assert result['S'].m==result['T'].m==0
    assert abs(result['trace']['terms'][0]['P'])==65025*8
    tiny=quantized([1,0x8001]+[0]*14)
    exact=evaluate(a,tiny,'bf16_postfold',16,1024,'rne')
    assert exact['T']==exact['S']


def test_cli_replay_and_width_metadata(tmp_path):
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(dict(seeds=[8],sequences=1,lengths=[32],formats=['bf16_postfold'],
                                       F=[11],rounding=['floor'],distributions=['gaussian'],output_root=str(tmp_path/'runs'))))
    assert load_config(path)['formats']==['bf16_postfold']
    proc=subprocess.run([sys.executable,'-m','mac_experiment','local-align-run','--config',str(path),'--quiet'],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    run=Path(proc.stdout.strip())
    row=json.loads((run/'summary.json').read_text())[0]
    assert (row['product_width'],row['coefficient_width'],row['window_width'],row['sum_width'])==(18,21,32,36)
    record=replay(run)
    assert record['mode']=='bf16_postfold'
    proc=subprocess.run([sys.executable,'-m','mac_experiment','local-align-replay','--run',str(run)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)==record
