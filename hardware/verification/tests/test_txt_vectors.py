"""TXT delivery contract, direct packed entry, and actual RTL consumers."""
from dataclasses import replace
from pathlib import Path
import pytest
from generate_vectors import generate, mac_sequences, pack_expected, f19
from reference.packed_mac import PackedAccumulator
from reference.mac_contract import ContractViolation, UnfrozenContract, SourceFormat as F
from run_txt_vectors import MAC_SOURCES, compile_tb, run, simulate


def test_pack_hand_calculated_specials():
    assert pack_expected(0x3c00,F.FP16,1)==0x0f0400
    assert pack_expected(1,F.FP16,1)==0x010001
    assert pack_expected(0x8000,F.FP16,1)==0
    assert pack_expected(0x7e00,F.FP16,1)==0
    assert pack_expected(0x7c00,F.FP16,1)==0x1e07ff
    assert pack_expected(0x3f80,F.BF16,1)==0x7f0400
    assert pack_expected(0xffff,F.INT8,1)==255
    assert pack_expected(0xffff,F.INT8,2)==65280
    assert pack_expected(0xffff,F.FP16,2)==0


def test_packed_entry_does_not_call_raw_converter(monkeypatch):
    import reference.mac_contract as contract
    def forbidden(*a,**kw):raise AssertionError('raw conversion used in packed path')
    monkeypatch.setattr(contract,'enter',forbidden)
    monkeypatch.setattr(contract,'_decode_finite_float',forbidden)
    acc=PackedAccumulator()
    steps=list(mac_sequences(1,4,False))
    expected,(_,a,_,x)=acc.evaluate(steps[0])
    assert a.emax==(32,20) and x.exp==(150,138)
    assert x.contribution==(16*64*(1<<23),16*64*(1<<23))
    assert expected[0]==1
    for s in steps[1:]:acc.evaluate(s)


def test_inactive_nonzero_payload_is_masked():
    s=next(mac_sequences(1,4,False))
    _,(m,a,p,x)=PackedAccumulator().evaluate(replace(s,group_valid=2,acc_first=2))
    assert not any(m.product0) and not any(m.exp_sum0)
    assert not any(a.lane0) and p.contribution0==0 and x.contribution[0]==0


def test_invalid_and_deferred_domains_are_not_wrapped():
    s=next(mac_sequences(1,4,False))
    with pytest.raises(ContractViolation,match='first required'):
        PackedAccumulator().evaluate(replace(s,acc_first=0))
    with pytest.raises(ContractViolation,match='scale255'):
        PackedAccumulator().evaluate(replace(s,scale_a=(255,127)))
    acc=PackedAccumulator()
    with pytest.raises(UnfrozenContract):
        acc.evaluate(replace(s,scale_a=(254,254),scale_b=(254,254)))
    assert not any(acc.valid) and all(f.sig==0 for f in acc.floating)


def test_int16_explicit_pairing_and_commit():
    seq=[s for s in mac_sequences(1,2,False) if int(s.mode)==2]
    first,second=seq[:2]
    acc=PackedAccumulator()
    result,_=acc.evaluate(first)
    assert result[0]==0 and result[5]==0
    with pytest.raises(ContractViolation,match='pairing'):
        acc.evaluate(replace(second,tag=(15,15)))
    result,_=acc.evaluate(second)
    assert result[0]==1 and acc.integer[0]==2147483647
    with pytest.raises(ContractViolation,match='phase1'):
        PackedAccumulator().evaluate(second)


def test_txt_reproducible_and_rtl_consumers(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    counts=generate(a,seed=43,random_count=8)
    assert generate(b,seed=43,random_count=8)==counts
    for file in a.glob('*.txt'):assert file.read_bytes()==(b/file.name).read_bytes()
    assert all('entry_m_stage' not in s and 'operand_pack' not in s for s in MAC_SOURCES)
    report=run(a,tmp_path/'build')
    assert 'schedule=0' in report and 'schedule=1' in report
    assert report.count('Expected assertion PASS')==4
    # Corrupt the expected output: this must produce a failure, not a false pass.
    expected=a/'mac_expected.txt'
    lines=expected.read_text().splitlines();fields=lines[0].split()
    fields[10]=f'{int(fields[10],16)^1:09x}'
    lines[0]=' '.join(fields);expected.write_text('\n'.join(lines)+'\n')
    with pytest.raises(RuntimeError,match='ACC state mismatch'):
        simulate(tmp_path/'build/mac.vvp',[f'+INPUTS={a/"mac_inputs.txt"}',f'+EXPECTED={expected}',f'+INTERMEDIATE={a/"mac_intermediate.txt"}'])


def test_directed_ties_ftz_inf_and_cancellation():
    acc=PackedAccumulator()
    saw_positive_tie=saw_negative_tie=saw_ftz=saw_inf=saw_clear=False
    for step in mac_sequences(3,2,False):
        _,(_,aligned,_,_)=acc.evaluate(step)
        if step.mode==0 and step.operand_a[1] & 0xfff in (1025,(-1025)&0xfff) and step.operand_b[1]==f19(1025,15):
            value=aligned.lane0[1]
            assert aligned.inexact[0] and abs(value)==525312
            saw_positive_tie |= value>0
            saw_negative_tie |= value<0
        if step.scale_a==(61,61):
            assert all(f.sig==0 and f.exp==0 for f in acc.floating)
            saw_ftz=True
        if acc.floating[0].infinity_sign==1 and acc.floating[1].infinity_sign==-1:
            saw_inf=True
        if saw_inf and step.acc_first==3 and not any(step.operand_a):
            assert all(f.sig==0 and f.exp==0 and f.infinity_sign==0 for f in acc.floating)
            saw_clear=True
    assert all((saw_positive_tie,saw_negative_tie,saw_ftz,saw_inf,saw_clear))
