import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from mac_experiment.arithmetic.dyadic import Dyadic
from mac_experiment.local_alignment.runner import Errors, load_config, replay, run_from_file


def test_config_validation(tmp_path):
    path=tmp_path/'config.yaml'
    for payload in ({'unknown':1},{'lengths':[0]},{'lengths':[16385]},{'F':[-1]},{'rounding':['truncate']},
                    {'formats':['fp8']},{'sequences':0},{'seeds':[True]},{'lengths':[32,32]}):
        path.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError):
            load_config(path)
    path.write_text('{}')
    assert load_config(path)['F']==[0,1,2,3,4,6,8,12,16]


def test_exact_quantiles_and_zero_denominator():
    metrics=Errors()
    for i in range(1,101):
        metrics.add(Dyadic(i,0),Dyadic.zero(),Dyadic(100,0))
    summary=metrics.summary()
    assert summary['absolute_p50']==50
    assert summary['absolute_p99']==99
    assert summary['error_over_l1_p50']==0.5
    assert summary['error_over_l1_p99']==0.99
    metrics.add(Dyadic.zero(),Dyadic.zero(),Dyadic.zero())
    assert metrics.summary()['zero_l1']==1
    assert metrics.summary()['error_over_l1_count']==100


def test_cli_outputs_replay_no_overwrite_and_shared_source(tmp_path):
    config=dict(seeds=[7],sequences=2,lengths=[32,64],F=[0,4],rounding=['floor','rne'],
                distributions=['positive'],formats=['mxfp8_e4m3fn','mxfp8_e4m3fn_e5m2','fp16','bf16'],
                output_root=str(tmp_path/'runs'))
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    proc=subprocess.run([sys.executable,'-m','mac_experiment','local-align-run','--config',str(path),'--quiet'],
                        capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    run=Path(proc.stdout.strip())
    rows=json.loads((run/'summary.json').read_text())
    assert len(rows)==32
    for row in rows:
        assert row['dot_count']==2
        assert row['local_count']==2*((row['length']+row['K']-1)//row['K'])
        assert row['sum_width']==row['window_width']+(row['K']-1).bit_length()
        if row['rounding']=='floor':
            assert row['dot_positive']==0
        assert row['fp32_unsaturated_bitmatch']<=row['fp32_unsaturated_count']
    with gzip.open(run/'sources.jsonl.gz','rt') as handle:
        sources=[json.loads(line) for line in handle]
    assert len(sources)==2
    assert len(sources[0]['targets']['e4m3fn'][0]['codes'])==64
    metadata=json.loads((run/'metadata.json').read_text())
    assert metadata['version']=='local-alignment-v2'
    first=replay(run,0)
    with gzip.open(run/'samples.jsonl.gz','rt') as handle:
        saved=next(x for x in map(json.loads,handle) if all(x[k]==first[k] for k in ('source_id','mode','length','F','rounding')))
    for name in ('S','T','L1','reference_bits','output_bits'):
        assert first[name]==saved[name]
    rerun=run_from_file(path,verbose=False)
    assert rerun!=run and run.exists()
    assert (rerun/'summary.json').read_bytes()==(run/'summary.json').read_bytes()
    result=subprocess.run([sys.executable,'-m','mac_experiment','local-align-replay','--run',str(run)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==first


def test_saturation_cannot_mask_internal_loss_and_saved_reference(tmp_path,monkeypatch):
    from fractions import Fraction
    from mac_experiment.arithmetic.fp32 import Fp32
    import mac_experiment.local_alignment.runner as runner
    values=[Fp32.from_float(1e30).bits, Fp32.from_float(1.0).bits]*16
    monkeypatch.setattr(runner,'sample_fp32_matrix',lambda *args: list(values))
    path=tmp_path/'sat.yaml'
    path.write_text(yaml.safe_dump(dict(seeds=[1],sequences=1,lengths=[32],F=[0],rounding=['floor'],
                                       distributions=['positive'],formats=['bf16'],output_root=str(tmp_path/'runs'))))
    run=run_from_file(path,False)
    row=json.loads((run/'summary.json').read_text())[0]
    assert row['reference_saturated']==row['output_saturated']==1
    assert row['saturated_bitmatch']==1
    assert row['fp32_unsaturated_count']==0
    assert row['fp32_unsaturated_bitmatch_rate'] is None
    assert row['dot_exact_match']==0
    assert row['dot_absolute_mean']>0
    assert row['changed']==row['zeroed']==16
    replayed=replay(run)
    with gzip.open(run/'sources.jsonl.gz','rt') as handle:
        src=json.loads(next(handle))
    a,b=[target['codes'] for target in src['targets']['bf16']]
    def decode_bf16(c):
        exp=(c>>7)&255
        frac=c&127
        return (-1 if c>>15 else 1)*Fraction((128 if exp else 0)+frac)*Fraction(2)**((exp or 1)-134)
    expected=sum((decode_bf16(ac)*decode_bf16(bc) for ac,bc in zip(a,b)),Fraction(0))
    assert Dyadic.from_dict(replayed['S']).to_fraction()==expected
    # The sixteen unit products are lost; large aligned products remain exact.
    assert Dyadic.from_dict(replayed['T']).to_fraction()==expected-16
    assert row['local_absolute_mean']==8
    assert row['local_absolute_p50']==8
    assert row['dot_signed_mean']==-16


def test_custom_windows_and_partial_lengths(tmp_path):
    from mac_experiment.local_alignment.model import PAIRS
    config=dict(seeds=[3],sequences=1,lengths=[1,17,31,33],F=[3,18],
                rounding=['floor','rne'],distributions=['gaussian'],
                formats=list(PAIRS),output_root=str(tmp_path/'runs'))
    path=tmp_path/'custom.yaml'
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)['F']==[3,18]
    run=run_from_file(path,False)
    rows=json.loads((run/'summary.json').read_text())
    assert len(rows)==len(PAIRS)*4*2*2
    for row in rows:
        assert row['local_count']==(row['length']+row['K']-1)//row['K']
    # Replay every partial-length case, including BF16 postfold trace padding.
    reps=json.loads((run/'representatives.json').read_text())
    for i in range(len(reps)):
        result=replay(run,i)
        assert result['trace']==reps[i]['trace']


def test_partial_mx_scale_uses_full_source_block(tmp_path, monkeypatch):
    from fractions import Fraction
    from mac_experiment.arithmetic.fp32 import Fp32
    from mac_experiment.local_alignment.model import element_value
    import mac_experiment.local_alignment.runner as runner
    # A prefix shares the scale selected for the full source block, even when
    # its largest value lies beyond the requested prefix.
    bits=[Fp32.from_float(1).bits]*16+[Fp32.from_float(128).bits]
    monkeypatch.setattr(runner,'sample_fp32_matrix',lambda *args: bits)
    path=tmp_path/'prefix.yaml'
    path.write_text(yaml.safe_dump(dict(lengths=[1,17],F=[18],rounding=['floor'],
        formats=['mxfp8_e4m3fn'],distributions=['gaussian'],output_root=str(tmp_path/'runs'))))
    run=run_from_file(path,False)
    with gzip.open(run/'sources.jsonl.gz','rt') as f: source=json.loads(next(f))
    a,b=source['targets']['e4m3fn']
    with gzip.open(run/'samples.jsonl.gz','rt') as f: samples=list(map(json.loads,f))
    for sample in samples:
        expected=Fraction(0)
        for i in range(sample['length']):
            expected+=element_value(a['codes'][i],'e4m3fn').to_fraction()*element_value(b['codes'][i],'e4m3fn').to_fraction()*Fraction(2)**(a['scales'][i//32]+b['scales'][i//32]-254)
        assert Dyadic.from_dict(sample['S']).to_fraction()==expected
