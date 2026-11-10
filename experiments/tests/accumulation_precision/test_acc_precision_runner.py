import gzip
import json
from pathlib import Path
import yaml
from mac_experiment.accumulation_precision.runner import load_config, run_from_file, replay, summarize


def test_formal_count():
    c=load_config('experiments/configs/accumulation_precision/formal_template.yaml')
    assert len(c['seeds'])*c['sequences']*len(c['distributions'])==768
    assert c['p_acc']==[40,42,44,46,48]
    assert 768*len(c['formats'])*len(c['lengths'])*len(c['p_acc'])==138240


def test_smoke_saved_replay(tmp_path):
    c=load_config('experiments/configs/accumulation_precision/smoke.yaml')
    c['output_root']=str(tmp_path/'runs')
    path=tmp_path/'c.yaml'; path.write_text(yaml.safe_dump(c))
    run=run_from_file(path,verbose=False)
    assert (run/'COMPLETE.json').exists()
    with gzip.open(run/'samples.jsonl.gz','rt') as f: records=list(map(json.loads,f))
    assert len(records)==36
    assert summarize(records)==json.loads((run/'summary.json').read_text())
    reps=json.loads((run/'representatives.json').read_text())
    for i,r in enumerate(reps): assert replay(run,i)==r
    assert len({r['source_id'] for r in records})==1


def test_saturation_and_zero_strata():
    from mac_experiment.accumulation_precision.model import evaluate
    from mac_experiment.local_alignment.quantize import Quantized
    a=Quantized([0],[],{})
    zero=evaluate(a,a,'fp16',1,40)
    zero.update(distribution='synthetic')
    zero['reference_comparison']['signed_zero_mismatch']=True
    saturated=dict(zero)
    saturated['reference_comparison']=dict(bitmatch=True,ulp=999,saturated=True,signed_zero_mismatch=False)
    saturated['local_comparison']=saturated['reference_comparison']
    row=summarize([zero,saturated])[0]
    assert row['reference_unsaturated_count']==1
    assert row['reference_unsaturated_ulp_max']==0
    assert row['reference_saturated_bitmatch']==1
    assert row['reference_signed_zero_mismatch']==1


def test_failure_no_completion(tmp_path,monkeypatch):
    import pytest
    from mac_experiment.accumulation_precision import runner
    c=load_config('experiments/configs/accumulation_precision/smoke.yaml')
    c['output_root']=str(tmp_path/'runs')
    path=tmp_path/'c.yaml';path.write_text(yaml.safe_dump(c))
    def fail(*args,**kwargs): raise OverflowError('signed10')
    monkeypatch.setattr(runner,'evaluate',fail)
    with pytest.raises(OverflowError): run_from_file(path,verbose=False)
    assert len(list((tmp_path/'runs').iterdir()))==1
    assert not list((tmp_path/'runs').glob('*/COMPLETE.json'))


def test_negative_replay_index(tmp_path):
    import pytest
    (tmp_path/'COMPLETE.json').write_text('{}')
    (tmp_path/'representatives.json').write_text('[{}]')
    with pytest.raises(ValueError,match='index'): replay(tmp_path,-1)


def test_config_accepts_intermediate_precisions(tmp_path):
    import pytest
    from fractions import Fraction
    from mac_experiment.accumulation_precision.model import State
    from mac_experiment.arithmetic.dyadic import Dyadic
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(dict(p_acc=[40,41,42,46,48,50], lengths=[1024])))
    assert load_config(path)['p_acc']==[40,41,42,46,48,50]
    for p in (41,42,46,50):
        old=State(p).update(Dyadic(1,0))
        assert old.update(Dyadic(1,-p))==old  # exactly half an ULP, even
        assert old.update(Dyadic(3,-p)).value.to_fraction()==Fraction(1)+Fraction(2)**(2-p)
    path.write_text(yaml.safe_dump(dict(p_acc=[39])))
    with pytest.raises(ValueError, match='p_acc'): load_config(path)
