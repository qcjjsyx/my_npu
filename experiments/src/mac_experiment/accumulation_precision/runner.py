"""Reproducible accumulation precision runs; formal execution is user initiated."""
import csv
import gzip
import hashlib
import itertools
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml
from ..arithmetic.dyadic import Dyadic
from ..local_alignment.model import PAIRS
from ..local_alignment.quantize import Quantized, quantize
from ..local_alignment.runner import Errors, DistributionCounts, write_json
from ..widthselect.distributions import WIDTH_DISTRIBUTIONS, sample_fp32_matrix, distribution_metadata
from .model import FLOORS, evaluate, prepare_blocks

VERSION = 'accumulation-precision-v1'


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    defaults = dict(seeds=[1], sequences=1, lengths=[32], formats=list(FLOORS),
                    p_acc=[40,44,48], distributions=['gaussian'],
                    output_root='results/numerical/accumulation_precision/runs')
    if not isinstance(config, dict) or set(config)-set(defaults):
        raise ValueError('invalid configuration keys')
    config = defaults | config
    for key in ('seeds','lengths','formats','p_acc','distributions'):
        xs = config[key]
        if not isinstance(xs,list) or not xs or len(set(xs)) != len(xs):
            raise ValueError(key+' must be a nonempty unique list')
    for key in ('seeds','lengths','p_acc'):
        if any(type(x) is not int for x in config[key]):
            raise ValueError(key+' must contain integers')
    if type(config['sequences']) is not int or config['sequences'] < 1:
        raise ValueError('invalid sequences')
    if any(x < 1 or x > 16384 for x in config['lengths']):
        raise ValueError('length outside [1,16384]')
    unknown_modes = set(config['formats'])-FLOORS.keys()
    if unknown_modes:
        raise ValueError(f'unsupported formats: {sorted(unknown_modes)}')
    if any(p < 40 for p in config['p_acc']):
        raise ValueError('p_acc values must be integers >=40')
    unknown_distributions = set(config['distributions'])-set(WIDTH_DISTRIBUTIONS)
    if unknown_distributions:
        raise ValueError(f'unsupported distributions: {sorted(unknown_distributions)}')
    if not isinstance(config['output_root'],str) or not config['output_root']:
        raise ValueError('invalid output root')
    return config


def summarize(records):
    rows=[]
    groups={}
    for r in records:
        groups.setdefault(tuple(r[k] for k in ('distribution','mode','length','p_acc')),[]).append(r)
    for key, rs in groups.items():
        row=dict(zip(('distribution','mode','length','p_acc'),key), count=len(rs))
        for label, candidate, reference in (('local','V','S'),('state','C','V'),('total','C','S')):
            errors=Errors()
            for r in rs:
                errors.add(Dyadic.from_dict(r[candidate]),Dyadic.from_dict(r[reference]),Dyadic.from_dict(r['L1']))
            row.update({label+'_'+k:v for k,v in errors.summary().items()})
        for label in ('reference','local'):
            cs=[r[label+'_comparison'] for r in rs]
            for subset, values in (('all',cs),('unsaturated',[c for c in cs if not c['saturated']]),('saturated',[c for c in cs if c['saturated']])):
                prefix=label+'_'+subset
                row[prefix+'_count']=len(values)
                row[prefix+'_bitmatch']=sum(c['bitmatch'] for c in values)
                row[prefix+'_bitmatch_rate']=sum(c['bitmatch'] for c in values)/len(values) if values else None
                ds=DistributionCounts()
                for c in values: ds.add(c['ulp'])
                row.update(ds.quantiles(prefix+'_ulp'))
                row[prefix+'_ulp_max']=max((c['ulp'] for c in values),default=None)
            row[label+'_signed_zero_mismatch']=sum(c['signed_zero_mismatch'] for c in cs)
        for name in ('updates','absorbed','output_zero','output_subnormal','output_saturated','reference_saturated','local_saturated'):
            row[name]=sum(r[name] for r in rs)
        row['H_min']=min((r['H_min'] for r in rs if r['H_min'] is not None),default=None)
        row['H_max']=max((r['H_max'] for r in rs if r['H_max'] is not None),default=None)
        rows.append(row)
    return rows


def run_from_file(path, verbose=True):
    config=load_config(path)
    run=Path(config['output_root'])/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',config)
    records=[]; representatives={}; quant_events={}
    maximum=max(config['lengths'])
    with gzip.open(run/'sources.jsonl.gz','wt') as sources, gzip.open(run/'samples.jsonl.gz','wt') as samples:
        for dist,seed,seq in itertools.product(config['distributions'],config['seeds'],range(config['sequences'])):
            sid=f'{dist}:{seed}:{seq}'
            ab=sample_fp32_matrix(dist,seed,'x',1,maximum,seq)
            bb=sample_fp32_matrix(dist,seed,'w',1,maximum,seq)
            targets={fmt:(quantize(ab,fmt,fmt.startswith('e')),quantize(bb,fmt,fmt.startswith('e'))) for fmt in sorted({f for m in config['formats'] for f in PAIRS[m]})}
            sources.write(json.dumps(dict(id=sid,A_fp32=ab,B_fp32=bb,targets={f:[asdict(q) for q in qs] for f,qs in targets.items()}))+'\n')
            for fmt,qs in targets.items():
                for side,q in zip('AB',qs):
                    key=f'{dist}:{fmt}:{side}'
                    counts=quant_events.setdefault(key,{k:0 for k in q.events})
                    for k,v in q.events.items(): counts[k]+=v
            for mode,length in itertools.product(config['formats'],config['lengths']):
                af,bf=PAIRS[mode]; a,b=targets[af][0],targets[bf][1]
                local=prepare_blocks(a,b,mode,length)
                for p in config['p_acc']:
                    r=evaluate(a,b,mode,length,p,local)
                    r.update(source_id=sid,distribution=dist)
                    trace=r.pop('trace')
                    records.append(r)
                    samples.write(json.dumps(r)+'\n')
                    key=(dist,mode,length,p)
                    error=abs(Dyadic.from_dict(r['state_error']))
                    if key not in representatives or error > representatives[key][0]:
                        representatives[key]=(error,dict(**r,trace=trace))
            if verbose: print('completed '+sid)
    rows=summarize(records)
    write_json(run/'summary.json',rows)
    with (run/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(run/'representatives.json',[v[1] for v in representatives.values()])
    write_json(run/'quantization.json',quant_events)
    def git(*args):
        result=subprocess.run(['git',*args],capture_output=True,text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    package=Path(__file__).resolve().parents[1]
    write_json(run/'metadata.json',dict(version=VERSION,git_head=git('rev-parse','HEAD'),git_status=git('status','--short'),
        implementation_sha256={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*.py'))},
        config_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        distributions={d:distribution_metadata(d) for d in config['distributions']},
        contract='exact dyadic addition then one magnitude RNE; signed10 H; numerical reference, no fixed GRS hardware claim',
        quantiles='nearest-rank empirical p50/p99; binary64 display; saturation excluded from unsaturated ULP',
        sample_count=len(records),source_count=len(config['distributions'])*len(config['seeds'])*config['sequences']))
    write_json(run/'COMPLETE.json',dict(version=VERSION,sample_count=len(records)))
    return run


def replay(run,index=0):
    run=Path(run)
    if not (run/'COMPLETE.json').exists(): raise ValueError('run is incomplete')
    representatives=json.loads((run/'representatives.json').read_text())
    if not 0 <= index < len(representatives): raise ValueError('representative index out of range')
    rep=representatives[index]
    with gzip.open(run/'sources.jsonl.gz','rt') as handle:
        source=next((s for s in map(json.loads,handle) if s['id']==rep['source_id']),None)
    if source is None: raise ValueError('source missing')
    af,bf=PAIRS[rep['mode']]
    result=evaluate(Quantized(**source['targets'][af][0]),Quantized(**source['targets'][bf][1]),rep['mode'],rep['length'],rep['p_acc'])
    result.update(source_id=rep['source_id'],distribution=rep['distribution'])
    if result != rep: raise ValueError('representative replay mismatch')
    return result
