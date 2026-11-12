"""Versioned repair: correlated line series, with fixed crossing-series coverage."""
from pathlib import Path
from collections import Counter
import numpy as np
from .domain_data import make_domain_spec,render_domain,validate_domain,annotation_overlay
from .io import read_json,write_json,write_jsonl,sha256
from .synthetic import fingerprint


def repaired_spec(cfg,index,split):
    spec=make_domain_spec(cfg,index,split); spec['parent_id']=spec['id']; spec['id']='adapt2-'+spec['id']
    spec['generator_version']='correlated-lines-v2'
    if spec['chart_type']=='line':
        values=np.asarray(spec['data']['values']); ns,n=values.shape
        lo,hi=float(values.min()),float(values.max()); u=(values-lo)/max(hi-lo,1e-9)
        t=np.linspace(-1.,1.,n); innovation=np.cumsum(u-.5,axis=1)
        innovation-=innovation.mean(axis=1,keepdims=True)
        crossing=(index//5)%4==0
        if crossing:
            # Explicit crossings retained in one quarter of line charts; no rejection.
            levels=.5+np.linspace(-.30,.30,ns)[:,None]*t+.025*innovation
        else:
            levels=np.linspace(.18,.82,ns)[:,None]+.10*t+.025*innovation
        values=np.round(lo+(hi-lo)*levels,4)
        spec['data']['values']=values.tolist(); spec['data_hash']=fingerprint(spec['data'])
        spec['numeric_hash']=fingerprint(values.tolist()); spec['line_process']='crossing' if crossing else 'common_trend'
    return spec


def prepare(root,cfg,limit=None):
    root=Path(root); output=root/cfg['synthetic_output']; total=cfg['train_count']+cfg['val_count']; count=total if limit is None else min(limit,total)
    if count<total and (output/'manifest.json').exists() and read_json(output/'manifest.json')['complete']: raise RuntimeError('Cannot truncate completed manifests')
    records=[]
    for global_index in range(count):
        split='train' if global_index<cfg['train_count'] else 'val'; index=global_index if split=='train' else global_index-cfg['train_count']
        spec=repaired_spec(cfg,index,split); path=output/'annotations'/f"{spec['id']}.json"
        if path.exists():
            record=read_json(path)
            if record['spec']!=spec: raise RuntimeError('Existing versioned spec changed')
        else:
            record=render_domain(spec,output/'images'/f"{spec['id']}.png",root); write_json(path,record)
        errors=validate_domain(record,root)
        if errors: raise RuntimeError((record['id'],errors))
        records.append(record)
        if global_index<15: annotation_overlay(root,record,root/cfg['experiment']/'preflight_overlays'/f"{spec['id']}.png")
    for split in ['train','val']: write_jsonl(output/f'{split}.jsonl',[r for r in records if r['split']==split])
    result={'complete':count==total,'records':len(records),'families':dict(Counter(r['family'] for r in records)),
            'line_process':dict(Counter(r['spec'].get('line_process','not_line') for r in records)),
            'generation_source_sha256':sha256(Path(__file__)),'parent':'data/synthetic/domain_v1b',
            'policy':'Same predetermined chart indices/fonts/layouts; globally declared line process. No retries, sample filtering, changed collision threshold, or model-result-driven selection.'}
    write_json(output/'manifest.json',result); return result
