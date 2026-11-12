from pathlib import Path
import copy
import numpy as np
import torch
from sensegraph.io import read_json
from sensegraph.adaptation_data import repaired_spec
from sensegraph.domain_data import make_domain_spec,render_domain,validate_domain
from sensegraph.domain_metrics import select_checkpoint,admission,bootstrap_real,partial_counts

CFG=read_json(Path(__file__).resolve().parents[1]/'configs/adaptation_v2.json')


def test_repair_preserves_inventory_and_does_not_reroll():
    modes=[]
    for i in range(100):
        old=make_domain_spec(CFG,i,'train'); new=repaired_spec(CFG,i,'train')
        assert new==repaired_spec(CFG,i,'train')
        for key in ['style','resolution','title','family']: assert new[key]==old[key]
        assert np.asarray(new['data']['values']).shape==np.asarray(old['data']['values']).shape
        if new['chart_type']!='line': assert new['data']==old['data']
        else:
            assert new['data']['values']!=old['data']['values']; modes.append(new['line_process'])
    assert modes.count('crossing')==5 and modes.count('common_trend')==15


def test_repaired_lines_have_exact_metadata(tmp_path):
    for i in [4,9]:
        r=render_domain(repaired_spec(CFG,i,'train'),tmp_path/f'{i}.png',tmp_path)
        assert validate_domain(r,tmp_path)==[]


def test_paired_bootstrap_is_deterministic_and_null_pair_zero():
    from sensegraph.adaptation_reporting import bootstrap_counts
    p={'boxes':torch.tensor([[1.,1.,5.,5.]]),'scores':torch.tensor([.9]),'labels':torch.tensor([1])}
    r={'id':'a','chart_type':'h_bar','width':10,'height':10,'nodes':[{'id':'b','type':'bar','bbox':[1,1,5,5]}]}
    cases=[partial_counts(p,r)]
    a=bootstrap_real(cases,cases,50,7)
    assert a==bootstrap_real(cases,cases,50,7) and a['ci95']==[0.,0.] and a['difference']==0
    b=bootstrap_counts(cases,cases,50,7)
    assert b['ci95']==a['ci95'] and b['difference']==a['difference']
