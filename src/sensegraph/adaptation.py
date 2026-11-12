"""Explicit v2 continuation; previous frozen protocols and ledgers remain intact."""
from pathlib import Path
import argparse,shutil,time,json
import torch
from .io import read_json,write_json,sha256


def initialize(root,cfg):
    d=root/cfg['experiment']
    if (d/'initial.json').exists(): return
    from .domain_training import verify
    verify(root,read_json(root/'configs/domain_generalization_v1.json'))
    inherited=read_json(root/'experiments/domain_generalization/v1/initial.json')
    archive=d/'archive'; archive.mkdir(parents=True,exist_ok=True)
    for name in ['PROJECT_STATUS.md','research/real_domain_generalization.md']:
        target=archive/Path(name).name; shutil.copyfile(root/name,target)
    prior=read_json(root/cfg['prior_gpu_ledger'])
    write_json(d/'initial.json',{'started_unix':time.time(),'config':cfg,'archive_mapping':inherited['archive_mapping'],
        'download_initial':inherited['download_initial'],'prior_gpu_ledger_sha256':sha256(root/cfg['prior_gpu_ledger']),
        'prior_gpu_seconds':prior['historical_seconds']+prior['charged_seconds'],
        'continuation_archives':{p.name:sha256(p) for p in archive.iterdir()},
        'parent_artifacts':{p.relative_to(root).as_posix():sha256(p) for p in (root/'experiments/domain_generalization/v1').rglob('*') if p.is_file()},
        'authorization':'User: Resolve blocker and complete pilot; same lifetime allowance, same5% gate; explicit versioned generator amendment'})
    for name in ['diagnosis.json','real_review.json']:
        shutil.copyfile(root/'experiments/domain_generalization/v1'/name,d/name)


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['init','prepare','validate','inspect','freeze','profile','fit','evaluate','report','verify']); p.add_argument('--config',default='configs/adaptation_v2.json'); p.add_argument('--limit',type=int)
    args=p.parse_args(); torch.set_num_threads(1); root=Path.cwd(); cfg=read_json(root/args.config); initialize(root,cfg)
    if args.command=='init': value={'initialized':True}
    elif args.command=='prepare':
        from .adaptation_data import prepare
        value=prepare(root,cfg,args.limit)
    elif args.command=='validate':
        from .domain_data import validate_inventory
        value=validate_inventory(root,cfg)
    elif args.command=='inspect':
        from .domain_review import inspect
        value=inspect(root,cfg)
    elif args.command in ['evaluate','report']:
        from . import adaptation_reporting
        value=getattr(adaptation_reporting,'evaluate_final' if args.command=='evaluate' else 'report')(root,cfg)
    else:
        from . import adaptation_training
        value=getattr(adaptation_training,args.command)(root,cfg)
    print(json.dumps(value,default=str),flush=True)
    if value.get('valid') is False or value.get('status','').startswith(('blocked','resource_partial')): raise SystemExit(1)


if __name__=='__main__': main()
