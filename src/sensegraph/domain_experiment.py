"""Configuration-driven controlled domain-generalization workflow."""
from pathlib import Path
import argparse,shutil,time
from collections import Counter
import torch
from .io import read_json,read_jsonl,write_json,sha256
from .domain_data import prepare_synthetic,validate_inventory
from .domain_metrics import partial_counts,summarize_partial


def initialize(root,cfg):
    d=root/cfg['experiment']; old=root/'experiments/visual_training/release/source_and_release_hashes.json'
    if not (d/'initial.json').exists():
        mismatches=[p for p,h in read_json(old)['files'].items() if sha256(root/p)!=h]
        if mismatches: raise RuntimeError(f'Historical snapshot changed: {mismatches[:5]}')
        archive=d/'archive/PROJECT_STATUS.md'; archive.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root/'PROJECT_STATUS.md',archive)
        write_json(d/'initial.json',{'started_unix':time.time(),'prior_snapshot_sha256':sha256(old),'archive_mapping':{'PROJECT_STATUS.md':str(archive.relative_to(root))},
                'download_initial':read_json(root/'experiments/download_ledger.json')['transferred_bytes'],'prior_gpu_ledger_sha256':sha256(root/cfg['prior_gpu_ledger']),
                'prior_gpu_seconds':read_json(root/cfg['prior_gpu_ledger'])['historical_seconds']+read_json(root/cfg['prior_gpu_ledger'])['charged_seconds'],'config':cfg})
    return read_json(d/'initial.json')


def diagnose(root,cfg):
    directory=root/cfg['experiment']; predictions=torch.load(root/'experiments/visual_training/real_partial/predictions.pt',map_location='cpu',weights_only=True); cases=[]; inventory=[]
    for pred in predictions:
        record=read_json(root/f"data/real_annotations/chartqa/converted/{pred['id']}.json")
        from PIL import Image
        with Image.open(root/record['image_path']) as image: record.update({'id':pred['id'],'width':image.width,'height':image.height})
        cases.append(partial_counts(pred,record)); inventory.append(record)
    result={'metrics':summarize_partial(cases),'cases':cases,'chart_types':dict(Counter(r['chart_type'] for r in inventory)),
            'real_text_longer_than_CTC_32_positions':sum(len(n.get('text') or '')>32 for r in inventory for n in r['nodes']),
            'diagnosis':['No horizontal bars in prior synthetic training; old real diagnostic contains no vertical bars.',
            'Thin line envelopes, differing styles and partial source coverage limit box-IoU interpretation.',
            'All prior OCR holdout strings were seen in training; poor font/rotation transfer and 32-position cap are observed limitations.',
            'Real precision/AP and point/tick accuracy cannot be inferred from missing annotations.'],
            'real_training_C_D':{'eligible':False,'reason':'Partial labels lack trusted background negatives; this matched-budget pilot fits synthetic arms only.'}}
    write_json(directory/'diagnosis.json',result); return {'images':len(cases),'chart_types':result['chart_types'],'metrics':result['metrics']}


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['init','diagnose','prepare','validate','acquire','inspect','freeze','profile','fit','evaluate','diagnose-validation','report','verify']); p.add_argument('--config',default='configs/domain_generalization_v1.json'); p.add_argument('--root',default='.'); p.add_argument('--limit',type=int)
    args=p.parse_args(); torch.set_num_threads(1); root=Path(args.root).resolve(); cfg=read_json(root/args.config); initialize(root,cfg)
    if args.command=='init': result={'initialized':cfg['version']}
    elif args.command=='diagnose': result=diagnose(root,cfg)
    elif args.command=='prepare': result=prepare_synthetic(root,cfg,args.limit)
    elif args.command=='validate': result=validate_inventory(root,cfg)
    elif args.command=='acquire':
        from .domain_sources import acquire
        result=acquire(root,cfg)
    elif args.command=='inspect':
        from .domain_review import inspect
        result=inspect(root,cfg)
    else:
        from .domain_training import dispatch
        result=dispatch(root,cfg,args.command)
    print(result,flush=True)
    if isinstance(result,dict) and result.get('valid') is False: raise SystemExit(1)


if __name__=='__main__': main()
