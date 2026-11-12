"""Single-factor corrected-target experiment reusing the exact frozen v2 trainer.

Dependency bindings occur only in this isolated command process, never at import.
"""
import argparse
from pathlib import Path
from . import full_visual_training as trainer
from .collision_targets import CollisionInvariantDataset
from .io import read_json,write_json,sha256

ORIGINAL_HASHES=trainer.source_hashes


def directory(root): return root/'experiments/visual_training/v3'


def hashes(root):
    result=ORIGINAL_HASHES(root)
    for p in ['src/sensegraph/corrected_visual_training.py','src/sensegraph/collision_targets.py','configs/corrected_visual_training.json','research/corrected_visual_training_protocol.md']:
        result[p]=sha256(root/p)
    return result


def bind():
    trainer.directory=directory
    trainer.source_hashes=hashes
    trainer.VisualDataset=CollisionInvariantDataset


def freeze(root,cfg):
    d=directory(root)
    if (d/'freeze.json').exists(): return trainer.verify(root)
    previous=root/'experiments/visual_training/v2'
    ledger=read_json(previous/'gpu_budget.json')
    if any(s['actual_seconds'] is None for s in ledger['sessions']): raise RuntimeError('Prior training still active/unreconciled')
    historical=ledger['historical_seconds']+ledger['charged_seconds']
    if historical+cfg['campaign_seconds']+600>18000: raise RuntimeError('Not enough of five-hour cumulative budget;600seconds reserved for evaluation')
    audit=read_json(d/'target_fix_audit.json')
    assert audit['order_invariant_all_records'] and audit['all_regression_and_valid_masks_unchanged']
    history=dict(read_json(previous/'freeze.json')['history'])
    for p in previous.rglob('*'):
        if p.is_file(): history[p.relative_to(root).as_posix()]=sha256(p)
    payload={'sources':hashes(root),'history':history,'historical_seconds':historical,'config':cfg,
             'single_factor_change':'Union Gaussian halos of all colliding instances, instead of first-instance radius. Two train charts and one validation chart change; all boxes/counts/masks unchanged.',
             'selection_rule':'Select highest complete300-chart validation AP50 across v2/v3, exact ties prefer corrected v3, earliest epoch within each run. Holdout outcomes must not change selection.',
             'authorization':'User requested autonomous completion/audits within cumulative five GPU hours; no graph/QA/frontend scope expansion.',
             'train_sha256':sha256(root/'data/synthetic/v3/train.jsonl'),'validation_sha256':sha256(root/'data/synthetic/v3/val.jsonl'),'test_access':False}
    write_json(d/'freeze.json',payload)
    for name,digest in payload['sources'].items():
        p=d/'source_snapshots'/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((root/name).read_bytes()); assert sha256(p)==digest
    return {'historical_minutes':historical/60,'new_campaign_max_minutes':cfg['campaign_seconds']/60}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['freeze','verify','run']); parser.add_argument('--root',default='.'); args=parser.parse_args(); root=Path(args.root).resolve()
    # Verify original v2 before binding the isolated v3 namespace.
    trainer.verify(root); cfg=read_json(root/'configs/corrected_visual_training.json'); bind()
    print(freeze(root,cfg) if args.command=='freeze' else trainer.verify(root) if args.command=='verify' else trainer.run(root,cfg),flush=True)


if __name__=='__main__': main()
