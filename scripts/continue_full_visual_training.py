"""Resource-only planning-window continuation under explicit no-limit authorization.

Never modifies the frozen scientific config, old ledgers, or committed optimizer cursor.
"""
from pathlib import Path
import argparse,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph.full_visual_training import run,verify


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--campaign-minutes',type=int,required=True); args=parser.parse_args()
    d=ROOT/'experiments/visual_training/v2'; verify(ROOT)
    ledger=read_json(d/'gpu_budget.json')
    if any(s['actual_seconds'] is None for s in ledger['sessions']): raise RuntimeError('An existing session is active/unreconciled; no concurrent continuation')
    state=torch.load(d/'last.pt',map_location='cpu',weights_only=False)
    if state.get('finished'):
        print({'status':state['finished'],'continuation_not_needed':True}); return
    original=read_json(ROOT/'configs/full_visual_training.json'); cfg={**original,'campaign_seconds':args.campaign_minutes*60}
    if read_json(d/'freeze.json')['historical_seconds']+cfg['campaign_seconds']>5*3600:
        raise ValueError('Latest user authorization caps total accounted GPU sessions at five hours')
    if cfg['campaign_seconds']<=ledger['charged_seconds']+200: raise ValueError('Continuation window must allow a complete validation reserve')
    number=len(list(d.glob('resource_continuation_*.json')))+1
    write_json(d/f'resource_continuation_{number}.json',{'time_unix':time.time(),'authorization':'Latest user: continue autonomously within five-hour limit, inspect leakage and overfitting. Treat five hours as a cumulative ceiling, not a requirement to waste GPU after convergence.',
          'change':'campaign_seconds only; no model/data/loss/optimizer/selection/stopping changes','original_campaign_seconds':original['campaign_seconds'],'new_campaign_seconds':cfg['campaign_seconds'],
          'previous_ledger_sha256':sha256(d/'gpu_budget.json'),'committed_checkpoint_sha256':sha256(d/'last.pt'),
          'committed_steps_before_resume':state['optimizer_steps'],'stage':state['stage'],'epoch':state['epoch'],'cursor':state['cursor'],
          'frozen_scientific_source_sha256':read_json(d/'freeze.json')['sources'],'historical_accounting_reset':False})
    if (d/'run_status.json').exists():
        write_json(d/f'run_status_before_continuation_{number}.json',read_json(d/'run_status.json'))
    write_json(d/'run_status.json',{'status':'running_resumed','committed_steps':state['optimizer_steps']})
    print(run(ROOT,cfg),flush=True)


if __name__=='__main__': main()
