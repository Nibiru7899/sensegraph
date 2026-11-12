"""Resource-only continuation of exact committed v3 optimizer/RNG/cursor."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph import full_visual_training as trainer
from sensegraph.corrected_visual_training import bind,directory
from sensegraph.io import read_json,write_json,sha256
from sensegraph.suspend_accounting import verified_suspend_correction,working_seconds,WorkingState


class AwakeCampaignSession:
    def __init__(self,root,cfg):
        self.path=directory(root)/'gpu_budget_awake.json'; self.cfg=cfg; raw=directory(root)/'gpu_budget.json'; evidence=directory(root)/'power_events.json'; audit=directory(root)/'suspend_reconciliation.json'
        if audit.exists():
            correction=read_json(audit)
            assert correction['original_ledger_sha256']==sha256(raw) and correction['power_events_sha256']==sha256(evidence)
        else:
            old=read_json(raw); correction={**verified_suspend_correction(old,read_json(evidence)),
                'original_ledger_sha256':sha256(raw),'power_events_sha256':sha256(evidence),'authorization':'Continue user-authorized five-hour GPU-work ceiling; exclude only independently verified non-execution suspension, not a budget reset',
                'resource_only':True,'scientific_sources_unchanged':trainer.source_hashes(root),'driver_sha256':sha256(Path(__file__))}
            write_json(audit,correction)
        old=read_json(raw)
        self.ledger=read_json(self.path) if self.path.exists() else {'historical_seconds':old['historical_seconds'],'charged_seconds':correction['reconciled_session_seconds'],'campaign_authorization_seconds':cfg['campaign_seconds'],
            'reconciliation_sha256':sha256(audit),'original_wall_ledger_preserved':True,'sessions':[{'actual_seconds':correction['reconciled_session_seconds'],'start_unix':old['sessions'][0]['start_unix'],'accounting':'Original session less independently verified sleep interval'}]}
        assert self.ledger['reconciliation_sha256']==sha256(audit)
        if any(s['actual_seconds'] is None for s in self.ledger['sessions']): raise RuntimeError('Prior working-time session unresolved')
        self.reserve=cfg['campaign_seconds']-self.ledger['charged_seconds']
        if self.reserve<200 or self.ledger['historical_seconds']+self.ledger['charged_seconds']+self.reserve+600>18000: raise RuntimeError('Insufficient cumulative GPU budget')
    def __enter__(self):
        self.power=WorkingState(); self.power.__enter__(); self.started=working_seconds(); self.wall=time.monotonic()
        self.entry={'reserved_seconds':self.reserve,'actual_seconds':None,'start_unix':time.time(),'clock':'QueryUnbiasedInterruptTime excludes sleep; includes all awake CPU/loading/evaluation overhead'}
        self.ledger['sessions'].append(self.entry); self.ledger['charged_seconds']+=self.reserve; write_json(self.path,self.ledger); return self
    @property
    def remaining(self): return self.reserve-(working_seconds()-self.started)
    def __exit__(self,*args):
        try:
            elapsed=working_seconds()-self.started; self.entry.update({'actual_seconds':elapsed,'wall_seconds':time.monotonic()-self.wall}); self.ledger['charged_seconds']+=elapsed-self.reserve; write_json(self.path,self.ledger)
        finally: self.power.__exit__(*args)


def main():
    trainer.verify(ROOT); bind(); trainer.verify(ROOT); trainer.CampaignSession=AwakeCampaignSession
    write_json(directory(ROOT)/'run_status_before_awake_resume.json',read_json(directory(ROOT)/'run_status.json'))
    print(trainer.run(ROOT,read_json(ROOT/'configs/corrected_visual_training.json')),flush=True)


if __name__=='__main__': main()
