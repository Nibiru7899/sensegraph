"""Sequential audit-session accounting under the cumulative five-hour ceiling."""
from .io import read_json,write_json,sha256
from .suspend_accounting import WorkingState,working_seconds


class QualitySession:
    def __init__(self,path,prior,max_seconds,min_seconds=120):
        self.path=path; self.prior=prior; self.max_seconds=max_seconds; self.min_seconds=min_seconds
    def __enter__(self):
        if self.path.exists(): raise RuntimeError('Quality session already attempted; preserve its accounting')
        old=read_json(self.prior)
        if 'sessions' in old:
            if any(s['actual_seconds'] is None for s in old['sessions']): raise RuntimeError('Prior session active/unreconciled')
        elif old.get('actual_seconds') is None: raise RuntimeError('Prior session active/unreconciled')
        historical=old['historical_seconds']+old['charged_seconds']; self.reserve=min(self.max_seconds,18000-historical)
        if self.reserve<self.min_seconds: raise RuntimeError('Insufficient cumulative five-hour budget')
        self.power=WorkingState(); self.power.__enter__(); self.start=working_seconds(); self.ledger={'historical_seconds':historical,'charged_seconds':self.reserve,'reserved_seconds':self.reserve,'actual_seconds':None,'prior_ledger':str(self.prior),'prior_sha256':sha256(self.prior),'ceiling_seconds':18000,'clock':'Windows working-state seconds, excludes sleep'}
        write_json(self.path,self.ledger); return self
    @property
    def remaining(self): return self.reserve-(working_seconds()-self.start)
    def __exit__(self,*args):
        elapsed=working_seconds()-self.start; self.ledger.update({'charged_seconds':elapsed,'actual_seconds':elapsed}); write_json(self.path,self.ledger); self.power.__exit__(*args)
        if self.ledger['historical_seconds']+elapsed>18000: raise RuntimeError('Cumulative five-hour ceiling exceeded')
