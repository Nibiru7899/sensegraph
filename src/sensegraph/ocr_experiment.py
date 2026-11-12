"""Independent ASCII crop-recognition stage; no vision/GAT/QA weight updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,time,math
from pathlib import Path
import numpy as np
import torch
from .io import read_json,write_json,sha256
from .neural_spec import CropCTCRecognizer,SenseGraphNet
from .ocr_visual import prepare,batch,ctc_loss,transcribe,recognition_metrics
from .relation_training import seed_everything


def directory(root): return root/'experiments/visual_training/ocr_v1'


def hashes(root):
    return {p:sha256(root/p) for p in ['src/sensegraph/ocr_visual.py','src/sensegraph/ocr_experiment.py','src/sensegraph/neural_spec.py','configs/ocr_visual.json','research/ocr_visual_protocol.md']}


def freeze(root,cfg):
    manifest=prepare(root); d=directory(root)
    if (d/'freeze.json').exists():
        if read_json(d/'freeze.json')['sources']!=hashes(root): raise RuntimeError('OCR frozen sources changed')
        return manifest
    write_json(d/'freeze.json',{'sources':hashes(root),'config':cfg,'data':manifest,'authorization':'User autonomous project quality work within cumulative five-hour ceiling. OCR is an already-designed visual perception branch; graph/QA/frontend stay untrained.',
            'holdout_ocr_used_for_selection':False,'primary_selection':'lowest complete validation CER, earliest exact tie','recognition_scope':'ASCII synthetic supervised crops; predicted-proposal OCR evaluated separately, no source text in inference'})
    return manifest


class OCRSession:
    def __init__(self,root,cfg): self.root=root; self.cfg=cfg; self.path=directory(root)/'gpu_budget.json'
    def __enter__(self):
        audit=read_json(self.root/'experiments/visual_training/independent_holdout/gpu_budget.json')
        if audit['actual_seconds'] is None: raise RuntimeError('Prior visual audit still active')
        if self.path.exists(): raise RuntimeError('OCR session already attempted; preserve results')
        historical=audit['historical_seconds']+audit['charged_seconds']; reserve=min(self.cfg['max_session_seconds'],18000-historical-180)
        if reserve<300: raise RuntimeError('Insufficient five-hour budget for OCR stage')
        self.reserve=reserve; self.started=time.monotonic(); self.ledger={'historical_seconds':historical,'charged_seconds':reserve,'reserved_seconds':reserve,'actual_seconds':None,'cumulative_limit_seconds':18000}; write_json(self.path,self.ledger); return self
    @property
    def remaining(self): return self.reserve-(time.monotonic()-self.started)
    def __exit__(self,*args):
        elapsed=time.monotonic()-self.started; self.ledger.update({'charged_seconds':elapsed,'actual_seconds':elapsed}); write_json(self.path,self.ledger)
        if self.ledger['historical_seconds']+elapsed>18000: raise RuntimeError('Five-hour cap exceeded')


def fresh_model(seed):
    seed_everything(seed); model=CropCTCRecognizer(); model.apply(SenseGraphNet._initialize); return model.cuda()


def update(model,opt,samples):
    model.train(); opt.zero_grad(set_to_none=True); pixels,lengths,labels,target_lengths=batch(samples)
    logits=model(pixels); loss=ctc_loss(logits,lengths,labels,target_lengths)
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite OCR CTC loss')
    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm): raise RuntimeError('Nonfinite OCR gradient')
    opt.step(); return float(loss.detach()),float(norm)


def train(root,cfg):
    freeze(root,cfg); d=directory(root)
    if (d/'attempt.json').exists(): raise RuntimeError('OCR training already attempted; no overwrite')
    if not read_json(root/'experiments/visual_training/independent_holdout/results.json')['complete']: raise RuntimeError('Finish frozen visual comparison/holdout before OCR GPU work')
    train=torch.load(root/'data/ocr_visual_v1/train.pt',weights_only=True); val=torch.load(root/'data/ocr_visual_v1/val.pt',weights_only=True)
    write_json(d/'attempt.json',{'started':time.time(),'training_crops':len(train),'validation_crops':len(val)})
    with OCRSession(root,cfg) as session:
        torch.cuda.reset_peak_memory_stats(); model=fresh_model(cfg['seed']); opt=torch.optim.AdamW(model.parameters(),lr=cfg['smoke_lr'],weight_decay=cfg['weight_decay']); fixed=train[:16]; smoke=[]; fit=None
        for step in range(cfg['smoke_max_steps']):
            loss,norm=update(model,opt,fixed); smoke.append({'step':step+1,'loss':loss,'gradient_norm':norm})
            if (step+1)%100==0:
                fit=recognition_metrics(fixed,transcribe(model,fixed))
                if fit['exact']>=.8 and fit['cer']<=.1: break
            if session.remaining<180: break
        passed=fit is not None and fit['exact']>=.8 and fit['cer']<=.1
        write_json(d/'smoke.json',{'log':smoke,'metrics':fit,'passed':passed,'scope':'Training-only GT crop memorization, not OCR generalization'})
        if not passed: return {'status':'OCR memorization failed; no full training'}
        # Reset; never initialize full OCR training from the tiny memorization model.
        del model,opt; torch.cuda.empty_cache(); model=fresh_model(cfg['seed']); opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']); history=[]; best=float('inf'); significant=float('inf'); stale=0
        for epoch in range(cfg['max_epochs']):
            model.train(); started=time.monotonic(); order=np.random.default_rng(cfg['seed']+epoch).permutation(len(train)); total_loss=0.; seen=0; max_norm=0.
            for cursor in range(0,len(train),cfg['batch_size']):
                if session.remaining<60: break
                indices=order[cursor:cursor+cfg['batch_size']]; part=[train[int(i)] for i in indices]
                factor=(cursor+len(part))/len(train) if epoch==0 else .01+.99*.5*(1+math.cos(math.pi*epoch/(cfg['max_epochs']-1)))
                opt.param_groups[0]['lr']=cfg['lr']*factor; loss,norm=update(model,opt,part); total_loss+=loss*len(part); seen+=len(part); max_norm=max(max_norm,norm)
            strings=transcribe(model,val); metrics=recognition_metrics(val,strings); complete=seen==len(train)
            row={'epoch':epoch+1,'train_loss':total_loss/max(1,seen),'train_crops':seen,'validation':metrics,'complete':complete,'seconds':time.monotonic()-started,'lr':opt.param_groups[0]['lr'],'gradient_norm_max':max_norm,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}; history.append(row)
            checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'epoch':epoch+1,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(checkpoint,d/'last.pt'); write_json(d/'history.json',history)
            if complete and metrics['cer']<best:
                best=metrics['cer']; torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch+1,'validation':metrics},d/'best.pt'); write_json(d/'validation_predictions.json',[{'chart':s['chart'],'node':s['node'],'reference':s['text'],'prediction':p} for s,p in zip(val,strings)])
            if significant-metrics['cer']>=max(1e-4,abs(significant)*.005) or not math.isfinite(significant): significant=metrics['cer']; stale=0
            else: stale+=1
            print({'epoch':epoch+1,'val_CER':metrics['cer'],'exact':metrics['exact'],'remaining_seconds':session.remaining},flush=True)
            if not complete: status='resource_limited_partial'; break
            if epoch+1>=cfg['min_epochs'] and stale>=cfg['patience']: status='early_stopped'; break
        else: status='maximum_epochs_complete'
        selected=torch.load(d/'best.pt',weights_only=True,map_location='cuda'); model.load_state_dict(selected['model']); model.eval(); sample=batch(val[:8])[0]
        with torch.inference_mode(): a=model(sample)
        model.load_state_dict(torch.load(d/'best.pt',weights_only=True,map_location='cuda')['model'])
        with torch.inference_mode(): b=model(sample)
        exact=torch.equal(a,b); assert exact
        write_json(d/'result.json',{'status':status,'selected_epoch':selected['epoch'],'validation':selected['validation'],'checkpoint_reload_exact':exact,'completed_epochs':sum(r['complete'] for r in history),'parameters':sum(p.numel() for p in model.parameters()),'holdout_used':False})
        return {'status':status,'best_val_CER':best,'selected_epoch':selected['epoch']}


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','train']); p.add_argument('--root',default='.'); args=p.parse_args(); root=Path(args.root).resolve(); cfg=read_json(root/'configs/ocr_visual.json')
    print(freeze(root,cfg) if args.command=='prepare' else train(root,cfg),flush=True)


if __name__=='__main__': main()
