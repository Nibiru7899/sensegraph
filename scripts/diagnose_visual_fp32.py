"""Explicit FP32 investigation after v1 AMP gradient failure; not a curriculum refit."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph.visual_experiment import Session,load_model,optimizer,evaluate,overlay,verify,location
from sensegraph.visual_training import VisualDataset,detection_loss
from sensegraph.relation_training import seed_everything


def main():
    verify(ROOT); cfg=read_json(ROOT/'configs/visual_training.json'); cfg={**cfg,'amp':False}
    d=location(ROOT)/'fp32_diagnostic'
    if (d/'attempt.json').exists(): raise ValueError('FP32 attempt already exists; do not overwrite')
    write_json(d/'attempt.json',{'reason':'v1 first AMP backward produced nonfinite gradient before optimizer step; explicit FP32 diagnostic only',
               'config':cfg,'script_sha256':sha256(Path(__file__)),'stage_order':'100 single-chart then300 four-chart steps;20 C4/C5 profile updates',
               'ledger':'experiments/visual_training/v1/gpu_budget.json','historical_attempt_preserved':True})
    seed_everything(cfg['seed']); dataset=VisualDataset(ROOT,'train',cfg['size']); samples=[dataset[i] for i in range(4)]
    with Session(ROOT,600,cfg) as session:
        torch.cuda.reset_peak_memory_stats(); model=load_model(ROOT,cfg); opt=optimizer(model,cfg,'warmup'); log=[]
        for step in range(cfg['smoke_steps']):
            if session.remaining<90: break
            image,target,_,_=samples[0 if step<100 else step%4]; model.train(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); started=time.perf_counter()
            _,output=model.encode_image(image[None].cuda()); loss,parts=detection_loss(output,target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite FP32 loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite FP32 gradient')
            opt.step(); torch.cuda.synchronize()
            log.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds':time.perf_counter()-started,
                        'parts':{k:float(v.detach()) for k,v in parts.items()}})
            if (step+1)%20==0:
                write_json(d/'training_log.json',log); print(f"step {step+1} loss {log[-1]['loss']:.4f}",flush=True)
        metrics,pred=evaluate(model,torch.utils.data.Subset(dataset,range(4)),cfg,session)
        model.eval(); image=samples[0][0][None].cuda()
        with torch.inference_mode(): before=model.encode_image(image)[1]
        torch.save({'model':model.state_dict(),'config':cfg,'steps':len(log),'optimizer':opt.state_dict(),'kind':'FP32 tiny training-set diagnostic, not full split training'},d/'checkpoint.pt')
        model.load_state_dict(torch.load(d/'checkpoint.pt',map_location='cuda',weights_only=True)['model'])
        with torch.inference_mode(): after=model.encode_image(image)[1]
        exact=all(torch.equal(before[k],after[k]) for k in before)
        for i,p in enumerate(pred): overlay(ROOT,samples[i][3],p,d/f'prediction_{i}.png')
        # Independently time20 genuine fine-tuning updates, retaining original tiny checkpoint.
        fine=[]; opt=optimizer(model,cfg,'perception')
        for i in range(cfg['profile_batches']):
            if session.remaining<15: break
            image,target,_,_=dataset[4+i]; model.train(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); started=time.perf_counter()
            _,output=model.encode_image(image[None].cuda()); loss,_=detection_loss(output,target); loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
            if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError('Nonfinite FP32 fine-tune profile')
            opt.step(); torch.cuda.synchronize(); fine.append(time.perf_counter()-started)
        warm=float(np.median([r['seconds'] for r in log[-20:]])); fine_time=float(np.median(fine)) if fine else None
        estimate=1.25*(5*(1400*warm+300*metrics['seconds']/4)+30*(1400*fine_time+300*metrics['seconds']/4)) if fine_time else None
        metrics.update({'steps':len(log),'precision':'float32','checkpoint_reload_exact':exact,'warmup_batch_seconds':warm,'finetune_batch_seconds':fine_time,
                       'fine_profile_optimizer_updates':len(fine),'campaign_estimated_seconds':estimate,
                       'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                       'loss_first20':float(np.mean([r['loss'] for r in log[:20]])),'loss_last20':float(np.mean([r['loss'] for r in log[-20:]])),
                       'passed':bool(len(log)==cfg['smoke_steps'] and exact and metrics['ap50']['macro_recall']>=.5 and np.mean([r['loss'] for r in log[-20:]])<np.mean([r['loss'] for r in log[:20]])*.5)})
        write_json(d/'result.json',metrics); write_json(d/'training_log.json',log)
        print({'passed':metrics['passed'],'macro_recall':metrics['ap50']['macro_recall'],'estimated_campaign_minutes':estimate/60 if estimate else None},flush=True)


if __name__=='__main__': main()
