"""Recover metrics after JSON numpy-bool export failure; never rerun tiny-set fitting."""
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json
from sensegraph.visual_experiment import Session,load_model,optimizer,evaluate,location
from sensegraph.visual_training import VisualDataset,detection_loss
from sensegraph.relation_training import seed_everything


def main():
    d=location(ROOT)/'fp32_diagnostic'; cfg={**read_json(ROOT/'configs/visual_training.json'),'amp':False}; seed_everything(cfg['seed'])
    if (d/'recovery_attempt.json').exists(): raise ValueError('Recovery already attempted')
    write_json(d/'recovery_attempt.json',{'reason':'400 optimizer steps and checkpoint completed; final export rejected numpy bool. Recover evaluation from saved weights; no tiny-set refit. Earlier20 fine-tune timing updates were discarded and are repeated solely for timing.'})
    with Session(ROOT,180,cfg) as session:
        torch.cuda.reset_peak_memory_stats(); model=load_model(ROOT,cfg); model.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True,map_location='cuda')['model'])
        dataset=VisualDataset(ROOT,'train',cfg['size']); metrics,pred=evaluate(model,torch.utils.data.Subset(dataset,range(4)),cfg,session)
        torch.save(pred,d/'predictions.pt'); model.eval(); image=dataset[0][0][None].cuda()
        with torch.inference_mode(): a=model.encode_image(image)[1]
        model.load_state_dict(torch.load(d/'checkpoint.pt',weights_only=True,map_location='cuda')['model'])
        with torch.inference_mode(): b=model.encode_image(image)[1]
        exact=all(torch.equal(a[k],b[k]) for k in a)
        fine=[]; opt=optimizer(model,cfg,'perception')
        for i in range(cfg['profile_batches']):
            image,target,_,_=dataset[4+i]; model.train(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); start=time.perf_counter()
            _,output=model.encode_image(image[None].cuda()); loss,_=detection_loss(output,target); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError('Nonfinite FP32 profile')
            opt.step(); torch.cuda.synchronize(); fine.append(time.perf_counter()-start)
        log=read_json(d/'training_log.json'); warm=float(np.median([v['seconds'] for v in log[-20:]])); fine_time=float(np.median(fine))
        estimated=1.25*(5*(1400*warm+300*metrics['seconds']/4)+30*(1400*fine_time+300*metrics['seconds']/4))
        metrics.update({'steps':len(log),'precision':'float32','checkpoint_reload_exact':exact,'warmup_batch_seconds':warm,'finetune_batch_seconds':fine_time,
                        'campaign_estimated_seconds':estimated,'fine_profile_optimizer_updates':20,'earlier_discarded_profile_updates':20,
                        'peak_allocated_bytes_recovery_profile':torch.cuda.max_memory_allocated(),'peak_reserved_bytes_recovery_profile':torch.cuda.max_memory_reserved(),
                        'loss_first20':float(np.mean([r['loss'] for r in log[:20]])),'loss_last20':float(np.mean([r['loss'] for r in log[-20:]])),
                        'passed':bool(len(log)==400 and exact and metrics['ap50']['macro_recall']>=.5 and np.mean([r['loss'] for r in log[-20:]])<np.mean([r['loss'] for r in log[:20]])*.5)})
        write_json(d/'result.json',metrics)
        print({'smoke_passed':metrics['passed'],'macro_recall':metrics['ap50']['macro_recall'],'projected_campaign_minutes':estimated/60})


if __name__=='__main__': main()
