"""Versioned real visual training; no graph/reasoning optimization or test reader."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
from pathlib import Path
import time
import math
import numpy as np
from PIL import Image,ImageDraw
import torch
from .io import read_json,write_json,sha256
from .acquisition import Downloader
from .neural_spec import SenseGraphNet
from .visual_training import VisualDataset,preprocess,targets,detection_loss,decode,transform_boxes
from .perception import detection_target,detection_metrics
from .relation_training import seed_everything


def location(root): return root/'experiments/visual_training/v1'


def historical(root):
    return read_json(root/'experiments/milestone4/gpu_budget.json')['historical_seconds']+read_json(root/'experiments/milestone4/gpu_budget.json')['charged_seconds']


def acquire(root,cfg):
    path=Downloader(root).fetch(cfg['weights_url'],cfg['weights_path'])
    if not sha256(path).startswith('f37072fd'): raise ValueError('Official SHA256 prefix mismatch')
    write_json(location(root)/'weights.json',{'url':cfg['weights_url'],'sha256':sha256(path),'bytes':path.stat().st_size})
    return {'path':str(path)}


def source_hashes(root):
    return {p:sha256(root/p) for p in ['src/sensegraph/visual_training.py','src/sensegraph/visual_experiment.py','src/sensegraph/neural_spec.py','configs/visual_training.json']}


def verify(root):
    frozen=read_json(location(root)/'freeze.json')
    if frozen['sources']!=source_hashes(root): raise ValueError('Frozen visual training implementation changed')
    changed=[p for p,h in frozen['history'].items() if sha256(root/p)!=h]
    if changed: raise ValueError(f'Historical files changed: {changed[:3]}')
    return {'protected_files':len(frozen['history']),'changed':[]}


def prepare(root,cfg):
    d=location(root)
    if (d/'freeze.json').exists(): return verify(root)
    history=read_json(root/'experiments/milestone4/freeze.json')['history']
    for folder in ['src','tests','research','configs']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ['visual_training.py','visual_experiment.py','test_visual_training.py','visual_training.json','perception_training_report.md','visual_training_protocol.md']:
                history[p.relative_to(root).as_posix()]=sha256(p)
    statistics={}
    for split in ['train','val']:
        dataset=VisualDataset(root,split,cfg['size']); counts=Counter(); collisions=Counter(); invalid=unsupported=0
        for i,r in enumerate(dataset.records):
            scale=min(cfg['size']/r['width'],cfg['size']/r['height']); nw,nh=round(r['width']*scale),round(r['height']*scale)
            affine=torch.tensor([nw/r['width'],nh/r['height'],(cfg['size']-nw)//2,(cfg['size']-nh)//2,r['width'],r['height']])
            target=targets(r,affine,cfg['size']); counts.update(target['counts']); collisions.update(target['collisions']); invalid+=target['invalid']; unsupported+=target['unsupported']
        statistics[split]={'images':len(dataset),'counts':dict(counts),'collision_objects':dict(collisions),'invalid':invalid,'unsupported':unsupported,
                           'collision_fraction':{k:collisions[k]/v for k,v in counts.items()}}
    write_json(d/'data_statistics.json',statistics)
    write_json(d/'freeze.json',{'sources':source_hashes(root),'history':history,'config':cfg,'historical_gpu_seconds':historical(root),
                               'train_sha256':sha256(root/'data/synthetic/v3/train.jsonl'),'val_sha256':sha256(root/'data/synthetic/v3/val.jsonl'),
                               'download_bytes_at_freeze':read_json(root/'experiments/download_ledger.json')['transferred_bytes'],'test_access':False})
    # Visual target center overlay in native pixels; never a predicted-box claim.
    dataset=VisualDataset(root,'train',cfg['size'])
    for i in range(3):
        image,target,affine,r=dataset[i]; canvas=Image.open(root/r['image_path']).convert('RGB'); draw=ImageDraw.Draw(canvas)
        for branch in ['box','key']:
            c,y,x=torch.where(target[branch+'_heat']==1)
            for ci,yi,xi in zip(c,y,x):
                off=target[branch+'_offset'][2*ci:2*ci+2,yi,xi]; point=(torch.stack((xi,yi))+off)*4
                sx,sy,l,t=affine[:4]; px,py=((point-torch.stack((l,t)))/torch.stack((sx,sy))).tolist()
                draw.ellipse((px-2,py-2,px+2,py+2),outline='red' if branch=='box' else 'blue',width=2)
        canvas.save(d/f'targets_{i}.png')
    return statistics


class Session:
    def __init__(self,root,seconds,cfg):
        self.path=location(root)/'gpu_budget.json'; self.cfg=cfg
        self.ledger=read_json(self.path) if self.path.exists() else {'historical_seconds':historical(root),'charged_seconds':0.,'sessions':[]}
        self.seconds=seconds
        if historical(root)+self.ledger['charged_seconds']+seconds>cfg['gpu_limit_seconds']: raise RuntimeError('Insufficient authorized GPU budget')
    def __enter__(self):
        self.start=time.monotonic(); self.entry={'reserved_seconds':self.seconds,'actual_seconds':None}; self.ledger['sessions'].append(self.entry); self.ledger['charged_seconds']+=self.seconds; write_json(self.path,self.ledger); return self
    @property
    def remaining(self): return self.seconds-(time.monotonic()-self.start)
    def __exit__(self,*args):
        elapsed=time.monotonic()-self.start; self.entry['actual_seconds']=elapsed; self.ledger['charged_seconds']+=elapsed-self.seconds; write_json(self.path,self.ledger)


def load_model(root,cfg):
    weights=root/cfg['weights_path']; provenance=read_json(location(root)/'weights.json')
    if sha256(weights)!=provenance['sha256'] or not provenance['sha256'].startswith('f37072fd'): raise ValueError('Missing or changed official ResNet18 weights')
    model=SenseGraphNet(); state=torch.load(weights,weights_only=True,map_location='cpu'); state.pop('fc.weight'); state.pop('fc.bias')
    model.backbone.load_state_dict(state,strict=True)
    return model.cuda()


def optimizer(model,cfg,stage):
    model.set_stage(stage)
    backbone=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    other=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    groups=[{'params':other,'lr':cfg['lr'],'base_lr':cfg['lr']}]
    if backbone: groups.append({'params':backbone,'lr':cfg['backbone_lr'],'base_lr':cfg['backbone_lr']})
    return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])


def overlay(root,record,pred,path):
    image=Image.open(root/record['image_path']).convert('RGB'); draw=ImageDraw.Draw(image)
    for box,score,label in zip(pred['boxes'],pred['scores'],pred['labels']):
        if score>=.25:
            b=box.tolist(); draw.rectangle(b,outline='red',width=1); draw.text(b[:2],f'{int(label)} {float(score):.2f}',fill='black')
    image.save(path)


def evaluate(model,dataset,cfg,session=None):
    model.eval(); predictions=[]; references=[]; loss_values=[]; errors=[]; key_stats={k:{'gt':0,'matched':0,'distances':[]} for k in [2,7]}
    for stats in key_stats.values(): stats['predicted']=0
    start=time.monotonic()
    with torch.inference_mode():
        for i in range(len(dataset)):
            if session and session.remaining<5: break
            image,target,affine,r=dataset[i]
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']): _,output=model.encode_image(image[None].cuda())
            loss,_=detection_loss(output,target); loss_values.append(float(loss)); pred=decode(output,affine,cfg)
            reference,ids,rejected=detection_target(r); predictions.append({k:pred[k] for k in ['boxes','labels','scores']}); references.append(reference)
            errors.append({'id':r['id'],'rejected':rejected})
            # Class-specific one-to-one keypoint recovery within 5 native-image pixels.
            for label in [2,7]:
                truth=reference['boxes'][reference['labels']==label]; centers=(truth[:,:2]+truth[:,2:])/2
                selected=pred['centers'][(pred['labels']==label)&(pred['scores']>=cfg['score_threshold'])]; used=set(); distances=[]
                for point in selected:
                    choices=[j for j in range(len(centers)) if j not in used]
                    if not choices: continue
                    j=min(choices,key=lambda j:float((point-centers[j]).norm())); distance=float((point-centers[j]).norm())
                    if distance<=5: used.add(j); distances.append(distance)
                stats=key_stats[label]; stats['gt']+=len(centers); stats['matched']+=len(used); stats['predicted']+=len(selected); stats['distances']+=distances
    for s in key_stats.values():
        s['recall']=s['matched']/max(1,s['gt']); s['precision']=s['matched']/max(1,s['predicted']); s['fp']=s['predicted']-s['matched']; s['fn']=s['gt']-s['matched']; s['mean_localization_pixels']=float(np.mean(s['distances'])) if s['distances'] else None
    return {'images':len(predictions),'loss':float(np.mean(loss_values)),'ap50':detection_metrics(predictions,references,.5,cfg['score_threshold']),
            'ap75':detection_metrics(predictions,references,.75,cfg['score_threshold']),'keypoints_5px':key_stats,
            'seconds':time.monotonic()-start,'errors':errors},predictions


def smoke(root,cfg):
    verify(root); d=location(root)
    if (d/'smoke_attempt.json').exists(): raise ValueError('Smoke already attempted; preserve it')
    # Fail before GPU allocation if required pretrained weights are unavailable.
    if not (d/'weights.json').exists(): raise FileNotFoundError('Run acquire; random-backbone fallback is forbidden')
    write_json(d/'smoke_attempt.json',{'started':time.time()}); seed_everything(cfg['seed'])
    dataset=VisualDataset(root,'train',cfg['size'])
    with Session(root,600,cfg) as session:
        torch.cuda.reset_peak_memory_stats(); model=load_model(root,cfg); opt=optimizer(model,cfg,'warmup')
        scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp']); samples=[dataset[i] for i in range(4)]; logs=[]
        for step in range(cfg['smoke_steps']):
            if session.remaining<60: break
            image,target,_,_=samples[0 if step<100 else step%4]; model.train(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); before=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']): _,output=model.encode_image(image[None].cuda())
            loss,parts=detection_loss(output,target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradients')
            scaler.step(opt); scaler.update(); torch.cuda.synchronize()
            logs.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds':time.perf_counter()-before})
        result,pred=evaluate(model,torch.utils.data.Subset(dataset,range(4)),cfg,session)
        model.eval(); image,target,affine,r=samples[0]
        with torch.inference_mode(): before=model.encode_image(image[None].cuda())[1]
        torch.save({'model':model.state_dict(),'config':cfg,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'kind':'training-only overfit diagnostic'},d/'smoke.pt')
        model.load_state_dict(torch.load(d/'smoke.pt',weights_only=True,map_location='cuda')['model'])
        with torch.inference_mode(): after=model.encode_image(image[None].cuda())[1]
        exact=all(torch.equal(before[k],after[k]) for k in before)
        for i,p in enumerate(pred): overlay(root,samples[i][3],p,d/f'smoke_{i}.png')
        for m in model.backbone.modules():
            if isinstance(m,torch.nn.BatchNorm2d): assert not m.training
        timings=[x['seconds'] for x in logs[5:]]
        seconds=float(np.median(timings))
        # Warm-up profile measured; fine-tuning estimate conservatively 2x, not measured.
        estimate=1.25*(5*(1400*seconds+300*result['seconds']/4)+30*(1400*seconds*2+300*result['seconds']/4))
        result.update({'log':logs,'checkpoint_reload_exact':exact,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                       'median_train_batch_seconds':seconds,'campaign_estimated_seconds':estimate,'estimate_method':'5 warmup epochs measured; 30 fine-tune epochs estimated at 2x warmup batch time; validation measured on four training images; 25% overhead',
                       'passed':len(logs)==cfg['smoke_steps'] and exact and np.mean([v['loss'] for v in logs[-20:]])<np.mean([v['loss'] for v in logs[:20]])*.5 and result['ap50']['macro_recall']>=.5})
        write_json(d/'smoke.json',result)
        return {'smoke_passed':result['passed'],'campaign_estimated_minutes':estimate/60}


def train(root,cfg):
    verify(root); d=location(root); smoke_result=read_json(d/'smoke.json')
    if not smoke_result['passed'] or not (d/'visual_approval.json').exists(): raise ValueError('Passed tiny-set learning and actual visual review required')
    stats=read_json(d/'data_statistics.json')
    if any(v>cfg['collision_limit'] for s in stats.values() for v in s['collision_fraction'].values()): raise ValueError('Target center collision gate failed; new target protocol required')
    ledger=read_json(d/'gpu_budget.json'); available=cfg['gpu_limit_seconds']-historical(root)-ledger['charged_seconds']
    if smoke_result['campaign_estimated_seconds']>available: raise RuntimeError('Campaign projection exceeds remaining authorized GPU time. Budget approval/new frozen limit required; no full training started.')
    if (d/'train_attempt.json').exists(): raise ValueError('Campaign already attempted; use explicit checkpoint continuation, do not overwrite')
    write_json(d/'train_attempt.json',{'started':time.time()}); seed_everything(cfg['seed']); train_data=VisualDataset(root,'train',cfg['size']); val=VisualDataset(root,'val',cfg['size'])
    with Session(root,available,cfg) as session:
        model=load_model(root,cfg); history=[]; best=-1.; stale=0
        for stage,epochs in [('warmup',cfg['warmup_epochs']),('perception',cfg['finetune_epochs'])]:
            opt=optimizer(model,cfg,stage); scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp']); best_stage=-1.; stale=0
            for epoch in range(epochs):
                torch.cuda.reset_peak_memory_stats(); model.train(); order=np.random.default_rng(cfg['seed']+len(history)).permutation(len(train_data)); losses=[]; norms=[]; start=time.monotonic()
                for group in opt.param_groups:
                    base=group['base_lr']
                    group['lr']=base*(1. if epoch==0 else .01+.99*.5*(1+math.cos(math.pi*epoch/max(1,epochs-1))))
                opt.zero_grad(set_to_none=True)
                for j,index in enumerate(order):
                    if session.remaining<60: break
                    if epoch==0:
                        for group in opt.param_groups: group['lr']=group['base_lr']*(j+1)/len(order)
                    image,target,_,_=train_data[int(index)]; divisor=min(cfg['accumulation'],len(order)-(j//cfg['accumulation'])*cfg['accumulation'])
                    with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']): _,output=model.encode_image(image[None].cuda())
                    loss,_=detection_loss(output,target)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    scaler.scale(loss/divisor).backward(); losses.append(float(loss.detach()))
                    if (j+1)%cfg['accumulation']==0 or j+1==len(order):
                        scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
                        if not torch.isfinite(norm): raise RuntimeError('Nonfinite training gradients')
                        norms.append(float(norm)); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                metrics,pred=evaluate(model,val,cfg,session); score=metrics['ap50']['macro_ap']; complete=len(losses)==len(train_data) and metrics['images']==len(val)
                row={'stage':stage,'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation':metrics,'complete':complete,'gradient_norm_max':max(norms,default=0),'lr':[g['lr'] for g in opt.param_groups],'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}; history.append(row)
                checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'config':cfg,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
                torch.save(checkpoint,d/f'{stage}_epoch_{epoch+1}.pt'); write_json(d/'history.json',history)
                for i in range(min(3,len(pred))): overlay(root,val.records[i],pred[i],d/f'{stage}_{epoch+1}_val_{i}.png')
                if score>best and complete: best=score; torch.save(checkpoint,d/'best.pt')
                if score>=best_stage*1.005+1e-4: best_stage=score; stale=0
                else: stale+=1
                if not complete: return {'status':'budget-limited partial campaign','epochs':len(history)}
                if stage=='perception' and epoch>=4 and stale>=5: break
        return {'status':'curriculum_finished','epochs':len(history),'best_validation_ap50':best}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['acquire','prepare','verify','smoke','train']); parser.add_argument('--root',default='.')
    args=parser.parse_args(); root=Path(args.root).resolve(); cfg=read_json(root/'configs/visual_training.json')
    function={'acquire':acquire,'prepare':prepare,'smoke':smoke,'train':train}
    print(verify(root) if args.command=='verify' else function[args.command](root,cfg),flush=True)


if __name__=='__main__': main()
