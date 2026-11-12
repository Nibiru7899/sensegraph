"""FP32 full-split v2 curriculum with committed optimizer-step resumption."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,math,time,random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import batched_nms
from .io import read_json,write_json,sha256
from .visual_training import VisualDataset,detection_loss,transform_boxes,BOX_TYPES,KEY_TYPES
from .visual_experiment import optimizer,load_model,overlay
from .perception import detection_target,detection_metrics
from .relation_training import seed_everything


def directory(root): return root/'experiments/visual_training/v2'


@torch.no_grad()
def decode_maps(output,affine,cfg):
    """Vectorized equivalent of the v1 decoder; same filtering and cap order."""
    device=output['center'].device; a=affine.to(device); sx,sy,l,t,w,h=a
    boxes=[]; scores=[]; labels=[]; centers=[]
    for names,hm,sz,off in [(BOX_TYPES,'center','size','offset'),(KEY_TYPES,'keypoint','key_extent','key_offset')]:
        heat=output[hm][0].float().sigmoid(); c,y,x=torch.where((heat==F.max_pool2d(heat[None],3,1,1)[0])&(heat>=cfg['decode_threshold']))
        if len(c)==0: continue
        channel=2*c[:,None]+torch.arange(2,device=device)[None]
        wh=4*F.softplus(output[sz][0,channel,y[:,None],x[:,None]].float())
        center=4*(torch.stack((x,y),1).float()+output[off][0,channel,y[:,None],x[:,None]].float().sigmoid())
        box=torch.cat((center-wh/2,center+wh/2),1); box[:,0::2].clamp_(l,l+w*sx); box[:,1::2].clamp_(t,t+h*sy)
        valid=(center[:,0]>=l)&(center[:,0]<l+w*sx)&(center[:,1]>=t)&(center[:,1]<t+h*sy)&(box[:,2:]>box[:,:2]).all(1)
        boxes.append(box[valid]); centers.append(center[valid]); scores.append(heat[c,y,x][valid]); labels.append(torch.tensor(names,device=device)[c[valid]])
    if not boxes or sum(len(b) for b in boxes)==0:
        return {'boxes':torch.empty(0,4),'scores':torch.empty(0),'labels':torch.empty(0,dtype=torch.long),'centers':torch.empty(0,2)}
    boxes=torch.cat(boxes); centers=torch.cat(centers); scores=torch.cat(scores); labels=torch.cat(labels)
    order=torch.argsort(scores,descending=True,stable=True)[:cfg['max_proposals']]; keep=order[batched_nms(boxes[order],scores[order],labels[order],cfg['nms_iou'])]
    native=transform_boxes(boxes[keep],a,True); native[:,0::2].clamp_(0,w); native[:,1::2].clamp_(0,h)
    return {'boxes':native.cpu(),'scores':scores[keep].cpu(),'labels':labels[keep].cpu(),'centers':((centers[keep]-torch.stack((l,t)))/torch.stack((sx,sy))).cpu()}


def keypoint_metrics(predictions,references):
    result={}
    for label in [2,7]:
        gt=tp=predicted=0; errors=[]
        for p,r in zip(predictions,references):
            boxes=r['boxes'][r['labels']==label]; centers=(boxes[:,:2]+boxes[:,2:])/2; gt+=len(centers)
            selected=p['centers'][(p['labels']==label)&(p['scores']>=.25)]; predicted+=len(selected); used=set()
            for point in selected:
                choices=[j for j in range(len(centers)) if j not in used]
                if not choices: break
                j=min(choices,key=lambda j:float((point-centers[j]).norm())); distance=float((point-centers[j]).norm())
                if distance<=5: used.add(j); errors.append(distance)
            tp+=len(used)
        result[str(label)]={'support':gt,'tp':tp,'fp':predicted-tp,'fn':gt-tp,'precision':tp/max(1,predicted),'recall':tp/max(1,gt),'mean_error_pixels':float(np.mean(errors)) if errors else None}
    return result


def evaluate(model,data,cfg,output_dir,label,save_predictions=True):
    model.eval(); predictions=[]; references=[]; losses=[]; confidences=[]; started=time.monotonic()
    with torch.inference_mode():
        for i in range(len(data)):
            image,target,affine,record=data[i]; _,dense=model.encode_image(image[None].cuda()); loss,_=detection_loss(dense,target)
            losses.append(float(loss)); pred=decode_maps(dense,affine,cfg); predictions.append(pred); references.append(detection_target(record)[0]); confidences.extend(pred['scores'].tolist())
            if i<3: overlay(data.root,record,pred,output_dir/f'{label}_overlay_{i}.png')
    metrics={'images':len(predictions),'loss':float(np.mean(losses)),'ap50':detection_metrics(predictions,references,.5,.25),'ap75':detection_metrics(predictions,references,.75,.25),
             'keypoints_5px':keypoint_metrics(predictions,references),'seconds':time.monotonic()-started,
             'confidence_quantiles':np.quantile(confidences,[0,.25,.5,.75,1]).tolist() if confidences else [],'chart_types':{}}
    for chart_type in ['bar','grouped_bar','line']:
        ids=[i for i,r in enumerate(data.records) if r['chart_type']==chart_type]
        metrics['chart_types'][chart_type]=detection_metrics([predictions[i] for i in ids],[references[i] for i in ids],.5,.25)
    if len(predictions)>=30:
        metrics['aligned_first30']=detection_metrics(predictions[:30],references[:30],.5,.25)
    write_json(output_dir/f'{label}_metrics.json',metrics)
    if save_predictions: torch.save({'predictions':predictions,'references':references,'ids':[r['id'] for r in data.records]},output_dir/f'{label}_predictions.pt')
    return metrics


def source_hashes(root):
    names=['src/sensegraph/full_visual_training.py','src/sensegraph/visual_training.py','src/sensegraph/neural_spec.py','src/sensegraph/visual_experiment.py','configs/full_visual_training.json','research/full_visual_training_protocol.md']
    return {p:sha256(root/p) for p in names}


def verify(root):
    frozen=read_json(directory(root)/'freeze.json')
    if source_hashes(root)!=frozen['sources']: raise RuntimeError('v2 frozen source changed')
    changed=[p for p,h in frozen['history'].items() if sha256(root/p)!=h]
    if changed: raise RuntimeError(f'Historical artifacts changed: {changed[:3]}')
    return {'protected_files':len(frozen['history']),'changed':[]}


def freeze(root,cfg):
    d=directory(root)
    if (d/'freeze.json').exists(): return verify(root)
    from .visual_experiment import verify as verify_v1
    verify_v1(root)
    stats=read_json(root/'experiments/visual_training/v1/data_statistics.json')
    if any(v>.05 for x in stats.values() for v in x['collision_fraction'].values()): raise RuntimeError('Approved5% exception exceeded')
    if not read_json(root/'experiments/visual_training/v1/fp32_diagnostic/result.json')['passed']: raise RuntimeError('FP32 memorization not verified')
    history=read_json(root/'experiments/visual_training/v1/freeze.json')['history']
    for p in (root/'experiments/visual_training/v1').rglob('*'):
        if p.is_file(): history[p.relative_to(root).as_posix()]=sha256(p)
    inherited=read_json(root/'experiments/visual_training/v1/gpu_budget.json'); old=inherited['historical_seconds']+inherited['charged_seconds']
    write_json(d/'freeze.json',{'sources':source_hashes(root),'history':history,'historical_seconds':old,'config':cfg,
                'train_sha256':sha256(root/'data/synthetic/v3/train.jsonl'),'validation_sha256':sha256(root/'data/synthetic/v3/val.jsonl'),
                'authorization':{'resource':'User: You dont have a limit you can train it for around1-1.5 hrs if required; planning window90 new campaign minutes','collisions':'User explicit answer: Approve the5% collision exception'},'test_access':False})
    return {'historical_minutes':old/60,'campaign_seconds':cfg['campaign_seconds']}


class CampaignSession:
    def __init__(self,root,cfg):
        self.path=directory(root)/'gpu_budget.json'; self.cfg=cfg
        self.ledger=read_json(self.path) if self.path.exists() else {'historical_seconds':read_json(directory(root)/'freeze.json')['historical_seconds'],'charged_seconds':0.,'campaign_authorization_seconds':cfg['campaign_seconds'],'sessions':[]}
        if any(x['actual_seconds'] is None for x in self.ledger['sessions']): raise RuntimeError('Prior session has unresolved reservation; inspect process and reconcile, never reset')
        self.reserve=cfg['campaign_seconds']-self.ledger['charged_seconds']
        if self.reserve<60: raise RuntimeError('Campaign planning window consumed; explicit continuation required')
    def __enter__(self):
        self.started=time.monotonic(); self.entry={'reserved_seconds':self.reserve,'actual_seconds':None,'start_unix':time.time()}; self.ledger['sessions'].append(self.entry); self.ledger['charged_seconds']+=self.reserve; write_json(self.path,self.ledger); return self
    @property
    def remaining(self): return self.reserve-(time.monotonic()-self.started)
    def __exit__(self,*args):
        elapsed=time.monotonic()-self.started; self.entry['actual_seconds']=elapsed; self.ledger['charged_seconds']+=elapsed-self.reserve; write_json(self.path,self.ledger)


def atomic_save(state,path):
    temporary=path.with_suffix('.tmp'); torch.save(state,temporary); temporary.replace(path)


def improved(score,best): return score-best>=max(1e-4,abs(best)*.005)


def run(root,cfg):
    verify(root); d=directory(root); seed_everything(cfg['seed']); train=VisualDataset(root,'train',640); val=VisualDataset(root,'val',640)
    with CampaignSession(root,cfg) as session:
        model=load_model(root,cfg); checkpoint=d/'last.pt'
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location='cpu',weights_only=False); model.load_state_dict(state['model'])
            # Only our hash-bound local checkpoint, never external pickle.
        else:
            initial=evaluate(model,val,cfg,d,'initial'); state={'stage':'warmup','epoch':0,'cursor':0,'optimizer_steps':0,'history':[],'best':-1.,'stage_best':-1.,'stale':0,'loss_sum':0.,'train_count':0,'norm_max':0.,'initial':initial,'optimizer':None}
        if state.get('finished'): return {'status':state['finished'],'history_epochs':len(state['history'])}
        stage=state['stage']; opt=optimizer(model,cfg,stage)
        if state['optimizer'] is not None: opt.load_state_dict(state['optimizer'])
        if 'torch_rng' in state:
            torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng']); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        def commit():
            state.update({'model':model.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'config':cfg})
            atomic_save(state,checkpoint)
        commit()
        while True:
            if session.remaining<cfg['validation_reserve_seconds']+20:
                commit(); write_json(d/'run_status.json',{'status':'budget_window_partial','committed_steps':state['optimizer_steps'],'stage':state['stage'],'epoch':state['epoch'],'cursor':state['cursor']}); return {'status':'budget_window_partial'}
            epoch=state['epoch']; stage=state['stage']; order=np.random.default_rng(cfg['seed']+(0 if stage=='warmup' else 100)+epoch).permutation(len(train))
            torch.cuda.reset_peak_memory_stats(); epoch_start=time.monotonic(); model.train()
            for cursor in range(state['cursor'],len(order),cfg['accumulation']):
                if session.remaining<cfg['validation_reserve_seconds']+20: break
                indices=order[cursor:cursor+cfg['accumulation']]; opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:
                    factor=(cursor+len(indices))/len(order) if epoch==0 else .01+.99*.5*(1+math.cos(math.pi*epoch/max(1,(5 if stage=='warmup' else 30)-1)))
                    group['lr']=group['base_lr']*factor
                loss_sum=0.
                for index in indices:
                    image,target,_,_=train[int(index)]; _,dense=model.encode_image(image[None].cuda()); loss,_=detection_loss(dense,target)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite FP32 training loss')
                    (loss/len(indices)).backward(); loss_sum+=float(loss.detach())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm): raise RuntimeError('Nonfinite FP32 gradient')
                opt.step(); state['cursor']=cursor+len(indices); state['optimizer_steps']+=1; state['loss_sum']+=loss_sum; state['train_count']+=len(indices); state['norm_max']=max(state['norm_max'],float(norm)); commit()
                if state['optimizer_steps']%25==0:
                    progress={'stage':stage,'epoch':epoch+1,'charts':state['cursor'],'total_optimizer_steps':state['optimizer_steps'],'loss':state['loss_sum']/state['train_count'],'remaining_seconds':session.remaining}
                    write_json(d/'progress.json',progress); print(progress,flush=True)
            if state['cursor']<len(train): continue
            metrics=evaluate(model,val,cfg,d,f'{stage}_{epoch+1}'); score=metrics['ap50']['macro_ap']
            row={'stage':stage,'epoch':epoch+1,'train_images':state['train_count'],'train_loss':state['loss_sum']/state['train_count'],'validation':metrics,'optimizer_steps':state['optimizer_steps'],'gradient_norm_max':state['norm_max'],'lr':[g['lr'] for g in opt.param_groups],'wall_seconds_this_resume_segment':time.monotonic()-epoch_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            state['history'].append(row); write_json(d/'history.json',state['history'])
            atomic_save({'model':model.state_dict(),'config':cfg,'stage':stage,'epoch':epoch+1},d/f'{stage}_{epoch+1}.pt')
            if score>state['best']:
                state['best']=score; atomic_save({'model':model.state_dict(),'config':cfg,'stage':stage,'epoch':epoch+1,'metrics':metrics},d/'best.pt')
            if improved(score,state['stage_best']): state['stage_best']=score; state['stale']=0
            else: state['stale']+=1
            print({'epoch_finished':f'{stage}_{epoch+1}','train_loss':row['train_loss'],'val_ap50':score,'point_recall':metrics['ap50']['per_class']['point']['recall'],'tick_recall':metrics['ap50']['per_class']['tick']['recall']},flush=True)
            state.update({'epoch':epoch+1,'cursor':0,'loss_sum':0.,'train_count':0,'norm_max':0.})
            if stage=='warmup' and epoch+1==5:
                state.update({'stage':'perception','epoch':0,'stage_best':-1.,'stale':0}); opt=optimizer(model,cfg,'perception')
            elif stage=='perception' and ((epoch+1>=5 and state['stale']>=5) or epoch+1==30):
                state['finished']='early_stopped' if state['stale']>=5 else 'maximum_curriculum_complete'; commit(); write_json(d/'run_status.json',{'status':state['finished'],'committed_steps':state['optimizer_steps']}); return {'status':state['finished']}
            commit()


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['freeze','verify','run']); p.add_argument('--root',default='.'); args=p.parse_args(); root=Path(args.root).resolve(); cfg=read_json(root/'configs/full_visual_training.json')
    print(verify(root) if args.command=='verify' else freeze(root,cfg) if args.command=='freeze' else run(root,cfg),flush=True)


if __name__=='__main__': main()
