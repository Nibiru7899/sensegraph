"""Frozen matched continuation experiment; no real-background or downstream training."""
from pathlib import Path
import time,math,random,shutil
from collections import Counter
import numpy as np
import torch
from .io import read_json,read_jsonl,write_json,sha256
from .neural_spec import SenseGraphNet
from .visual_training import detection_loss
from .collision_targets import CollisionInvariantDataset
from .domain_data import DomainDataset,annotation_overlay
from .domain_metrics import complete_metrics,partial_counts,summarize_partial,admission,select_checkpoint,bootstrap_real
from .full_visual_training import decode_maps,keypoint_metrics,atomic_save,improved
from .perception import detection_target
from .visual_experiment import overlay
from .relation_training import seed_everything
from .quality_budget import QualitySession


def directory(root,cfg): return Path(root)/cfg['experiment']


def latest_ledger(root,cfg):
    chain=directory(root,cfg)/'resource_chain.json'
    return Path(root)/read_json(chain)[-1]['path'] if chain.exists() else Path(root)/cfg['prior_gpu_ledger']


def stage_used(root,cfg,stage):
    chain=directory(root,cfg)/'resource_chain.json'
    return sum(read_json(Path(root)/r['path'])['charged_seconds'] for r in read_json(chain) if r['stage']==stage) if chain.exists() else 0.


class DomainSession:
    def __init__(self,root,cfg,stage,reserve_after=0): self.root=Path(root); self.cfg=cfg; self.stage=stage; self.reserve_after=reserve_after
    def __enter__(self):
        d=directory(self.root,self.cfg); chain=d/'resource_chain.json'; rows=read_json(chain) if chain.exists() else []; prior=latest_ledger(self.root,self.cfg); old=read_json(prior)
        if 'sessions' in old:
            if any(s['actual_seconds'] is None for s in old['sessions']): raise RuntimeError('Prior GPU reservation unresolved')
        elif old.get('actual_seconds') is None: raise RuntimeError('Prior GPU session active')
        remaining=18000-old['historical_seconds']-old['charged_seconds']; maximum=self.cfg[self.stage+'_seconds']-stage_used(self.root,self.cfg,self.stage)
        allowed=min(maximum,remaining-self.reserve_after)
        if allowed<30: raise RuntimeError('Insufficient bounded stage budget')
        path=d/f'gpu_{self.stage}_{len(rows):03d}.json'; self.session=QualitySession(path,prior,allowed,min_seconds=30); value=self.session.__enter__()
        rows.append({'stage':self.stage,'path':path.relative_to(self.root).as_posix()}); write_json(chain,rows); return value
    def __exit__(self,*args): return self.session.__exit__(*args)


def load_reference(root,cfg):
    path=Path(root)/cfg['reference']
    if sha256(path)!=cfg['reference_sha256']: raise RuntimeError('Reference checkpoint identity changed')
    saved=torch.load(path,weights_only=True,map_location='cpu'); model=SenseGraphNet(); model.load_state_dict(saved['model'],strict=True); return model


def optimizer(model,cfg):
    model.set_stage('perception'); groups=[]
    for backbone,rate in [(False,cfg['lr']),(True,cfg['backbone_lr'])]:
        parameters=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('backbone.')==backbone]
        groups.append({'params':parameters,'lr':rate,'base_lr':rate})
    return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])


def datasets(root,cfg):
    return {'control':CollisionInvariantDataset(Path(root),'train',640),'realistic':DomainDataset(root,str(Path(cfg['synthetic_output'])/'train.jsonl'),'train')},DomainDataset(root,str(Path(cfg['synthetic_output'])/'val.jsonl'),'val')


@torch.inference_mode()
def evaluate_synthetic(model,data,cfg,out,label,session=None):
    model.eval(); preds=[]; refs=[]; losses=[]; started=time.monotonic(); failed=[]
    for i in range(len(data)):
        image,target,affine,record=data[i]
        if session and session.remaining<15: raise RuntimeError('Insufficient budget for complete validation')
        _,dense=model.encode_image(image[None].cuda()); loss,_=detection_loss(dense,target)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite validation loss')
        pred=decode_maps(dense,affine,cfg); preds.append(pred); refs.append(detection_target(record)[0]); losses.append(float(loss))
        if i<3: overlay(data.root,record,pred,out/f'{label}_{i}.png')
    result={'images':len(preds),'loss':float(np.mean(losses)),'ap50':complete_metrics(preds,refs,.5),'ap75':complete_metrics(preds,refs,.75),'keypoints_5px':keypoint_metrics(preds,refs),'seconds':time.monotonic()-started,
            'mean_proposals':float(np.mean([int((p['scores']>=.25).sum()) for p in preds])),'families':{}}
    for family in sorted({r.get('family',r['chart_type']) for r in data.records}):
        indices=[i for i,r in enumerate(data.records) if r.get('family',r['chart_type'])==family]
        result['families'][family]=complete_metrics([preds[i] for i in indices],[refs[i] for i in indices])
    write_json(out/f'{label}_metrics.json',result); torch.save({'predictions':preds,'references':refs,'ids':[r['id'] for r in data.records]},out/f'{label}_predictions.pt'); return result


@torch.inference_mode()
def evaluate_real(model,root,records,cfg,out,label,session=None):
    from .visual_training import preprocess
    model.eval(); preds=[]; cases=[]; started=time.monotonic(); failures=[]
    for i,record in enumerate(records):
        try:
            if session and session.remaining<15: raise RuntimeError('Inference reserve reached')
            image,affine=preprocess(Path(root)/record['image_path'],640); _,dense=model.encode_image(image[None].cuda()); pred=decode_maps(dense,affine,cfg)
            if not torch.isfinite(pred['boxes']).all(): raise RuntimeError('Nonfinite prediction')
        except Exception as error:
            failures.append({'id':record['id'],'error':str(error)}); pred={'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long),'scores':torch.empty(0),'centers':torch.empty(0,2)}
        preds.append(pred); cases.append(partial_counts(pred,record))
        if i<6: overlay(root,record,pred,out/f'{label}_{i}.png')
    result={**summarize_partial(cases),'seconds':time.monotonic()-started,'failures':failures,'complete':not failures}
    write_json(out/f'{label}_metrics.json',result); write_json(out/f'{label}_cases.json',cases); torch.save({'predictions':preds,'ids':[r['id'] for r in records]},out/f'{label}_predictions.pt'); return result


def source_hashes(root,cfg):
    root=Path(root)
    paths=list((root/'src/sensegraph').glob('adaptation*.py'))+[root/'configs/adaptation_v2.json',root/'research/adaptation_protocol_v2.md']
    inherited=read_json(root/'experiments/domain_generalization/v1/freeze.json')['sources']
    if any(sha256(root/p)!=h for p,h in inherited.items()): raise RuntimeError('Frozen v1 source changed')
    return {**inherited,**{p.relative_to(root).as_posix():sha256(p) for p in paths}}


def verify(root,cfg):
    d=directory(root,cfg); frozen=read_json(d/'freeze.json')
    if frozen['config']!=cfg: raise RuntimeError('Frozen configuration changed')
    if frozen['sources']!=source_hashes(root,cfg): raise RuntimeError('Domain frozen code/config changed')
    for p,h in frozen['data_hashes'].items():
        if sha256(Path(root)/p)!=h: raise RuntimeError(f'Frozen dataset changed: {p}')
    initial=read_json(d/'initial.json'); old=read_json(Path(root)/'experiments/visual_training/release/source_and_release_hashes.json')['files']; changed=[]
    for name,h in initial['continuation_archives'].items():
        if sha256(d/'archive'/name)!=h: raise RuntimeError('Continuation archive changed')
    for p,h in initial['parent_artifacts'].items():
        if sha256(Path(root)/p)!=h: raise RuntimeError('Parent pilot artifact changed')
    for p,h in old.items():
        actual=Path(root)/initial['archive_mapping'].get(p,p)
        if sha256(actual)!=h: changed.append(p)
    if changed: raise RuntimeError(f'Historical artifact changed: {changed[:4]}')
    historical=read_json(Path(root)/'experiments/visual_training/v3/freeze.json')['history']
    for p,h in historical.items():
        if sha256(Path(root)/p)!=h: raise RuntimeError(f'Protected prior experiment changed: {p}')
    if sha256(Path(root)/cfg['prior_gpu_ledger'])!=initial['prior_gpu_ledger_sha256']: raise RuntimeError('Inherited closed resource ledger changed')
    prior=Path(root)/cfg['prior_gpu_ledger']; chain=d/'resource_chain.json'
    for row in read_json(chain) if chain.exists() else []:
        path=Path(root)/row['path']; ledger=read_json(path)
        if ledger['prior_sha256']!=sha256(prior) or ledger['actual_seconds'] is None: raise RuntimeError('Resource chain changed or session unresolved')
        if ledger['historical_seconds']+ledger['charged_seconds']>cfg['gpu_ceiling_seconds']: raise RuntimeError('GPU ceiling exceeded')
        prior=path
    download=read_json(Path(root)/'experiments/download_ledger.json')['transferred_bytes']
    if download>cfg['download_ceiling'] or download-initial['download_initial']>cfg['new_download_limit']: raise RuntimeError('Download ceiling exceeded')
    return {'historical_snapshot_files':len(old),'historical_changes':0,'protected_experiment_files':len(historical),'domain_frozen_sources':len(frozen['sources']),
            'status_archive_sha256':sha256(Path(root)/initial['archive_mapping']['PROJECT_STATUS.md']),'resource_chain_closed':True}


def freeze(root,cfg):
    d=directory(root,cfg)
    if (d/'freeze.json').exists(): return verify(root,cfg)
    gate=read_json(d/'synthetic_validation.json')
    real=read_json(Path(root)/cfg['real_output']/'manifest.json')
    if not real['complete'] or not (d/'real_review.json').exists(): raise RuntimeError('Real inventory/review incomplete')
    files={}
    for base in [Path(root)/cfg['synthetic_output'],Path(root)/cfg['real_output'],Path(root)/'data/models/domain_trocr']:
        for file in base.rglob('*'):
            if file.is_file() and 'range_cache' not in file.parts: files[file.relative_to(root).as_posix()]=sha256(file)
    for name in ['data_review.json','real_review.json','synthetic_validation.json','diagnosis.json','initial.json']:
        files[(d/name).relative_to(root).as_posix()]=sha256(d/name)
    write_json(d/'freeze.json',{'sources':source_hashes(root,cfg),'data_hashes':files,'reference_sha256':cfg['reference_sha256'],'config':cfg,
        'training_gate_passed':gate['valid'],'training_gate_failures':gate['failures'],
        'primary':'macro supported real-validation recall50; synthetic validation AP50 guard >= A-.05; earliest epoch ties; overall A/control/realistic tie order',
        'test_use':'Fresh real final inventory once after selections; old synthetic/protected audits remain sealed','C_D_status':'not fitted; missing real background supervision and separate matched allocation'})
    for p,h in source_hashes(root,cfg).items():
        dest=d/'source_snapshots'/p; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(Path(root)/p,dest)
    return verify(root,cfg)


def fitting_gate(root,cfg):
    frozen=read_json(directory(root,cfg)/'freeze.json')
    if not frozen['training_gate_passed']: raise RuntimeError('Synthetic annotation gate failed; fitting/profiling updates prohibited')


def profile(root,cfg):
    verify(root,cfg); fitting_gate(root,cfg); d=directory(root,cfg)
    if (d/'profile.json').exists(): return read_json(d/'profile.json')
    seed_everything(cfg['seed']); train,val=datasets(root,cfg); realval=read_jsonl(Path(root)/cfg['real_output']/'val.jsonl')
    times=[]; ocr_probe=None
    with DomainSession(root,cfg,'profile',reserve_after=cfg['evaluation_seconds']) as session:
        model=load_reference(root,cfg).cuda().eval(); torch.cuda.reset_peak_memory_stats()
        synthetic=evaluate_synthetic(model,val,cfg,d,'A_synthetic_val',session); real=evaluate_real(model,root,realval,cfg,d,'A_real_val',session)
        opt=optimizer(model,cfg); model.train()
        frozen_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        buffers_before={n:p.cpu().clone() for n,p in model.named_buffers()}
        # Discard this fitting-only profile; both scientific arms reload exact A weights.
        for step in range(20):
            if session.remaining<30: raise RuntimeError('Profile capacity exhausted before complete admission evidence')
            started=time.monotonic(); opt.zero_grad(set_to_none=True)
            arm='control' if step<10 else 'realistic'
            for index in range((step%10)*8,(step%10+1)*8):
                image,target,_,_=train[arm][index]; _,dense=model.encode_image(image[None].cuda()); loss,_=detection_loss(dense,target)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite profile loss')
                (loss/8).backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite profile gradient')
            opt.step(); atomic_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state()},d/'profile_discarded.pt'); torch.cuda.synchronize(); times.append(time.monotonic()-started)
        invariant={'frozen_parameters_equal':all(torch.equal(p.cpu(),frozen_before[n]) for n,p in model.named_parameters() if n in frozen_before),
                   'BN_buffers_equal':all(torch.equal(p.cpu(),buffers_before[n]) for n,p in model.named_buffers())}
        saved=torch.load(d/'profile_discarded.pt',map_location='cpu',weights_only=True)
        invariant['checkpoint_reload_equal']=all(torch.equal(p.cpu(),saved['model'][n]) for n,p in model.state_dict().items())
        if not all(invariant.values()): raise RuntimeError('Profile invariant failed')
        del model,opt,saved,frozen_before,buffers_before; torch.cuda.empty_cache()
        source=Path(root)/'data/models/domain_trocr/source_manifest.json'
        if source.exists() and read_json(source).get('permission_resolved') and session.remaining>30:
            from .domain_ocr import Recognizer,trusted_text
            from PIL import Image
            recognizer=Recognizer(root,'trocr'); rec=read_jsonl(Path(root)/cfg['real_output']/'train.jsonl')[0]; boxes=[v['box'] for v in trusted_text(rec)][:5]
            started=time.monotonic()
            with Image.open(Path(root)/rec['image_path']) as image: words=recognizer.predict(image.convert('RGB'),boxes)
            ocr_probe={'crops':len(boxes),'seconds':time.monotonic()-started,'strings':words}; del recognizer; torch.cuda.empty_cache()
        projected=float(np.quantile(times,.95))*175*1.15+synthetic['seconds']+real['seconds']+10
        result={'invariants':invariant,'profile_updates_discarded':20,'profile_arms':['control']*10+['realistic']*10,'profile_charts':160,'step_seconds':times,'projected_epoch_seconds':projected,'reference_synthetic_ap50':synthetic['ap50']['macro_ap'],
                'reference_real_recall':real['macro_supported_recall50'],'ocr_probe':ocr_probe,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    ledger=read_json(latest_ledger(root,cfg)); remaining=18000-ledger['historical_seconds']-ledger['charged_seconds']; result['admission']=admission(result,remaining,cfg['evaluation_seconds'])
    write_json(d/'profile.json',result); print({'admission':result['admission'],'projected_epoch_seconds':projected},flush=True); return result


def rng_state():
    state=np.random.get_state(); return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]]}


def restore_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda']); random.setstate(state['python']); a=state['numpy']; np.random.set_state((a[0],np.asarray(a[1],dtype=np.uint32),a[2],a[3],a[4]))


def fit(root,cfg):
    verify(root,cfg); fitting_gate(root,cfg); d=directory(root,cfg); prof=read_json(d/'profile.json')
    ledger=read_json(latest_ledger(root,cfg)); capacity=admission(prof,18000-ledger['historical_seconds']-ledger['charged_seconds'],cfg['evaluation_seconds'])
    if not prof['admission']['admitted'] or not capacity['admitted']: return {'status':'blocked_capacity','evidence':capacity}
    if (d/'fit_status.json').exists() and read_json(d/'fit_status.json')['status'] in ['paired_early_stop','five_epoch_pilot_complete']: return read_json(d/'fit_status.json')
    seed_everything(cfg['seed']); train,val=datasets(root,cfg); realval=read_jsonl(Path(root)/cfg['real_output']/'val.jsonl'); cursorfile=d/'paired_cursor.json'
    cursor=read_json(cursorfile) if cursorfile.exists() else {'epoch':1,'arm':0}; histories={a:read_json(d/f'{a}_history.json') if (d/f'{a}_history.json').exists() else [] for a in train}
    with DomainSession(root,cfg,'fit',reserve_after=cfg['evaluation_seconds']) as session:
        while cursor['epoch']<=cfg['max_epochs']:
            epoch=cursor['epoch']; arm=['control','realistic'][cursor['arm']]; path=d/arm; path.mkdir(parents=True,exist_ok=True)
            seed_everything(cfg['seed']); model=load_reference(root,cfg).cuda(); opt=optimizer(model,cfg); last=path/'last.pt'
            frozen_reference={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
            buffer_reference={n:p.cpu().clone() for n,p in model.named_buffers()}
            if last.exists():
                state=torch.load(last,map_location='cpu',weights_only=True); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
                if state['freeze_sha256']!=sha256(d/'freeze.json'): raise RuntimeError('Resume checkpoint belongs to another frozen protocol')
            else: state={'epoch':epoch,'cursor':0,'steps':0,'loss_sum':0.,'seen':0}
            if state['epoch']==epoch+1 and state['cursor']==0 and any(r['epoch']==epoch for r in histories[arm]):
                cursor={'epoch':epoch if cursor['arm']==0 else epoch+1,'arm':1 if cursor['arm']==0 else 0}; write_json(cursorfile,cursor)
                del model,opt; torch.cuda.empty_cache(); continue
            if state['epoch']!=epoch: raise RuntimeError('Paired cursor/checkpoint epoch mismatch')
            order=np.random.default_rng(cfg['seed']+epoch).permutation(len(train[arm])); model.train(); started=time.monotonic(); torch.cuda.reset_peak_memory_stats(); norm_max=0.
            def commit():
                state.update({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'freeze_sha256':sha256(d/'freeze.json')}); atomic_save(state,last)
            for position in range(state['cursor'],len(order),8):
                if session.remaining<max(90,prof['projected_epoch_seconds']-float(np.quantile(prof['step_seconds'],.95))*175+30):
                    commit(); write_json(d/'fit_status.json',{'status':'resource_partial','epoch':epoch,'arm':arm,'cursor':state['cursor']}); return {'status':'resource_partial'}
                indices=order[position:position+8]; opt.zero_grad(set_to_none=True)
                factor=(position+len(indices))/len(order) if epoch==1 else .01+.99*.5*(1+math.cos(math.pi*(epoch-1)/4))
                for group in opt.param_groups: group['lr']=group['base_lr']*factor
                accumulated=0.
                for index in indices:
                    image,target,_,_=train[arm][int(index)]; _,dense=model.encode_image(image[None].cuda()); loss,_=detection_loss(dense,target)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite adaptation loss')
                    (loss/len(indices)).backward(); accumulated+=float(loss.detach())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm): raise RuntimeError('Nonfinite adaptation gradient')
                opt.step(); state.update({'cursor':position+len(indices),'steps':state['steps']+1,'loss_sum':state['loss_sum']+accumulated,'seen':state['seen']+len(indices)}); norm_max=max(norm_max,float(norm)); commit()
                if state['steps']%25==0: print({'arm':arm,'epoch':epoch,'charts':state['cursor'],'steps':state['steps'],'loss':state['loss_sum']/state['seen'],'remaining_seconds':session.remaining},flush=True)
            model.eval(); syn=evaluate_synthetic(model,val,cfg,path,f'epoch_{epoch}_synthetic_val',session); real=evaluate_real(model,root,realval,cfg,path,f'epoch_{epoch}_real_val',session)
            if not real['complete']: raise RuntimeError('Incomplete real validation cannot select checkpoint')
            if not all(torch.equal(p.cpu(),frozen_reference[n]) for n,p in model.named_parameters() if n in frozen_reference): raise RuntimeError('Frozen parameter changed')
            if not all(torch.equal(p.cpu(),buffer_reference[n]) for n,p in model.named_buffers()): raise RuntimeError('Frozen BN buffer changed')
            row={'arm':arm,'epoch':epoch,'train_images':state['seen'],'optimizer_steps':state['steps'],'train_loss':state['loss_sum']/state['seen'],'synthetic_ap50':syn['ap50']['macro_ap'],
                 'real_recall':real['macro_supported_recall50'],'complete':True,'frozen_parameters_equal':True,'BN_buffers_equal':True,'seconds_this_resume_segment':time.monotonic()-started,'gradient_norm_max':norm_max,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'lr':[g['lr'] for g in opt.param_groups]}
            weights={'model':model.state_dict(),'config':cfg,'stage':'domain_adaptation','epoch':epoch,'arm':arm,'validation':row}; atomic_save(weights,path/f'epoch_{epoch}.pt')
            histories[arm]=[r for r in histories[arm] if r['epoch']!=epoch]+[row]; write_json(d/f'{arm}_history.json',histories[arm]); selected=select_checkpoint(histories[arm],prof['reference_synthetic_ap50'],cfg['synthetic_guard_drop'])
            if selected and selected['epoch']==epoch: atomic_save(weights,path/'best.pt')
            state.update({'epoch':epoch+1,'cursor':0,'loss_sum':0.,'seen':0}); commit(); del model,opt; torch.cuda.empty_cache()
            print({'epoch_finished':epoch,'arm':arm,'real_recall':row['real_recall'],'synthetic_ap50':row['synthetic_ap50']},flush=True)
            cursor={'epoch':epoch if cursor['arm']==0 else epoch+1,'arm':1 if cursor['arm']==0 else 0}; write_json(cursorfile,cursor)
            if cursor['arm']==0 and epoch>=3:
                stale=[]
                for name,rows in histories.items():
                    significant=-1.; count=0
                    for r in rows:
                        score=r['real_recall']
                        if score-significant>=max(.001,abs(significant)*.005): significant=score; count=0
                        else: count+=1
                    stale.append(count>=2)
                if all(stale):
                    result={'status':'paired_early_stop','complete_paired_epochs':epoch,'interpretation':'Shared stopping, equal update budgets; one seed'}; write_json(d/'fit_status.json',result); return result
        result={'status':'five_epoch_pilot_complete','complete_paired_epochs':5,'interpretation':'Bounded pilot; convergence not implied if validation still improving'}; write_json(d/'fit_status.json',result); return result
