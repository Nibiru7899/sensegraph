"""One-time independent audit after validation-only selection; no retraining/tuning."""
from pathlib import Path
import sys,time,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,read_jsonl,write_json,sha256
from sensegraph.neural_spec import SenseGraphNet
from sensegraph.collision_targets import CollisionInvariantDataset
from sensegraph.full_visual_training import evaluate,keypoint_metrics
from sensegraph.perception import detector,image_tensor,detection_target,detection_metrics
from sensegraph.suspend_accounting import WorkingState,working_seconds


class AuditDataset(CollisionInvariantDataset):
    def __init__(self,root):
        self.root=root; self.size=640
        self.records=read_jsonl(root/'data/synthetic/perception_holdout_v1/test.jsonl')


def select_winner(candidates):
    return max(candidates,key=lambda name:(candidates[name]['validation_ap50'],name=='corrected_v3'))


class AuditSession:
    def __enter__(self):
        path=ROOT/'experiments/visual_training/v3/gpu_budget_awake.json'
        prior=read_json(path if path.exists() else path.with_name('gpu_budget.json'))
        if any(s['actual_seconds'] is None for s in prior['sessions']): raise RuntimeError('Training session active/unreconciled')
        self.path=ROOT/'experiments/visual_training/independent_holdout/gpu_budget.json'
        if self.path.exists(): raise RuntimeError('Holdout GPU attempt already exists; preserve it')
        inherited=prior['historical_seconds']+prior['charged_seconds']; self.reserve=min(600,18000-inherited)
        if self.reserve<300: raise RuntimeError('Insufficient five-hour budget for the declared audit')
        self.power=WorkingState(); self.power.__enter__(); self.start=working_seconds(); self.ledger={'historical_seconds':inherited,'charged_seconds':self.reserve,'reserved_seconds':self.reserve,'actual_seconds':None,'cumulative_ceiling_seconds':18000,'clock':'Windows working-state seconds, excludes sleep'}
        write_json(self.path,self.ledger); return self
    @property
    def remaining(self): return self.reserve-(working_seconds()-self.start)
    def __exit__(self,*args):
        elapsed=working_seconds()-self.start; self.ledger.update({'charged_seconds':elapsed,'actual_seconds':elapsed}); write_json(self.path,self.ledger); self.power.__exit__(*args)
        if self.ledger['historical_seconds']+elapsed>18000: raise RuntimeError('Cumulative five-hour budget exceeded')


def old_detector_evaluation(data,d):
    config=read_json(ROOT/'configs/perception.json'); model=detector(ROOT,config).cuda().eval()
    model.load_state_dict(torch.load(ROOT/'experiments/milestone4/detector.pt',weights_only=True,map_location='cuda')['state_dict'])
    predictions=[]; references=[]; start=time.monotonic()
    with torch.inference_mode():
        for record in data.records:
            image=image_tensor(ROOT/record['image_path']); p={k:v.cpu() for k,v in model([image.cuda()])[0].items()}
            p['centers']=(p['boxes'][:,:2]+p['boxes'][:,2:])/2; predictions.append(p); references.append(detection_target(record)[0])
    metrics={'images':len(predictions),'ap50':detection_metrics(predictions,references,.5,.25),'ap75':detection_metrics(predictions,references,.75,.25),'keypoints_5px':keypoint_metrics(predictions,references),'seconds':time.monotonic()-start,
             'differences':'Original checkpoint/native320 transform, min score .01, cap100; new heads use640,min .10,cap128. Same physical GT/class identities and .25 operating threshold. Keypoint location uses box center for this baseline.'}
    write_json(d/'mobilenet_metrics.json',metrics); torch.save({'predictions':predictions,'references':references,'ids':[r['id'] for r in data.records]},d/'mobilenet_predictions.pt')
    del model; torch.cuda.empty_cache(); return metrics


def main():
    torch.set_num_threads(1); d=ROOT/'experiments/visual_training/independent_holdout'; release=ROOT/'experiments/visual_training/release'
    if (d/'evaluation_attempt.json').exists() or (d/'selection.json').exists(): raise RuntimeError('Holdout already attempted/selected; no silent repeated evaluation')
    for version in ['v2','v3']:
        path=ROOT/f'experiments/visual_training/{version}/gpu_budget_awake.json'
        if not path.exists(): path=path.with_name('gpu_budget.json')
        budget=read_json(path)
        if any(s['actual_seconds'] is None for s in budget['sessions']): raise RuntimeError('Training active; no holdout selection or inference')
    candidates={}
    for version,name in [('v2','original_v2'),('v3','corrected_v3')]:
        run=ROOT/f'experiments/visual_training/{version}'; status=read_json(run/'run_status.json')['status']
        if status not in ['early_stopped','maximum_curriculum_complete','budget_window_partial']: raise RuntimeError('Training not terminal')
        history=read_json(run/'history.json'); best=max(history,key=lambda r:r['validation']['ap50']['macro_ap'])
        candidates[name]={'checkpoint':str((run/'best.pt').relative_to(ROOT)),'sha256':sha256(run/'best.pt'),'validation_ap50':best['validation']['ap50']['macro_ap'],'stage':best['stage'],'epoch':best['epoch'],'run_status':status}
    winner=select_winner(candidates); manifest=ROOT/'data/synthetic/perception_holdout_v1/test.jsonl'
    assert sha256(manifest)==read_json(d/'generation_complete.json')['manifest_sha256']
    records=read_jsonl(manifest)
    assert len(records)==300 and all(sha256(ROOT/r['image_path'])==r['image_sha256'] for r in records)
    overlap=read_json(d/'leakage_audit.json'); assert not overlap['exact_image_duplicates'] and not overlap['existing_numeric_reuse'] and not overlap['existing_template_reuse']
    selection={'selected':winner,'candidates':candidates,'selection_basis':'Complete300-chart validation AP50 only; locked before first holdout inference. No subsequent model selection or tuning from these results.','holdout_sha256':sha256(manifest),'script_sha256':sha256(Path(__file__)),
               'evaluation_order':[winner]+[name for name in candidates if name!=winner]+['mobilenet'],'time_unix':time.time()}
    write_json(d/'selection.json',selection); release.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/candidates[winner]['checkpoint'],release/'selected.pt'); assert sha256(release/'selected.pt')==candidates[winner]['sha256']
    write_json(release/'manifest.json',{**selection,'checkpoint_sha256':sha256(release/'selected.pt'),'scope':'Trained visual perception only; graph/QA/OCR/segmentation remain untrained.'})
    write_json(d/'evaluation_attempt.json',{'started':time.time(),'selection_sha256':sha256(d/'selection.json'),'model_selection_already_locked':True})
    data=AuditDataset(ROOT); results={}
    with AuditSession() as session:
        for name in selection['evaluation_order']:
            if session.remaining<100: break
            torch.cuda.reset_peak_memory_stats()
            if name=='mobilenet': metrics=old_detector_evaluation(data,d)
            else:
                path=ROOT/candidates[name]['checkpoint']; assert sha256(path)==candidates[name]['sha256']
                checkpoint=torch.load(path,weights_only=True,map_location='cpu'); model=SenseGraphNet().cuda().eval(); model.load_state_dict(checkpoint['model'],strict=True)
                metrics=evaluate(model,data,checkpoint['config'],d,name); del model; torch.cuda.empty_cache()
            metrics.update({'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            results[name]=metrics; write_json(d/f'{name}_metrics.json',metrics)
            print({'model':name,'holdout_ap50':metrics['ap50']['macro_ap'],'holdout_recall50':metrics['ap50']['macro_recall']},flush=True)
        training_diagnostic=None
        if len(results)==3 and session.remaining>=300:
            checkpoint=torch.load(release/'selected.pt',weights_only=True,map_location='cpu'); model=SenseGraphNet().cuda().eval(); model.load_state_dict(checkpoint['model'],strict=True)
            training_diagnostic=evaluate(model,CollisionInvariantDataset(ROOT,'train',640),checkpoint['config'],d,'selected_training_DIAGNOSTIC'); del model; torch.cuda.empty_cache()
            print({'scope':'TRAINING overfitting diagnostic only','ap50':training_diagnostic['ap50']['macro_ap']},flush=True)
        write_json(d/'results.json',{'selected_before_inference':winner,'results':results,'complete':len(results)==3,'selected_training_DIAGNOSTIC':training_diagnostic,'heldout_model_selection_or_retraining':False})


if __name__=='__main__': main()
