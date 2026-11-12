"""Audit-only recall on twenty already-acquired ChartQA TRAIN images, partial labels."""
from pathlib import Path
import sys,time
from collections import Counter
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph.neural_spec import SenseGraphNet,CropCTCRecognizer
from sensegraph.visual_training import preprocess
from sensegraph.full_visual_training import decode_maps
from sensegraph.perception import detection_target,match_nodes,CLASSES
from sensegraph.visual_experiment import overlay
from sensegraph.ocr_inference import recognize_regions
from sensegraph.quality_budget import QualitySession


def main():
    torch.set_num_threads(1); base=ROOT/'experiments/visual_training'; d=base/'real_partial'; release=base/'release'
    if (d/'attempt.json').exists(): raise RuntimeError('Real partial audit already attempted; preserve it')
    assert read_json(base/'independent_holdout/results.json')['complete']
    paths=sorted((ROOT/'data/real_annotations/chartqa/converted').glob('*.json')); assert len(paths)==20
    manifest=read_json(ROOT/'data/real_annotations/chartqa/source_manifest.json'); assert manifest['split']=='train'
    for item in manifest['files']: assert sha256(ROOT/item['path'])==item['sha256']
    checkpoint=release/'selected.pt'; saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
    inventory=read_json(d/'annotation_inventory.json') if (d/'annotation_inventory.json').exists() else None
    write_json(d/'attempt.json',{'started':time.time(),'script_sha256':sha256(Path(__file__)),'checkpoint_sha256':sha256(checkpoint),
        'converted_annotation_hashes':{p.name:sha256(p) for p in paths},'source_manifest_sha256':sha256(ROOT/'data/real_annotations/chartqa/source_manifest.json'),'preinference_inventory_sha256':sha256(d/'annotation_inventory.json') if inventory else None,
        'scope':'Existing twenty ChartQA training images, partial source annotations. Audit only: no adaptation, selection or complete precision/AP claim.'})
    prior=next(p for p in [base/'graph_transfer/gpu_budget.json',base/'ocr_holdout/gpu_budget.json',base/'ocr_v1/gpu_budget.json',base/'independent_holdout/gpu_budget.json'] if p.exists())
    counts={}; cases=[]; predictions=[]; rejected=Counter(); failure=[]
    with QualitySession(d/'gpu_budget.json',prior,180,min_seconds=60) as session:
        started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        model=SenseGraphNet().cuda().eval(); model.load_state_dict(saved['model'],strict=True)
        recognizer=None
        if (release/'ocr.pt').exists():
            recognizer=CropCTCRecognizer().cuda().eval(); recognizer.load_state_dict(torch.load(release/'ocr.pt',weights_only=True,map_location='cuda')['model'])
        for path in paths:
            record=read_json(path)
            with Image.open(ROOT/record['image_path']) as raw: w,h=raw.size
            if sha256(ROOT/record['image_path'])!=record['image_sha256']: raise RuntimeError('Paired real image changed')
            record.update({'id':path.stem,'width':w,'height':h})
            target,ids,bad=detection_target(record); rejected.update(item['reason'] for item in bad)
            if session.remaining<10:
                failure.append(path.stem); pred={'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long),'scores':torch.empty(0)}; words=[]
            else:
                image,affine=preprocess(ROOT/record['image_path'],640)
                with torch.inference_mode(): _,dense=model.encode_image(image[None].cuda()); pred=decode_maps(dense,affine,saved['config'])
                active=pred['scores']>=.25; pred={k:v[active] for k,v in pred.items()}
                words=[]
                if recognizer is not None:
                    text=(pred['labels']==4)|(pred['labels']==5)
                    with Image.open(ROOT/record['image_path']) as raw: words=recognize_regions(recognizer,raw.convert('RGB'),pred['boxes'][text].tolist(),device='cuda')
            # Predictions are complete before source annotations are matched for scoring.
            matched,_=match_nodes(pred,target,.5); found=set(g for g in matched.tolist() if g>=0)
            for i,label in enumerate(target['labels'].tolist()):
                row=counts.setdefault(CLASSES[label],{'source_supported_boxes':0,'recovered':0}); row['source_supported_boxes']+=1; row['recovered']+=i in found
            predictions.append({'id':path.stem,**pred}); cases.append({'id':path.stem,'source_chart_type':record['chart_type'],'supported_source_boxes':len(ids),'recovered':len(found),'predicted_texts':words,'prediction_count':len(pred['boxes'])})
            overlay(ROOT,record,pred,d/f'{path.stem}_overlay.png')
        resource={'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    for row in counts.values(): row['recall50']=row['recovered']/max(1,row['source_supported_boxes'])
    if inventory: assert {k:v['source_supported_boxes'] for k,v in counts.items()}==inventory['counts'] and dict(rejected)==inventory['rejected']
    write_json(d/'results.json',{'images':len(paths),'complete':not failure,'failures':failure,'by_source_supported_class':counts,'rejected_annotation_reasons':dict(rejected),'cases':cases,'resource':resource,
        'limits':'Partial labels cannot define false positives, full AP/precision, complete recall or QA accuracy. These are twenty existing official TRAIN examples, not an independent benchmark test. Source line boxes are segment envelopes; zero-area boxes remain excluded. Horizontal bars and real typography are outside synthetic training support. Raw OCR strings are unverified, not numerical calibration. No post-audit tuning permitted.'})
    torch.save(predictions,d/'predictions.pt'); print({'images':len(paths),'supported_class_recall':counts},flush=True)


if __name__=='__main__': main()
