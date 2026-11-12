"""One-time OCR audit: separate oracle-crop diagnostic and predicted-region pipeline."""
from pathlib import Path
import sys,shutil,time,re
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,read_jsonl,write_json,sha256
from sensegraph.neural_spec import CropCTCRecognizer
from sensegraph.ocr_visual import collect_crops,transcribe,recognition_metrics
from sensegraph.ocr_inference import recognize_regions
from sensegraph.perception import detection_target,match_nodes
from sensegraph.quality_budget import QualitySession
from sensegraph.ocr_evaluation import region_text_counts


def main():
    torch.set_num_threads(1); base=ROOT/'experiments/visual_training'; d=base/'ocr_holdout'; ocr=base/'ocr_v1'; visual=base/'independent_holdout'
    if (d/'attempt.json').exists(): raise RuntimeError('OCR holdout already attempted; no repeated test tuning')
    selected=read_json(visual/'selection.json')['selected']; manifest=ROOT/'data/synthetic/perception_holdout_v1/test.jsonl'
    assert sha256(manifest)==read_json(visual/'generation_complete.json')['manifest_sha256']
    source=ocr/'best.pt'; checkpoint=torch.load(source,weights_only=True,map_location='cpu')
    write_json(d/'attempt.json',{'time_unix':time.time(),'ocr_checkpoint_sha256':sha256(source),'ocr_selected_epoch':checkpoint['epoch'],'ocr_selection':'validation CER only; before holdout inference','vision_selection_locked':selected,'holdout_sha256':sha256(manifest)})
    records=read_jsonl(manifest); predictions=torch.load(visual/f'{selected}_predictions.pt',weights_only=True,map_location='cpu')['predictions']
    with QualitySession(d/'gpu_budget.json',ocr/'gpu_budget.json',180,min_seconds=120):
        started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        model=CropCTCRecognizer().cuda().eval(); model.load_state_dict(checkpoint['model'],strict=True)
        training=torch.load(ROOT/'data/ocr_visual_v1/train.pt',weights_only=True,map_location='cpu')
        train_diagnostic=recognition_metrics(training,transcribe(model,training)); training_vocabulary={s['text'] for s in training}; del training
        oracle,bad=collect_crops(ROOT,manifest); oracle_strings=transcribe(model,oracle); conditional=recognition_metrics(oracle,oracle_strings)
        numeric=[(s,p) for s,p in zip(oracle,oracle_strings) if s['type']=='tick_label' and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?',s['text'])]
        conditional['numeric_tick_labels']=recognition_metrics([s for s,p in numeric],[p for s,p in numeric])
        unseen=[(s,p) for s,p in zip(oracle,oracle_strings) if s['text'] not in training_vocabulary]
        conditional['unseen_transcriptions_DIAGNOSTIC']=recognition_metrics([s for s,p in unseen],[p for s,p in unseen]) if unseen else {'crops':0,'cer':None,'exact':None,'status':'unavailable_no_unseen_transcriptions'}
        conditional['vocabulary_overlap']={'unique_holdout_strings':len({s['text'] for s in oracle}),'seen_unique_strings':len({s['text'] for s in oracle}&training_vocabulary),'unseen_crops':len(unseen),'scope':'Scoring only; vocabulary is never used for decoding/correction'}
        totals={'gt_words':0,'pred_words':0,'correct_words':0,'matched':0,'gt_characters':0,'character_errors':0,'matched_characters':0,'matched_errors':0}; cases=[]
        for record,p in zip(records,predictions):
            active=((p['labels']==4)|(p['labels']==5))&(p['scores']>=.25)
            pred={k:p[k][active] for k in ['boxes','scores','labels']}
            with Image.open(ROOT/record['image_path']) as image:
                words=recognize_regions(model,image.convert('RGB'),pred['boxes'].tolist(),device='cuda')
            target,ids,_=detection_target(record); keep=(target['labels']==4)|(target['labels']==5); target={k:v[keep] for k,v in target.items()}; ids=[identity for identity,flag in zip(ids,keep.tolist()) if flag]
            lookup={n['id']:n for n in record['nodes']}; gold=[lookup[identity]['text'] for identity in ids]
            matches,_=match_nodes(pred,target,.5); counts=region_text_counts(words,gold,matches.tolist())
            for key,value in counts.items(): totals[key]+=value
            cases.append({'id':record['id'],'predicted_words':words,'predicted_boxes':pred['boxes'].tolist(),'reference_texts_for_scoring_only':gold,'matched_gt_indices':matches.tolist(),'counts':counts})
        result={'training_crop_recognition_DIAGNOSTIC':train_diagnostic,'selected_validation_crop_recognition':checkpoint['validation'],'resource':{'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},'oracle_gt_crop_recognition_DIAGNOSTIC':conditional,'unsupported_oracle_text':bad,'end_to_end':{
           **totals,'word_precision':totals['correct_words']/max(1,totals['pred_words']),'word_recall':totals['correct_words']/max(1,totals['gt_words']),
           'word_f1':2*totals['correct_words']/max(1,totals['pred_words']+totals['gt_words']),'cer_including_missing_and_extra_regions':totals['character_errors']/max(1,totals['gt_characters']),
           'conditional_matched_region_cer':totals['matched_errors']/max(1,totals['matched_characters'])},'model_selection_changed':False,
           'definitions':'All GT text/legend regions in denominator. Missing GT => deleted characters/word FN; unmatched predicted strings => inserted characters/word FP. Exact-word error counts wrong transcription as both FP/FN. End-to-end CER can exceed1. No source string/crop/type used in prediction; GT only after recognizing all predicted text regions.'}
        write_json(d/'results.json',result); write_json(d/'cases.json',cases); write_json(d/'oracle_predictions.json',[{'chart':s['chart'],'node':s['node'],'reference':s['text'],'prediction':p} for s,p in zip(oracle,oracle_strings)])
    release=base/'release'; shutil.copyfile(source,release/'ocr.pt')
    vision=torch.load(release/'selected.pt',weights_only=True,map_location='cpu'); combined={**vision,'model':dict(vision['model']),'ocr_selected_epoch':checkpoint['epoch'],'trained_scopes':['visual_perception','ASCII_chart_crop_OCR']}
    for key,value in checkpoint['model'].items(): combined['model']['ocr.'+key]=value
    assert all(torch.equal(vision['model'][key],value) for key,value in combined['model'].items() if not key.startswith('ocr.'))
    torch.save(combined,release/'selected_with_ocr.pt')
    write_json(release/'ocr_manifest.json',{'ocr_sha256':sha256(release/'ocr.pt'),'combined_checkpoint_sha256':sha256(release/'selected_with_ocr.pt'),'vision_non_ocr_tensors_unchanged':True,'scope':'Vision+ASCII OCR only; graph, question reasoning and numerical calibration are NOT trained/validated by this export.'})
    print({'oracle_crop_CER':conditional['cer'],'end_to_end_word_f1':result['end_to_end']['word_f1'],'end_to_end_CER':result['end_to_end']['cer_including_missing_and_extra_regions']})


if __name__=='__main__': main()
