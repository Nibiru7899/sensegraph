"""Matched native-preprocessing OCR; partial real labels never imply negatives."""
from pathlib import Path
import re,time
from PIL import Image
import torch
from .io import read_json,write_json
from .neural_spec import CropCTCRecognizer
from .ocr_inference import recognize_regions
from .perception import detection_target,match_nodes
from .domain_metrics import text_metrics,supported_text_metrics


def native_crop(image,box):
    import math
    l,t=max(0,math.floor(box[0])),max(0,math.floor(box[1])); r,b=min(image.width,math.ceil(box[2])),min(image.height,math.ceil(box[3]))
    if r<=l or b<=t: raise ValueError('Invalid OCR box')
    crop=image.crop((l,t,r,b)).convert('RGB')
    return crop.rotate(-90,expand=True) if crop.height>1.5*crop.width else crop


class Recognizer:
    def __init__(self,root,kind,device='cuda'):
        self.kind=kind; self.device=device; self.truncations=0
        if kind=='ctc':
            self.model=CropCTCRecognizer().to(device).eval(); self.model.load_state_dict(torch.load(Path(root)/'experiments/visual_training/release/ocr.pt',map_location=device,weights_only=True)['model'])
        elif kind=='trocr':
            from transformers import TrOCRProcessor,VisionEncoderDecoderModel
            directory=Path(root)/'data/models/domain_trocr'; source=read_json(directory/'source_manifest.json')
            if not source['permission_resolved']: raise RuntimeError('OCR permission unresolved')
            self.processor=TrOCRProcessor.from_pretrained(str(directory),local_files_only=True,use_fast=False)
            self.model=VisionEncoderDecoderModel.from_pretrained(str(directory),local_files_only=True,use_safetensors=True,torch_dtype=torch.float32).to(device).eval()
        else: raise ValueError(kind)
    @torch.inference_mode()
    def predict(self,image,boxes):
        if self.kind=='ctc': return recognize_regions(self.model,image,boxes,self.device)
        result=[]
        for start in range(0,len(boxes),4):
            part=[native_crop(image,b) for b in boxes[start:start+4]]
            inputs=self.processor(images=part,return_tensors='pt').pixel_values.to(self.device)
            tokens=self.model.generate(inputs,do_sample=False,num_beams=1,max_new_tokens=128)
            eos=self.model.generation_config.eos_token_id
            eos=[eos] if isinstance(eos,int) else (eos or [])
            for row in tokens:
                if len(row)>=129 and not any(int(v) in eos for v in row[1:]): self.truncations+=1
            result.extend(self.processor.batch_decode(tokens,skip_special_tokens=True))
        return result


def trusted_text(record):
    from .domain_mapping import supported_record
    record,_=supported_record(record)
    target,ids,_=detection_target(record); lookup={n['id']:n for n in record['nodes']}; selected=[]
    for box,label,identity in zip(target['boxes'],target['labels'],ids):
        text=lookup[identity].get('text')
        if int(label) in (4,5) and isinstance(text,str) and text:
            selected.append({'id':identity,'box':box.tolist(),'label':int(label),'text':text,'type':lookup[identity]['type']})
    return selected


def oracle_ocr(root,records,recognizer,session=None):
    cases=[]; failures=[]; seconds=time.monotonic(); truncations_before=recognizer.truncations
    for record in records:
        labels=trusted_text(record)
        try:
            if session and session.remaining<60: raise RuntimeError('OCR resource reserve reached')
            with Image.open(Path(root)/record['image_path']) as raw: words=recognizer.predict(raw.convert('RGB'),[n['box'] for n in labels])
            if len(words)!=len(labels): raise RuntimeError('Recognizer output count mismatch')
        except Exception as error:
            words=['']*len(labels); failures.append({'id':record['id'],'error':f'{type(error).__name__}: {error}'})
        cases.extend({'chart':record['id'],'node':label['id'],'type':label['type'],'reference':label['text'],'prediction':word,
                      'longer_than_CTC_32':len(label['text'])>32,'geometry_vertical':label['box'][3]-label['box'][1]>1.5*(label['box'][2]-label['box'][0])} for label,word in zip(labels,words))
    refs=[v['reference'] for v in cases]; preds=[v['prediction'] for v in cases]; metrics=text_metrics(refs,preds); subsets={}
    vocabulary=set()
    # Vocabulary is used only AFTER recognition for scoring subgroup membership.
    training=torch.load(Path(root)/'data/ocr_visual_v1/train.pt',map_location='cpu',weights_only=True); vocabulary={s['text'] for s in training}; del training
    for name,predicate in {
        'numeric_tick_string':lambda v:v['type']=='tick_label' and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?',v['reference']) is not None,
        'negative_numeric':lambda v:re.fullmatch(r'-\d+(?:\.\d*)?%?',v['reference']) is not None,
        'decimal_numeric':lambda v:re.fullmatch(r'[+-]?\d+\.\d+%?',v['reference']) is not None,
        'long_text':lambda v:v['longer_than_CTC_32'], 'vertical_geometry':lambda v:v['geometry_vertical'],
        'unseen_transcription':lambda v:v['reference'] not in vocabulary,
    }.items():
        sub=[v for v in cases if predicate(v)]; subsets[name]=text_metrics([v['reference'] for v in sub],[v['prediction'] for v in sub])
    return {'metrics':metrics,'subsets':subsets,'cases':cases,'failures':failures,'complete':not failures,'seconds':time.monotonic()-seconds,'truncations':recognizer.truncations-truncations_before,
            'CTC_maximum_output_positions':32 if recognizer.kind=='ctc' else None,
            'references_exceeding_32_positions':sum(len(v['reference'])+sum(a==b for a,b in zip(v['reference'],v['reference'][1:]))>32 for v in cases),
            'scope':'GT-crop recognition diagnostic; never image-only inference. Numeric-string tags do not imply verified numeric-axis semantics.'}


def detected_ocr(root,records,predictions,recognizer,session=None):
    cases=[]; failures=[]; references=[]; words_all=[]; correct=provided=unknown=matched=0; truncations_before=recognizer.truncations
    for record,p in zip(records,predictions):
        mask=((p['labels']==4)|(p['labels']==5))&(p['scores']>=.25); pred={k:p[k][mask] for k in ['boxes','scores','labels']}
        try:
            if session and session.remaining<60: raise RuntimeError('OCR resource reserve reached')
            with Image.open(Path(root)/record['image_path']) as raw: words=recognizer.predict(raw.convert('RGB'),pred['boxes'].tolist())
            if len(words)!=len(pred['boxes']): raise RuntimeError('Recognizer output count mismatch')
        except Exception as error:
            words=['']*len(pred['boxes']); failures.append({'id':record['id'],'error':str(error)})
        # All predictions are recognized before looking up reference strings/matches.
        gold=trusted_text(record); target={'boxes':torch.tensor([v['box'] for v in gold]).reshape(-1,4),'labels':torch.tensor([v['label'] for v in gold],dtype=torch.long)}
        matches,_=match_nodes(pred,target,.5); metrics=supported_text_metrics(words,[v['text'] for v in gold],matches.tolist())
        mapping={int(g):word for g,word in zip(matches,words) if int(g)>=0}
        references.extend(v['text'] for v in gold); words_all.extend(mapping.get(i,'') for i in range(len(gold)))
        correct+=sum(mapping.get(i)==v['text'] for i,v in enumerate(gold)); provided+=len(gold); unknown+=int((matches<0).sum()); matched+=len(mapping)
        cases.append({'id':record['id'],'boxes':pred['boxes'].tolist(),'predicted_strings':words,'scoring_only_references':[v['text'] for v in gold],'scoring_only_matches':matches.tolist(),'metrics':metrics})
    summary=text_metrics(references,words_all); summary.update({'supported_exact_region_recall':correct/provided if provided else None,'provided_regions':provided,'matched_regions':matched,
        'unknown_unmatched_predictions':unknown,'precision':None,'F1':None,'scope':'Supported real-text recovery only; missed provided text is deletion, unmatched real predictions unknown'})
    return {'metrics':summary,'cases':cases,'failures':failures,'complete':not failures,'truncations':recognizer.truncations-truncations_before}
