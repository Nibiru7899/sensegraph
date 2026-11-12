"""Pixel-only crop preparation/CTC recognition, with labels separated for scoring."""
from pathlib import Path
import math
import hashlib
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from .io import read_json,read_jsonl,write_json,sha256

TEXT_TYPES={'title','axis_label','tick_label','legend_item'}


def encode_text(text):
    if not text or any(ord(c)<32 or ord(c)>126 for c in text): raise ValueError('Unsupported or empty OCR transcription')
    return [ord(c)-31 for c in text]


def decode_path(path):
    output=[]; last=None
    for value in path:
        token=int(value)
        if token!=last and token!=0: output.append(chr(token+31))
        last=token
    return ''.join(output)


def crop_pixels(image,box):
    w,h=image.size; left,top=max(0,math.floor(box[0])),max(0,math.floor(box[1])); right,bottom=min(w,math.ceil(box[2])),min(h,math.ceil(box[3]))
    if right<=left or bottom<=top: raise ValueError('Invalid OCR crop')
    crop=image.crop((left,top,right,bottom)).convert('L'); rotated=crop.height>1.5*crop.width
    if rotated: crop=crop.rotate(-90,expand=True)
    scale=min(32/crop.height,128/crop.width); nw,nh=max(1,round(crop.width*scale)),max(1,round(crop.height*scale))
    canvas=Image.new('L',(128,32),255); canvas.paste(crop.resize((nw,nh),Image.Resampling.BILINEAR),(0,(32-nh)//2))
    tensor=torch.from_numpy(np.asarray(canvas).copy())[None]
    return tensor,min(32,math.ceil(nw/4)),rotated


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]


def collect_crops(root,manifest):
    samples=[]; unsupported=[]
    for record in read_jsonl(manifest):
        with Image.open(root/record['image_path']) as source:
            image=source.convert('RGB')
            for node in record['nodes']:
                if node['type'] not in TEXT_TYPES: continue
                try:
                    labels=encode_text(node['text']); crop,length,rotated=crop_pixels(image,node['bbox'])
                except (ValueError,KeyError) as error:
                    unsupported.append({'chart':record['id'],'node':node['id'],'reason':str(error)}); continue
                needed=len(labels)+sum(a==b for a,b in zip(labels,labels[1:]))
                samples.append({'pixels':crop,'length':length,'text':node['text'],'labels':labels,'feasible':needed<=length,
                    'chart':record['id'],'node':node['id'],'type':node['type'],'rotated_by_geometry':rotated,
                    'hash':hashlib.sha256(crop.numpy().tobytes()+bytes([length])).hexdigest()})
    return samples,unsupported


def prepare(root):
    output=root/'data/ocr_visual_v1'; d=root/'experiments/visual_training/ocr_v1'
    if (output/'manifest.json').exists():
        manifest=read_json(output/'manifest.json')
        if manifest['preparation_source_sha256']!=sha256(Path(__file__)): raise RuntimeError('OCR preprocessing changed; create a new cache version')
        if sha256(output/'train.pt')!=manifest['train_cache_sha256'] or sha256(output/'val.pt')!=manifest['val_cache_sha256']: raise RuntimeError('OCR cache identity changed')
        return manifest
    train,bad_train=collect_crops(root,root/'data/synthetic/v3/train.jsonl'); val,bad_val=collect_crops(root,root/'data/synthetic/v3/val.jsonl')
    val_pixels={s['hash'] for s in val}; overlap=[{'chart':s['chart'],'node':s['node']} for s in train if s['hash'] in val_pixels]
    train_kept=[s for s in train if s['feasible'] and s['hash'] not in val_pixels]
    output.mkdir(parents=True,exist_ok=True); torch.save(train_kept,output/'train.pt'); torch.save(val,output/'val.pt')
    manifest={'train_source_charts':1400,'val_source_charts':300,'train_crops_original':len(train),'train_crops':len(train_kept),'val_crops':len(val),
       'cross_split_exact_pixel_crops_removed_from_train':overlap,'infeasible_train':sum(not s['feasible'] for s in train),'infeasible_val':sum(not s['feasible'] for s in val),
       'unsupported_train':bad_train,'unsupported_val':bad_val,'unique_train_texts':len({s['text'] for s in train_kept}),'unique_val_texts':len({s['text'] for s in val}),
       'shared_text_strings':len({s['text'] for s in train_kept}&{s['text'] for s in val}),
       'train_source_sha256':sha256(root/'data/synthetic/v3/train.jsonl'),'val_source_sha256':sha256(root/'data/synthetic/v3/val.jsonl'),
       'train_cache_sha256':sha256(output/'train.pt'),'val_cache_sha256':sha256(output/'val.pt'),'preparation_source_sha256':sha256(Path(__file__)),
       'scope':'GT crops for supervised recognition and conditional diagnostics only; end-to-end recognition must use predicted regions. No holdout source read.'}
    write_json(output/'manifest.json',manifest); write_json(d/'data_manifest.json',manifest); return manifest


def batch(samples,device='cuda'):
    pixels=torch.stack([s['pixels'] for s in samples]).to(device).float()/255
    lengths=torch.tensor([s['length'] for s in samples],dtype=torch.long)
    labels=torch.tensor([c for s in samples for c in s['labels']],dtype=torch.long,device=device)
    target_lengths=torch.tensor([len(s['labels']) for s in samples],dtype=torch.long)
    return pixels,lengths,labels,target_lengths


def ctc_loss(logits,lengths,labels,target_lengths):
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),labels,lengths,target_lengths,blank=0,zero_infinity=False)


@torch.no_grad()
def transcribe(model,samples,device='cuda',batch_size=64):
    model.eval(); strings=[]
    for start in range(0,len(samples),batch_size):
        part=samples[start:start+batch_size]; pixels=torch.stack([s['pixels'] for s in part]).to(device).float()/255
        paths=model(pixels).argmax(-1).cpu()
        strings.extend(decode_path(path[:s['length']]) for path,s in zip(paths,part))
    return strings


def recognition_metrics(samples,strings):
    errors=sum(edit_distance(s['text'],p) for s,p in zip(samples,strings)); chars=sum(len(s['text']) for s in samples)
    result={'crops':len(samples),'character_errors':errors,'characters':chars,'cer':errors/max(1,chars),'exact':sum(s['text']==p for s,p in zip(samples,strings))/max(1,len(samples)),'by_type':{}}
    for kind in TEXT_TYPES:
        pairs=[(s,p) for s,p in zip(samples,strings) if s['type']==kind]
        result['by_type'][kind]={'crops':len(pairs),'cer':sum(edit_distance(s['text'],p) for s,p in pairs)/max(1,sum(len(s['text']) for s,p in pairs)),'exact':sum(s['text']==p for s,p in pairs)/max(1,len(pairs))}
    rotated=[(s,p) for s,p in zip(samples,strings) if s['rotated_by_geometry']]
    result['geometry_rotated']={'crops':len(rotated),'exact':sum(s['text']==p for s,p in rotated)/max(1,len(rotated))}
    return result
