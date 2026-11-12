"""CPU annotation/overlap audits and contact sheets; never infer label completeness."""
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image,ImageDraw
import numpy as np
import torch
from .io import read_json,read_jsonl,write_json,sha256
from .domain_data import annotation_overlay
from .domain_sources import dhash
from .visual_training import preprocess,transform_boxes,TYPE_CLASS
from .perception import detection_target
from .domain_mapping import supported_record


def contact_sheets(paths,destination,columns=4,rows=3):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True); pages=[]
    for start in range(0,len(paths),columns*rows):
        canvas=Image.new('RGB',(columns*320,rows*260),'#dddddd'); draw=ImageDraw.Draw(canvas)
        for j,path in enumerate(paths[start:start+columns*rows]):
            with Image.open(path) as raw:
                image=raw.convert('RGB'); image.thumbnail((316,232))
            x=j%columns*320; y=j//columns*260; canvas.paste(image,(x+(320-image.width)//2,y+22)); draw.text((x+4,y+3),Path(path).stem,fill='black')
        path=destination/f'page_{start//(columns*rows):03d}.png'; canvas.save(path); pages.append(str(path))
    return pages


def inspect(root,cfg):
    root=Path(root); d=root/cfg['experiment']; rows=read_jsonl(root/cfg['synthetic_output']/'train.jsonl')+read_jsonl(root/cfg['synthetic_output']/'val.jsonl')
    anomalies=[]; counts=Counter(); recognition=Counter(); colliding=[]; overlays=[]
    for record in rows:
        _,affine=preprocess(root/record['image_path'],640); cells=defaultdict(list)
        for node in record['nodes']:
            counts[f"{record['split']}:{node['type']}"]+=1
            if 'recognition_supervision' in node: recognition[node['recognition_supervision']]+=1
            center=transform_boxes(torch.tensor(node['bbox']),affine).reshape(2,2).mean(0)/4
            cells[(TYPE_CLASS[node['type']],*center.floor().long().tolist())].append({'id':node['id'],'center':center.tolist()})
        groups=[{'cell':list(key),'nodes':nodes,'coincident_to_1e-5':bool(np.max(np.ptp(np.asarray([n['center'] for n in nodes]),axis=0))<1e-5)} for key,nodes in cells.items() if len(nodes)>1]
        if groups:
            colliding.append({'id':record['id'],'split':record['split'],'groups':groups})
        ambiguous=[n['id'] for n in record['nodes'] if n.get('recognition_supervision')=='ambiguous_overlap_or_clip']
        if groups or ambiguous or record['unavailable_artists']:
            anomalies.append({'id':record['id'],'ambiguous_text':ambiguous,'unavailable_artists':record['unavailable_artists'],'collision_groups':len(groups)})
            path=d/'anomaly_overlays'/f"{record['id']}.png"; annotation_overlay(root,record,path)
            if groups and record['split']=='val': overlays.append(path)
    pages=contact_sheets(overlays,d/'collision_contacts')
    hashed=[(r,dhash(root/r['image_path'])) for r in rows]; near=[]
    for record,value in hashed:
        if record['split']!='train': continue
        for other,ph in hashed:
            if other['split']=='val' and (value^ph).bit_count()<=4: near.append({'train':record['id'],'val':other['id'],'dhash_distance':(value^ph).bit_count()})
    result={'synthetic_images':len(rows),'node_counts':dict(counts),'recognition_supervision':dict(recognition),'anomaly_charts':len(anomalies),
            'anomalies':anomalies,'collision_charts':colliding,'validation_collision_contact_pages':pages,'cross_split_near_hash_candidates':near,
            'near_hash_scope':'Potential visual similarities only; dHash is not proof of shared source. No samples replaced.'}
    realpath=root/cfg['real_output']/'manifest.json'
    if realpath.exists():
        actual=[]; failures=[]; supported=Counter(); unknowntext=0; rawissues=[]
        for split in ['train','val','final']:
            records=read_jsonl(root/cfg['real_output']/f'{split}.jsonl')
            for record in records:
                if sha256(root/record['image_path'])!=record['image_sha256']: failures.append(record['id']+':image_hash')
                annotation=root/cfg['real_output']/'raw/annotations'/f"{record['id']}.json"
                if sha256(annotation)!=record['source_annotation_sha256']: failures.append(record['id']+':annotation_hash')
                filtered,exclusions=supported_record(record)
                target,ids,rejected=detection_target(filtered); supported.update(str(int(v)) for v in target['labels'])
                unknowntext+=sum(n['type'] in ('text','title','axis_label','tick_label','legend_item') and n.get('bbox') is not None and n.get('text') is None for n in record['nodes'])
                rawissues.append({'id':record['id'],'split':split,'issues':record['issues'],'rejected':rejected,'pre_inference_coverage_exclusions':exclusions,'supported_node_ids':ids})
                actual.append({'id':record['id'],'split':split,'group':record['source_group'],'image_sha':record['image_sha256']})
        groups=[v['group'] for v in actual]
        if len(groups)!=len(set(groups)): failures.append('source_group_overlap')
        if len({v['image_sha'] for v in actual})!=len(actual): failures.append('exact_image_overlap')
        result['real']={'images':len(actual),'supported_box_classes':dict(supported),'valid_detection_boxes_without_transcription':unknowntext,
                        'failures':failures,'source_issues':rawissues,'valid':not failures,'coverage':'partial; no negative/exhaustive-background guarantee'}
        result['real_contact_pages']={split:contact_sheets(sorted((d/'real_annotation_overlays'/split).glob('*.png')),d/'real_contacts'/split) for split in ['train','val','final']}
    write_json(d/'data_review.json',result); return {'synthetic_images':len(rows),'anomaly_charts':len(anomalies),'collision_contact_pages':len(pages),'near_hash_candidates':len(near),'real':result.get('real',{}).get('valid')}
