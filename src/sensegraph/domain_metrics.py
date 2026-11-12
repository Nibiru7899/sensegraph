"""Comparable complete-data AP and partial-label real/OCR scoring."""
from collections import Counter,defaultdict
import unicodedata
import numpy as np
import torch
from torchvision.ops import box_iou
from .perception import match_nodes,CLASSES,detection_target
from .ocr_visual import edit_distance


def complete_metrics(predictions,targets,iou=.5,threshold=.25):
    matches=[match_nodes(p,t,iou)[0] for p,t in zip(predictions,targets)]; result={}
    for label,name in enumerate(CLASSES[1:],1):
        scored=[]; total=sum(int((t['labels']==label).sum()) for t in targets); errors=[]
        for p,t,m in zip(predictions,targets,matches):
            for index in torch.where(p['labels']==label)[0].tolist():
                g=int(m[index]); score=float(p['scores'][index]); scored.append((score,g>=0))
                if g>=0 and score>=threshold:
                    a,b=p['boxes'][index],t['boxes'][g]; errors.append(float(((a[:2]+a[2:]-b[:2]-b[2:])/2).norm()/(b[2:]-b[:2]).norm().clamp_min(1e-8)))
        scored.sort(key=lambda x:-x[0]); y=np.asarray([int(v[1]) for v in scored]); tp=np.cumsum(y); fp=np.cumsum(1-y)
        precision=tp/np.maximum(1,tp+fp); recall=tp/max(1,total); envelope=np.maximum.accumulate(precision[::-1])[::-1]
        ap=float(np.sum(np.diff(np.r_[0.,recall])*envelope)) if total else None
        accepted=[v for v in scored if v[0]>=threshold]; found=sum(v[1] for v in accepted)
        result[name]={'support':total,'tp':found,'fp':len(accepted)-found,'fn':total-found,'precision':found/max(1,len(accepted)),'recall':found/max(1,total),'ap':ap,'normalized_center_error':float(np.mean(errors)) if errors else None}
    return {'per_class':result,'macro_ap':float(np.mean([r['ap'] for r in result.values() if r['ap'] is not None])) if any(r['support'] for r in result.values()) else None,
            'macro_recall':float(np.mean([r['recall'] for r in result.values() if r['support']])) if any(r['support'] for r in result.values()) else None}


def partial_counts(pred,record,threshold=.25):
    from .domain_mapping import supported_record
    supported,exclusions=supported_record(record)
    target,ids,rejected=detection_target(supported); p={k:v[pred['scores']>=threshold] for k,v in pred.items() if k in ['boxes','scores','labels']}
    matching,overlap=match_nodes(p,target,.5); found={int(g):i for i,g in enumerate(matching) if g>=0}; classes={}; rows=[]
    for i,(box,label) in enumerate(zip(target['boxes'],target['labels'])):
        name=CLASSES[int(label)]; row=classes.setdefault(name,{'support':0,'recovered':0}); row['support']+=1; row['recovered']+=i in found
        typed=p['labels']==label; best_typed=float(overlap[typed,i].max()) if typed.any() else 0.; best_any=float(overlap[:,i].max()) if len(p['boxes']) else 0.
        reason='recovered' if i in found else 'class_confusion' if best_any>=.5 else 'localization' if best_typed>=.1 else 'detection_miss'
        width,height=(box[2:]-box[:2]).tolist()
        rows.append({'node':ids[i],'class':name,'reason':reason,'bbox':box.tolist(),'area':width*height,'aspect':max(width,height)/max(1e-8,min(width,height)),
                     'thin_under4px':min(width,height)<4,'tiny_under1024':width*height<1024,'best_typed_iou':best_typed,'best_any_iou':best_any,
                     'matched_iou':float(overlap[found[i],i]) if i in found else None,
                     'matched_center_error_px':float(((p['boxes'][found[i],:2]+p['boxes'][found[i],2:]-box[:2]-box[2:])/2).norm()) if i in found else None})
    return {'id':record['id'],'source_chart_type':record['chart_type'],'classes':classes,'objects':rows,'rejected':rejected,'coverage_exclusions':exclusions,'proposals':len(p['boxes']),
            'unmatched_predictions_unknown':int((matching<0).sum()),'provided_nodes':len(ids),'recovered':len(found),
            'resolution':{'width':record['width'],'height':record['height']},
            'strata':{'orientation':'horizontal' if record['chart_type']=='h_bar' else 'vertical' if record['chart_type']=='v_bar' else 'line',
                      'resolution':'native_long_side_under640' if max(record['width'],record['height'])<640 else 'native_long_side_at_least640',
                      'supplied_label_density':'supplied_nodes_under50' if len(ids)<50 else 'supplied_nodes_at_least50',
                      'typography':'font_unavailable','line_thickness':'stroke_width_unavailable',
                      'legend_layout':'legend_present' if any(n['type']=='legend_item' and n.get('bbox') for n in record['nodes']) else 'legend_not_supplied'}}


def summarize_partial(cases):
    counts={}; reasons=Counter(); groups=defaultdict(lambda:Counter())
    for case in cases:
        for name,c in case['classes'].items():
            row=counts.setdefault(name,{'support':0,'recovered':0}); row['support']+=c['support']; row['recovered']+=c['recovered']
        for obj in case['objects']:
            reasons[obj['reason']]+=1
            for key in [case['source_chart_type'],obj['class'],'thin' if obj['thin_under4px'] else 'nonthin','tiny' if obj['tiny_under1024'] else 'nontiny',
                        'aspect_ge5' if obj['aspect']>=5 else 'aspect_lt5',*case.get('strata',{}).values()]:
                groups[key]['support']+=1; groups[key]['recovered']+=obj['reason']=='recovered'
    for row in counts.values(): row['recall50']=row['recovered']/row['support']
    return {'images':len(cases),'per_class':counts,'macro_supported_recall50':float(np.mean([v['recall50'] for v in counts.values()])) if counts else None,
            'micro_supported_recall50':sum(v['recovered'] for v in counts.values())/max(1,sum(v['support'] for v in counts.values())),
            'mean_proposals':sum(c['proposals'] for c in cases)/max(1,len(cases)),'unmatched_predictions_unknown':sum(c['unmatched_predictions_unknown'] for c in cases),
            'failure_categories':dict(reasons),'groups':{k:{**dict(v),'recall50':v['recovered']/v['support']} for k,v in groups.items()},
            'matched_localization':{name:{'matches':sum(obj['class']==name and obj.get('matched_iou') is not None for c in cases for obj in c['objects']),
                'mean_iou':float(np.mean([obj['matched_iou'] for c in cases for obj in c['objects'] if obj['class']==name and obj.get('matched_iou') is not None])) if any(obj['class']==name and obj.get('matched_iou') is not None for c in cases for obj in c['objects']) else None} for name in counts},
            'AP':None,'precision':None,'F1':None,'point_tick_accuracy':None,'coverage':'partial source labels; unmatched predictions are unknown, not false positives'}


def token_edits(reference,prediction):
    a=reference.split(); b=prediction.split(); dp=[[(0,0,0,0)]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1,len(a)+1): dp[i][0]=(0,0,i,0)
    for j in range(1,len(b)+1): dp[0][j]=(0,0,0,j)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            c,s,d,ins=dp[i-1][j-1]; diagonal=(c+int(a[i-1]==b[j-1]),s+int(a[i-1]!=b[j-1]),d,ins)
            c,s,d,ins=dp[i-1][j]; delete=(c,s,d+1,ins)
            c,s,d,ins=dp[i][j-1]; insert=(c,s,d,ins+1)
            dp[i][j]=min([diagonal,delete,insert],key=lambda row:(sum(row[1:]),-row[0]))
    return dp[-1][-1]


def text_metrics(references,predictions):
    if len(references)!=len(predictions): raise ValueError('OCR alignment mismatch')
    refs=[unicodedata.normalize('NFC',s) for s in references]; preds=[unicodedata.normalize('NFC',s) for s in predictions]
    characters=sum(map(len,refs)); errors=sum(edit_distance(a,b) for a,b in zip(refs,preds)); words=sum(len(s.split()) for s in refs)
    edits=np.asarray([token_edits(a,b) for a,b in zip(refs,preds)],dtype=int).sum(0) if refs else np.zeros(4,dtype=int)
    return {'regions':len(refs),'characters':characters,'character_errors':errors,'CER':errors/characters if characters else None,
            'exact_region_accuracy':sum(a==b for a,b in zip(refs,preds))/len(refs) if refs else None,
            'reference_words':words,'correct_words':int(edits[0]),'substitutions':int(edits[1]),'deletions':int(edits[2]),'insertions':int(edits[3]),
            'word_recovery_accuracy':int(edits[0])/words if words else None,'WER':int(edits[1:].sum())/words if words else None,
            'normalization':'NFC only; case, punctuation, sign and decimal differences retained'}


def supported_text_metrics(words,gold,matches):
    if len(words)!=len(matches): raise ValueError('OCR prediction/match length mismatch')
    used={}; unknown=[]
    for word,g in zip(words,matches):
        if g>=0:
            if g in used or g>=len(gold): raise ValueError('Invalid duplicate OCR match')
            used[g]=word
        else: unknown.append(word)
    result=text_metrics(gold,[used.get(i,'') for i in range(len(gold))]); result.update({'matched_regions':len(used),'missing_provided_regions':len(gold)-len(used),
        'supported_exact_region_recall':sum(used.get(i)==text for i,text in enumerate(gold))/len(gold) if gold else None,
        'unmatched_predictions_unknown':len(unknown),'precision':None,'F1':None,'scope':'Provided-text recovery only; unmatched real predictions remain unknown'})
    return result


def select_checkpoint(rows,reference_synthetic,drop=.05):
    eligible=[r for r in rows if r['synthetic_ap50']>=reference_synthetic-drop and r.get('complete',True)]
    return max(eligible,key=lambda r:(r['real_recall'],-r['epoch'])) if eligible else None


def admission(profile,remaining,evaluation_reserve=1200):
    required=2*5*profile['projected_epoch_seconds']+evaluation_reserve
    return {'admitted':required<=remaining and 2*5*profile['projected_epoch_seconds']<=3300,'projected_required_seconds':required,'available_seconds':remaining,'evaluation_reserve_seconds':evaluation_reserve}


def bootstrap_real(a,b,samples=2000,seed=20261110):
    if [r['id'] for r in a]!=[r['id'] for r in b]: raise ValueError('Unpaired inventories')
    groups=defaultdict(list)
    for i,r in enumerate(a): groups[r['source_chart_type']].append(i)
    rng=np.random.default_rng(seed); deltas=[]
    for _ in range(samples):
        idx=np.concatenate([rng.choice(indices,len(indices),replace=True) for indices in groups.values()])
        deltas.append(summarize_partial([b[i] for i in idx])['macro_supported_recall50']-summarize_partial([a[i] for i in idx])['macro_supported_recall50'])
    return {'difference':summarize_partial(b)['macro_supported_recall50']-summarize_partial(a)['macro_supported_recall50'],'ci95':np.quantile(deltas,[.025,.975]).tolist(),
            'samples':samples,'scope':'Chart-type-stratified paired chart bootstrap, conditional on one training seed and this source inventory'}
