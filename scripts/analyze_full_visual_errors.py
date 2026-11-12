"""Post-selection CPU diagnostics for all validation targets; never tunes the run."""
from pathlib import Path
from collections import Counter
import sys,argparse
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,read_jsonl,write_json
from sensegraph.perception import match_nodes,detection_target


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',choices=['v2','v3'],default='v2'); args=parser.parse_args()
    d=ROOT/f'experiments/visual_training/{args.run}'; history=read_json(d/'history.json')
    best=max(history,key=lambda x:x['validation']['ap50']['macro_ap']); label=f"{best['stage']}_{best['epoch']}"
    saved=torch.load(d/f'{label}_predictions.pt',weights_only=True,map_location='cpu'); records=read_jsonl(ROOT/'data/synthetic/v3/val.jsonl')
    collisions=read_json(ROOT/'experiments/visual_training/v2/collision_analysis.json')['examples']; collision_ids={}
    for example in collisions:
        if example['split']=='val': collision_ids.setdefault(example['id'],set()).update(x[0] for x in example['instances'])
    counts=Counter(); images=[]; scores=[]
    for record,pred,gt in zip(records,saved['predictions'],saved['references']):
        scores.extend(pred['scores'].tolist())
        accepted=pred['scores']>=.25; p={k:v[accepted] for k,v in pred.items() if k in ['boxes','labels','scores']}
        matched,_=match_nodes(p,gt,.5); recovered=set(matched[matched>=0].tolist()); _,identities,rejected=detection_target(record)
        missed=[]
        for i,box in enumerate(gt['boxes']):
            width,height=(box[2:]-box[:2]).tolist(); tiny=width*height<32**2; collided=identities[i] in collision_ids.get(record['id'],set())
            for group,flag in [('all',True),('tiny_native_area_lt1024',tiny),('collision_target',collided),('noncollision_target',not collided)]:
                if flag:
                    counts[group+':gt']+=1; counts[group+':matched']+=i in recovered
            if i not in recovered: missed.append({'id':identities[i],'class':int(gt['labels'][i]),'box':box.tolist(),'tiny':tiny,'collision':collided})
        false=[{'box':p['boxes'][i].tolist(),'class':int(p['labels'][i]),'confidence':float(p['scores'][i])} for i in torch.where(matched<0)[0].tolist()]
        images.append({'id':record['id'],'chart_type':record['chart_type'],'missed_gt':missed,'false_positives':false,'unsupported_or_invalid':rejected})
    summary={group:{'support':counts[group+':gt'],'matched':counts[group+':matched'],'recall':counts[group+':matched']/max(1,counts[group+':gt'])} for group in ['all','tiny_native_area_lt1024','collision_target','noncollision_target']}
    write_json(d/'best_error_analysis.json',{'checkpoint':label,'iou':.5,'confidence':.25,'summary':summary,'images':images,
               'limits':'Tiny means original-image box area<1024, not COCO resized-area AP. Subgroup statistics are GT-recovery recall, not subgroup AP or precision. All false positives remain in main metrics.'})
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(8,4)); FigureCanvasAgg(fig); ax=fig.subplots()
    ax.hist(scores,bins=30,range=(.1,1.)); ax.axvline(.25,color='red',linestyle='--',label='Fixed reporting threshold .25')
    ax.set(xlabel='Detection score (not calibrated probability)',ylabel='Returned detections',title=f'{label}: all300 validation charts'); ax.legend(); fig.tight_layout(); fig.savefig(d/'best_confidence_distribution.png')
    print(summary)


if __name__=='__main__': main()
