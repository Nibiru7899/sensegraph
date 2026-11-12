"""Frozen legacy graph-model transfer; no GAT/MLP training or threshold fitting."""
from pathlib import Path
import sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,read_jsonl,write_json,sha256
from sensegraph.graph_data import observable_graph
from sensegraph.perception import image_tensor,detection_target,match_nodes,relation_recovery
from sensegraph.relation_models import RelationInputs,message_graph,rule_scores
from sensegraph.relation_training import load_checkpoint
from sensegraph.relation_metrics import f1_from_counts,paired_bootstrap
from sensegraph.synthetic import RELATIONS
from sensegraph.quality_budget import QualitySession


def summarize(counts):
    array=np.asarray(counts); total=array.sum(0); macro,micro=f1_from_counts(total)
    return {'macro_f1':macro,'micro_f1':micro,'per_class':{r:{'tp':int(c[0]),'fp':int(c[1]),'fn':int(c[2]),'f1':2*int(c[0])/max(1,int(2*c[0]+c[1]+c[2]))} for r,c in zip(RELATIONS,total)}}


def main():
    torch.set_num_threads(1); base=ROOT/'experiments/visual_training'; d=base/'graph_transfer'; holdout=base/'independent_holdout'
    if (d/'attempt.json').exists(): raise RuntimeError('Frozen graph audit already attempted; preserve it')
    if not read_json(holdout/'results.json')['complete']: raise RuntimeError('Finish independent visual audit first')
    selected=read_json(holdout/'selection.json')['selected']; records=read_jsonl(ROOT/'data/synthetic/perception_holdout_v1/test.jsonl')
    predictions=torch.load(holdout/f'{selected}_predictions.pt',weights_only=True,map_location='cpu')['predictions']
    specs=[]
    for architecture in ['gat','mlp']:
        for seed in [42,43,44]:
            run=ROOT/f'experiments/milestone3/runs/v3_{architecture}_main_s{seed}'; threshold=read_json(run/'thresholds.json'); checkpoint=run/'checkpoint.pt'
            assert sha256(checkpoint)==threshold['checkpoint_sha256']
            specs.append({'architecture':architecture,'seed':seed,'checkpoint':str(checkpoint.relative_to(ROOT)),'sha256':sha256(checkpoint),'thresholds':threshold['values'],'thresholds_sha256':sha256(run/'thresholds.json')})
    write_json(d/'attempt.json',{'started':time.time(),'scope':'Frozen oracle-trained model TRANSFER only; not new matched graph learning','specs':specs,'modes':['predicted','oracle_diagnostic'],'model_selection_or_threshold_fit':False,'selected_detector_locked':selected,'script_sha256':sha256(Path(__file__))})
    # CPU feature preparation contains pixels/proposals only; evaluation mapping stored separately.
    caches={mode:[] for mode in ['predicted','oracle_diagnostic']}; mappings={mode:[] for mode in caches}; coverage=[]
    for record,p in zip(records,predictions):
        image=image_tensor(ROOT/record['image_path']); active=p['scores']>=.25; proposal={k:p[k][active] for k in ['boxes','scores','labels']}
        target,ids,_=detection_target(record); match,_=match_nodes(proposal,target,.5); mapping={i:ids[int(g)] if int(g)>=0 else None for i,g in enumerate(match)}
        coverage.append({'id':record['id'],'predicted_nodes':len(match),'matched_nodes':sum(v is not None for v in mapping.values()),'all_gt_nodes':len(record['nodes'])})
        for mode,boxes in [('predicted',proposal['boxes'].tolist()),('oracle_diagnostic',[n['bbox'] for n in record['nodes']])]:
            graph=observable_graph(image,boxes); caches[mode].append({'x':graph.x,'candidate':graph.edge_index,'messages':message_graph(graph.x)})
        mappings['predicted'].append(mapping); mappings['oracle_diagnostic'].append({i:n['id'] for i,n in enumerate(record['nodes'])})
    torch.save(caches,d/'observable_features.pt'); write_json(d/'evaluation_only_node_maps.json',mappings)
    prior=base/'ocr_holdout/gpu_budget.json'
    if not prior.exists(): prior=base/'ocr_v1/gpu_budget.json'
    if not prior.exists(): prior=holdout/'gpu_budget.json'
    all_counts={}; reports={}; complete=True
    with QualitySession(d/'gpu_budget.json',prior,600,min_seconds=180) as session:
        for spec in specs:
            model,saved=load_checkpoint(ROOT/spec['checkpoint'],device='cuda'); model.eval()
            assert saved['variant']['mode']=='all' and saved['variant']['connectivity']=='spatial' and saved['config']['spatial_neighbors']==6
            assert saved['variant']['architecture']==spec['architecture'] and saved['seed']==spec['seed']
            for mode in caches:
                started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
                end_counts=[]; conditional_counts=[]; raw=[]; offsets=[0]; failures=[]; numeric_conflicts=0; evaluated_pairs=0
                for i,(record,features) in enumerate(zip(records,caches[mode])):
                    edges=[]
                    if session.remaining>=20:
                        inputs=RelationInputs(features['x'].cuda(),features['candidate'].cuda(),features['messages'].cuda())
                        with torch.inference_mode(): scores=torch.sigmoid(model(inputs)).cpu().numpy() if len(features['x']) else np.empty((0,10),dtype=np.float32)
                        rows,classes=np.where(scores>=np.asarray(spec['thresholds']))
                        numeric=[RELATIONS.index(r) for r in ['greater_than','less_than','equal_to']]
                        numeric_conflicts+=int(((scores[:,numeric]>=np.asarray(spec['thresholds'])[numeric]).sum(1)>1).sum()); evaluated_pairs+=len(scores)
                        pairs=features['candidate'].t().numpy(); edges=[(int(pairs[row,0]),int(pairs[row,1]),RELATIONS[int(c)]) for row,c in zip(rows,classes)]
                        raw.append(scores); offsets.append(offsets[-1]+len(scores))
                    else:
                        failures.append(record['id']); complete=False; raw.append(np.full((features['candidate'].shape[1],10),np.nan,dtype=np.float32)); offsets.append(offsets[-1]+features['candidate'].shape[1])
                    scored=relation_recovery(record,edges,mappings[mode][i]); end_counts.append([[scored[r]['tp'],scored[r]['fp'],scored[r]['fn'],0] for r in RELATIONS]); conditional_counts.append([[scored[r]['tp'],scored[r]['conditional_fp'],scored[r]['conditional_fn'],0] for r in RELATIONS])
                name=f"{spec['architecture']}_s{spec['seed']}_{mode}"; all_counts[name]={'end':end_counts,'conditional':conditional_counts}
                reports[name]={'end_to_end':summarize(end_counts),'conditional_matched_nodes':summarize(conditional_counts),'failures':failures,'complete':not failures,
                               'numeric_mutual_exclusion_violations':{'contradictory_pairs':numeric_conflicts,'evaluated_pairs':evaluated_pairs,'fraction':numeric_conflicts/max(1,evaluated_pairs),'definition':'Two or more of greater_than/less_than/equal_to simultaneously positive; diagnostic only, predictions unchanged'},
                               'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                np.savez_compressed(d/f'{name}_scores.npz',scores=np.concatenate(raw),offsets=np.asarray(offsets))
                print({'run':name,'macro_F1':reports[name]['end_to_end']['macro_f1'],'conditional_macro_F1':reports[name]['conditional_matched_nodes']['macro_f1']},flush=True)
            del model; torch.cuda.empty_cache()
        # Rule baseline uses the same observed11 features; no threshold calibration here.
        for mode in caches:
            counts=[]
            for i,(record,features) in enumerate(zip(records,caches[mode])):
                inputs=RelationInputs(features['x'],features['candidate'],features['messages']); scores=rule_scores(inputs).numpy(); rows,classes=np.where(scores>=.5); pairs=features['candidate'].t().numpy()
                edges=[(int(pairs[row,0]),int(pairs[row,1]),RELATIONS[int(c)]) for row,c in zip(rows,classes)]; scored=relation_recovery(record,edges,mappings[mode][i]); counts.append([[scored[r]['tp'],scored[r]['fp'],scored[r]['fn'],0] for r in RELATIONS])
            reports['rule_'+mode]={'end_to_end':summarize(counts),'threshold':.5,'scope':'fixed rules, not recalibrated on holdout'}
    comparisons={}
    if complete:
        for mode in caches:
            for scope in ['end','conditional']:
                gat=np.asarray([all_counts[f'gat_s{s}_{mode}'][scope] for s in [42,43,44]]); mlp=np.asarray([all_counts[f'mlp_s{s}_{mode}'][scope] for s in [42,43,44]])
                comparisons[f'{mode}_{scope}']=paired_bootstrap(gat,mlp,samples=2000,seed=20261107)
    write_json(d/'per_chart_counts.json',all_counts); write_json(d/'results.json',{'complete':complete,'runs':reports,'paired_chart_intervals':comparisons,'coverage':coverage,
               'limits':'All10 relations visible, including irrecoverable series-aggregate endpoints in end-to-end FN. Conditional F1 is a different universe. Oracle mode explicitly uses GT proposals as a diagnostic, never deployment. Legacy models use geometry/RGB only, no new visual embeddings/OCR, no graph retraining. Bootstrap conditions on these frozen models/detector/synthetic charts, not real domains or new graph-training uncertainty.'})


if __name__=='__main__': main()
