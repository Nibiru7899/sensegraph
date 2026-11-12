"""CPU provenance/leakage/target audit; no protected-test model evaluation."""
from pathlib import Path
from collections import Counter
import ast,sys
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,read_jsonl,sha256,write_json
from sensegraph.leakage import leakage_report
from sensegraph.synthetic import fingerprint
from sensegraph.full_visual_training import verify
from sensegraph.visual_training import targets


def main():
    d=ROOT/'experiments/visual_training/v2'; integrity=verify(ROOT)
    records=[]; sources={}
    for split in ['train','val','test']:
        # Test read only for split/provenance fingerprints, never inference or tuning.
        path=ROOT/f'data/synthetic/v3/{split}.jsonl'; part=read_jsonl(path); sources[split]={'count':len(part),'manifest_sha256':sha256(path)}
        assert all(r['split']==split for r in part)
        records.extend(part)
    actual_mismatches=[r['id'] for r in records if sha256(ROOT/r['image_path'])!=r['image_sha256']]
    assert not actual_mismatches
    leakage=leakage_report(records,ROOT)
    assert not leakage['violations'] and not leakage['exact_duplicates']
    v2_values={fingerprint(r['spec']['data']['values']) for split in ['train','val','test'] for r in read_jsonl(ROOT/f'data/synthetic/v2/{split}.jsonl')}
    reused=[r['id'] for r in records if fingerprint(r['spec']['data']['values']) in v2_values]
    source=ast.parse((ROOT/'src/sensegraph/full_visual_training.py').read_text())
    dataset_splits=[node.args[1].value for node in ast.walk(source) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='VisualDataset' and isinstance(node.args[1],ast.Constant)]
    assert set(dataset_splits)=={'train','val'}
    encoder_calls=[node for node in ast.walk(source) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='encode_image']
    assert all(len(c.args)==1 and not c.keywords for c in encoder_calls)
    order_issues=[]
    for record in records:
        if record['split']=='test': continue
        w,h=record['width'],record['height']; scale=min(640/w,640/h); nw,nh=round(w*scale),round(h*scale)
        affine=torch.tensor([nw/w,nh/h,(640-nw)//2,(640-nh)//2,w,h]); a=targets(record,affine)
        reversed_record={**record,'nodes':list(reversed(record['nodes']))}; b=targets(reversed_record,affine)
        difference=max(float((a[name]-b[name]).abs().max()) for name in ['box_heat','key_heat'])
        if difference: order_issues.append({'id':record['id'],'split':record['split'],'max_heatmap_difference':difference})
    result={'historical_integrity':integrity,'sources':sources,'actual_image_hash_mismatches':actual_mismatches,
            'leakage':leakage,'v2_numeric_array_reuse':reused,'model_reader_splits':dataset_splits,'encoder_calls_accept_image_only':len(encoder_calls),
            'annotation_order_heatmap_dependence':order_issues,
            'test_use':'Read only for membership/image/data/template fingerprint audit; no test inference, metrics, architecture choice or tuning.',
            'limits':['Static and fixture audit cannot prove absence of all leakage.','Validation has been reused for development and checkpoint selection; it is not an untouched final evaluation.','Perceptual hash candidates are not proof of shared source.','Coincident segment-center Gaussian radius currently uses first colliding instance; any order dependence is reported rather than silently changing a frozen run.','Pretrained data membership and real cross-dataset upstream overlap are not fully auditable locally.']}
    write_json(d/'integrity_audit.json',result)
    print({'hash_mismatches':len(actual_mismatches),'split_violations':len(leakage['violations']),'exact_duplicates':len(leakage['exact_duplicates']),'near_hash_candidates':len(leakage['near_duplicates']),'v2_reuse':len(reused),'order_dependent_heatmaps':len(order_issues)})


if __name__=='__main__': main()
