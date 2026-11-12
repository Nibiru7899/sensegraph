"""Independent holdout identity/near-overlap audit; no model inference or tuning."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,read_jsonl,sha256,write_json
from sensegraph.synthetic import fingerprint
from sensegraph.leakage import difference_hash


def main():
    d=ROOT/'experiments/visual_training/independent_holdout'; manifest=ROOT/'data/synthetic/perception_holdout_v1/test.jsonl'
    generated=read_json(d/'generation_complete.json'); assert sha256(manifest)==generated['manifest_sha256']
    heldout=read_jsonl(manifest); old=[(version,r) for version in ['v2','v3'] for split in ['train','val','test'] for r in read_jsonl(ROOT/f'data/synthetic/{version}/{split}.jsonl')]
    old_numeric={fingerprint(r['spec']['data']['values']) for _,r in old}; old_templates={r['spec']['template_hash'] for _,r in old}; old_hashes={r['image_sha256'] for _,r in old}
    near=[]; old_perceptual=[(version,r['id'],r['split'],difference_hash(ROOT/r['image_path'])) for version,r in old]
    actual_mismatches=[]; reused=[]; templates=[]; exact=[]; balance=Counter()
    for r in heldout:
        if sha256(ROOT/r['image_path'])!=r['image_sha256']: actual_mismatches.append(r['id'])
        if fingerprint(r['spec']['data']['values']) in old_numeric: reused.append(r['id'])
        if r['spec']['template_hash'] in old_templates: templates.append(r['id'])
        if r['image_sha256'] in old_hashes: exact.append(r['id'])
        balance[(r['chart_type'],r['spec']['template']['id'])]+=1
        h=difference_hash(ROOT/r['image_path'])
        for version,identity,split,other in old_perceptual:
            distance=(h^other).bit_count()
            if distance<=4: near.append({'holdout':r['id'],'source_version':version,'source_id':identity,'source_split':split,'distance':distance})
    assert not actual_mismatches and not reused and not templates and not exact and all(n==25 for n in balance.values())
    write_json(d/'leakage_audit.json',{'images':len(heldout),'image_hash_mismatches':actual_mismatches,'existing_numeric_reuse':reused,'existing_template_reuse':templates,'exact_image_duplicates':exact,'type_style_counts':{str(k):v for k,v in balance.items()},'near_hash_candidates':near,
               'limits':'64-bit grayscale differences are only candidate overlap indicators. No near-hash sample is removed based on appearance or model results. Synthetic renderer primitives and numerical support are intentionally shared.'})
    print({'images':len(heldout),'identity_violations':0,'near_hash_candidates':len(near),'balanced_type_style_cells':len(balance)})


if __name__=='__main__': main()
