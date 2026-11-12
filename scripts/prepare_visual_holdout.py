"""Freeze and render an independent audit-only synthetic holdout, before inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,read_jsonl,write_json,write_jsonl,sha256
from sensegraph.synthetic import make_spec,render_chart,fingerprint
from sensegraph.structural_validation import validate_record
from matplotlib.font_manager import findfont


def main():
    d=ROOT/'experiments/visual_training/independent_holdout'; output=ROOT/'data/synthetic/perception_holdout_v1'
    if (d/'selection.json').exists(): raise RuntimeError('Holdout already evaluated/selected; no regeneration')
    if output.exists() and any(output.iterdir()): raise RuntimeError('Holdout generation already started; preserve it, do not overwrite')
    cfg=read_json(ROOT/'configs/structural_v3.json'); cfg.update({'seed':61720261104,'count':300,'output':'data/synthetic/perception_holdout_v1','splits':{'test':300}})
    fonts=['Arial','Times New Roman','Calibri','Arial']
    palettes=[['#2C6EAA','#CF6C25','#59813B'],['#703D8E','#217C78','#B34A4A'],['#A45A2A','#377B91','#8063A6'],['#286E66','#B04D6D','#8A7135']]
    templates=[]
    for i,font in enumerate(fonts):
        findfont(font,fallback_to_default=False)
        templates.append({'id':f'independent_font_bundle_{i}','font':font,'palette':palettes[i],'background':['#ffffff','#f6f3ec','#f1f6f7','#fffaf2'][i],
                          'grid':i%2==0,'layout':[[.22,.24,.72,.60],[.21,.25,.73,.59],[.23,.23,.70,.61],[.20,.26,.74,.58]][i],'rotation':[0,15,35,10][i]})
    cfg['templates']={'test':templates}
    source_records=[r for version in ['v2','v3'] for split in ['train','val','test'] for r in read_jsonl(ROOT/f'data/synthetic/{version}/{split}.jsonl')]
    used={fingerprint(r['spec']['data']['values']) for r in source_records}; source_images={r['image_sha256'] for r in source_records}
    write_json(d/'generation_freeze.json',{'config':cfg,'config_sha256':fingerprint(cfg),'purpose':'Independent audit only; never train/tune/select from its results',
               'selection_rule':'Lock highest complete300-chart validation AP50 across v2/v3 BEFORE holdout model inference; exact ties prefer corrected v3. Evaluate both once for comparison without changing selection.',
               'no_model_results_used':True,'balancing':'3 chart types x4 style bundles; each combination25 charts.4 and3 cycles are coprime to avoid type/style coupling.',
               'uniqueness_recipe':'Reject exact preexisting numerical arrays by deterministic seed+attempt*1000000, max1000 attempts per index, before rendering. Reject image identity collision. No accuracy-conditioned regeneration.',
               'generator_sha256':sha256(ROOT/'src/sensegraph/synthetic.py'),'script_sha256':sha256(Path(__file__))})
    records=[]; rejection_log=[]
    for i in range(300):
        for attempt in range(1000):
            spec=make_spec({**cfg,'seed':cfg['seed']+attempt*1000000},i); key=fingerprint(spec['data']['values'])
            if key not in used: break
        else: raise RuntimeError('Unable to acquire independent numerical configuration')
        used.add(key); spec['id']=f'perception-holdout-v1-{i:05d}'
        if attempt: rejection_log.append({'index':i,'attempts':attempt})
        record=render_chart(spec,output/'images'/f"{spec['id']}.png",ROOT); errors=validate_record(record,ROOT)
        if errors: raise ValueError(f"Invalid independent holdout {record['id']}: {errors}")
        if record['image_sha256'] in source_images: raise RuntimeError('Independent holdout exact image collision; no silent replacement')
        source_images.add(record['image_sha256']); records.append(record); write_json(output/'annotations'/f"{record['id']}.json",record)
    write_jsonl(output/'test.jsonl',records)
    write_json(output/'manifest.json',{'config':cfg,'records':[{'id':r['id'],'image_sha256':r['image_sha256'],'data_hash':r['spec']['data_hash'],'template_hash':r['spec']['template_hash']} for r in records],
               'test_manifest_sha256':sha256(output/'test.jsonl'),'numeric_uniqueness_rejections':rejection_log,'training_use_forbidden':True})
    write_json(d/'generation_complete.json',{'images':300,'chart_type_counts':{t:sum(r['chart_type']==t for r in records) for t in cfg['chart_types']},'invalid':0,'existing_numerical_array_reuse':0,'exact_image_duplicates':0,'manifest_sha256':sha256(output/'test.jsonl')})
    print('Frozen independent holdout:300 validated charts, new fonts/styles, no exact data/image overlap.')


if __name__=='__main__': main()
