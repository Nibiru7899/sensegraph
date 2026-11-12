from pathlib import Path
import copy,hashlib
import numpy as np
import torch
import pytest
from sensegraph.io import read_json,write_jsonl
from sensegraph.domain_data import make_domain_spec,render_domain,validate_domain,DomainDataset
from sensegraph.domain_metrics import complete_metrics,partial_counts,summarize_partial,text_metrics,supported_text_metrics,select_checkpoint,admission
from sensegraph.perception import detection_metrics
from sensegraph.domain_sources import partition_candidates
from sensegraph.domain_mapping import chartqa_partial

CFG=read_json(Path(__file__).resolve().parents[1]/'configs/domain_generalization_v1.json')


def test_horizontal_and_markerless_renderer_metadata(tmp_path):
    for index in [1,4]:
        spec=make_domain_spec(CFG,index,'train')
        if index==4: spec['style']['marker']='none'
        record=render_domain(spec,tmp_path/f'{index}.png',tmp_path)
        assert not validate_domain(record,tmp_path)
        if index==1:
            assert record['axes']['numeric_axis']=='x' and record['axes']['categorical_axis']=='y'
            node=next(n for n in record['nodes'] if n['type']=='bar'); assert node['numeric_value']==node['data_xy'][0]
        else:
            assert not any(n['type']=='point' for n in record['nodes'])
            assert sum(n['type']=='line_segment' for n in record['nodes'])==len(spec['data']['series'])*(len(spec['data']['categories'])-1)
        broken=copy.deepcopy(record); broken['nodes'][0]['bbox'][0]=-1
        assert validate_domain(broken)


def test_styles_titles_and_numeric_configs_are_split_disjoint():
    train=[make_domain_spec(CFG,i,'train') for i in range(100)]; val=[make_domain_spec(CFG,i,'val') for i in range(30)]
    for key in ['title','numeric_hash','style_hash']: assert not({r[key] for r in train}&{r[key] for r in val})
    assert {r['family'] for r in train}==set(r['family'] for r in val)
    assert not({r['style']['font'] for r in train}&{r['style']['font'] for r in val})
    assert all('Synthetic' not in r['title'] for r in train+val)


def test_partial_records_cannot_enter_full_background_training(tmp_path):
    write_jsonl(tmp_path/'real.jsonl',[{'split':'train','annotation_coverage':'partial'}])
    with pytest.raises(ValueError,match='Only complete synthetic'): DomainDataset(tmp_path,'real.jsonl','train')
    with pytest.raises(ValueError): DomainDataset(tmp_path,'real.jsonl','final')


def test_preflight_cannot_truncate_completed_manifests(tmp_path):
    from sensegraph.domain_data import prepare_synthetic
    from sensegraph.io import write_json
    cfg={**CFG,'synthetic_output':'full'}
    write_json(tmp_path/'full/manifest.json',{'complete':True})
    with pytest.raises(ValueError,match='not truncated'): prepare_synthetic(tmp_path,cfg,100)


def test_complete_AP_matches_historical_definition():
    pred={'boxes':torch.tensor([[0.,0.,5.,5.],[0.,0.,5.,5.],[8.,8.,12.,12.]]),'labels':torch.tensor([1,1,4]),'scores':torch.tensor([.9,.8,.7])}
    target={'boxes':torch.tensor([[0.,0.,5.,5.],[8.,8.,12.,12.]]),'labels':torch.tensor([1,4])}
    old=detection_metrics([pred],[target]); new=complete_metrics([pred],[target])
    assert old['macro_ap']==new['macro_ap']
    for name in old['per_class']:
        for key in ['ap','support','tp','fp','fn','precision','recall']: assert old['per_class'][name][key]==new['per_class'][name][key]


def test_partial_recall_does_not_invent_false_positives():
    record={'id':'a','chart_type':'h_bar','width':20,'height':20,'nodes':[{'id':'b','type':'bar','bbox':[0,0,5,5]}]}
    p={'boxes':torch.tensor([[0.,0.,5.,5.],[8.,8.,12.,12.]]),'labels':torch.tensor([1,4]),'scores':torch.tensor([.9,.8])}
    result=summarize_partial([partial_counts(p,record)])
    assert result['micro_supported_recall50']==1 and result['unmatched_predictions_unknown']==1
    assert result['AP'] is None and result['precision'] is None and result['point_tick_accuracy'] is None


def test_word_region_and_partial_ocr_denominators():
    m=text_metrics(['red cat','x'*33],['red dog extra','x'*32])
    assert m['regions']==2 and m['exact_region_accuracy']==0 and m['reference_words']==3
    assert m['correct_words']==1 and m['substitutions']==2 and m['insertions']==1
    a=supported_text_metrics(['12','unlabeled'],['12','missing'],[0,-1]); b=supported_text_metrics(['12'],['12','missing'],[0])
    assert a['CER']==b['CER'] and a['unmatched_predictions_unknown']==1 and a['missing_provided_regions']==1
    assert a['precision'] is None and a['F1'] is None
    assert text_metrics(['-1.00'],['1.00'])['character_errors']==1


def test_checkpoint_guard_and_complete_budget_admission():
    rows=[{'epoch':1,'real_recall':.3,'synthetic_ap50':.6},{'epoch':2,'real_recall':.5,'synthetic_ap50':.4},
          {'epoch':3,'real_recall':.9,'synthetic_ap50':.8,'complete':False}]
    assert select_checkpoint(rows,.6)['epoch']==1
    assert select_checkpoint([{'epoch':2,'real_recall':.3,'synthetic_ap50':.6},*rows],.6)['epoch']==1
    assert admission({'projected_epoch_seconds':100},2300)['admitted']
    assert not admission({'projected_epoch_seconds':100},2100)['admitted']


def test_source_groups_and_prior_near_images_never_cross_splits():
    cfg={**CFG,'real_counts_per_type':{'train':1,'val':1,'final':1}}; records=[]
    for kind in ['h_bar','v_bar','line']:
        for i in range(5):
            key=f'{kind}-{i}'; digest=hashlib.sha256(key.encode()).hexdigest()
            records.append({'id':key,'chart_type':kind,'image_sha256':digest,'dhash':hex(int(digest[:16],16)),'source_keys':{'title':key,'table_hash':digest},'near_prior':['old'] if i==0 else []})
    records.append({**records[1],'id':'same-source-copy'})
    split=partition_candidates(records,cfg); assert split is not None
    assert len({r['source_group'] for rows in split.values() for r in rows})==9
    assert not any(r['near_prior'] for s in ['val','final'] for r in split[s])


def test_partial_mapping_retains_boxes_without_transcription():
    box={'x':2,'y':3,'w':4,'h':5}
    raw={'type':'h_bar','general_figure_info':{'x_axis':{'major_labels':{'bboxes':[box]},'label':{'bbox':[box],'text':'Revenue'}}},
         'models':[{'bboxes':[box],'x':['A'],'y':[2]}]}
    record=chartqa_partial(raw,(30,30)); nodes={n['id']:n for n in record['nodes']}
    assert nodes['x_axis:label:0']['bbox']==[2,3,6,8] and nodes['x_axis:label:0']['text'] is None
    assert nodes['x_axis:description:0']['text']=='Revenue'
    assert nodes['mark:0:0']['type']=='bar'
    from sensegraph.domain_mapping import supported_record
    filtered,excluded=supported_record(record)
    assert len(excluded)==2 and {n['id'] for n in filtered['nodes']}=={'plot_area','series:0','mark:0:0'}
    multi=copy.deepcopy(record); multi['nodes'].append({'id':'series:1','type':'series','bbox':None})
    filtered,excluded=supported_record(multi)
    assert not any(n['type']=='bar' for n in filtered['nodes']) and len(excluded)==3
    raw['type']='line'; record=chartqa_partial(raw,(30,30))
    assert next(n for n in record['nodes'] if n['id']=='mark:0:0')['type']=='unsupported_model_box'
    raw['models'][0].update(x=['A','B'],y=[2,3],bboxes=[{**box,'h':0}])
    record=chartqa_partial(raw,(30,30))
    assert next(n for n in record['nodes'] if n['id']=='mark:0:0')['bbox'] is None
    assert not any(n['type']=='point' for n in record['nodes'])


def test_failed_annotation_gate_blocks_any_fitting_profile(tmp_path):
    from sensegraph.domain_training import fitting_gate
    from sensegraph.io import write_json
    cfg={**CFG,'experiment':'run'}
    write_json(tmp_path/'run/freeze.json',{'training_gate_passed':False})
    with pytest.raises(RuntimeError,match='prohibited'): fitting_gate(tmp_path,cfg)
    assert not (tmp_path/'run/resource_chain.json').exists()


def test_orientation_rule_uses_pixels_not_reference_text():
    from PIL import Image
    from sensegraph.domain_ocr import native_crop
    from sensegraph.ocr_visual import crop_pixels
    image=Image.new('RGB',(40,90),'white')
    crop=native_crop(image,[0,0,20,80]); _,_,rotated=crop_pixels(image,[0,0,20,80])
    assert crop.size==(80,20) and rotated


def test_cpu_fixture_gradients_memorization_and_frozen_BN():
    # Wiring-only optimization of a fabricated 128px fixture; no campaign data/GPU.
    from sensegraph.neural_spec import SenseGraphNet
    from sensegraph.domain_training import optimizer
    from sensegraph.visual_training import targets,detection_loss
    torch.set_num_threads(1); torch.manual_seed(56)
    model=SenseGraphNet(); opt=optimizer(model,CFG); model.train()
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    buffers={n:v.clone() for n,v in model.named_buffers()}
    record={'width':128,'height':128,'nodes':[{'id':'box','type':'bar','bbox':[20,20,55,85]}]}
    target=targets(record,torch.tensor([1.,1.,0.,0.,128.,128.]),128)
    image=torch.rand(1,3,128,128); losses=[]
    for _ in range(6):
        opt.zero_grad(set_to_none=True); _,dense=model.encode_image(image); loss,_=detection_loss(dense,target)
        assert torch.isfinite(loss); losses.append(float(loss.detach())); loss.backward()
        assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    assert losses[-1]<losses[0]
    assert all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen)
    assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers())
