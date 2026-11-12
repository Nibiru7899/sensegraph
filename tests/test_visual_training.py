import torch
import pytest
from sensegraph.visual_training import preprocess,transform_boxes,targets,detection_loss,decode,VisualDataset
from PIL import Image


def record():
    return {'width':100,'height':60,'nodes':[{'id':'bar','type':'bar','bbox':[10,10,30,40]},
           {'id':'tick','type':'tick','bbox':[8,44,9,46]},{'id':'series','type':'series','bbox':[10,10,30,40]}]}


def test_letterbox_roundtrip_small_targets_and_masks(tmp_path):
    path=tmp_path/'chart.png'; Image.new('RGB',(100,60)).save(path)
    image,a=preprocess(path,128); box=torch.tensor([[10.,10.,30.,40.]])
    assert image.shape==(3,128,128)
    assert torch.allclose(transform_boxes(transform_boxes(box,a),a,True),box,atol=1e-5)
    target=targets(record(),a,128)
    assert target['counts']=={'1':1,'7':1} and target['unsupported']==1
    assert int((target['key_heat']==1).sum())==1 and int(target['key_mask'].sum())==1
    assert not target['valid'][0].any()


def test_collisions_are_counted_not_overwritten():
    r=record(); r['nodes'].append({'id':'bar2','type':'bar','bbox':[10,10,30,40]})
    t=targets(r,torch.tensor([1.,1.,0.,0.,100.,60.]),128)
    assert t['collisions']['1']==2 and not t['box_mask'].any()


def test_losses_finite_gradients_and_padding():
    target=targets(record(),torch.tensor([1.,1.,0.,0.,100.,60.]),128)
    output={k:torch.zeros(1,c,32,32,requires_grad=True) for k,c in [('center',5),('size',10),('offset',10),('keypoint',2),('key_extent',4),('key_offset',4)]}
    loss,parts=detection_loss(output,target); assert torch.isfinite(loss)
    loss.backward(); assert all(torch.isfinite(v.grad).all() for v in output.values())
    assert output['center'].grad[0,:,31,31].abs().sum()==0
    assert 'segmentation' not in parts


def test_decode_known_center_and_empty():
    cfg={'decode_threshold':.1,'max_proposals':128,'nms_iou':.5}
    out={k:torch.zeros(1,c,32,32) for k,c in [('center',5),('size',10),('offset',10),('keypoint',2),('key_extent',4),('key_offset',4)]}
    out['center'].fill_(-20); out['keypoint'].fill_(-20); out['center'][0,0,4,5]=10
    a=torch.tensor([1.,1.,0.,0.,128.,128.]); p=decode(out,a,cfg)
    assert p['labels'].tolist()==[1] and torch.allclose(p['centers'],torch.tensor([[22.,18.]]))
    assert (p['boxes'][:,2:]>p['boxes'][:,:2]).all()
    out['center'].fill_(-20); assert decode(out,a,cfg)['boxes'].shape==(0,4)


def test_no_protected_test_access():
    with pytest.raises(ValueError): VisualDataset('.', 'test')
