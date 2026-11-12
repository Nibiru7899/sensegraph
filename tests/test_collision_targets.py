import torch
from sensegraph.collision_targets import invariant_targets
from sensegraph.visual_training import targets


def test_colliding_gaussian_union_is_order_invariant():
    torch.set_num_threads(1)
    record={'width':160,'height':160,'nodes':[{'id':'large','type':'line_segment','bbox':[10,10,130,130]},{'id':'small','type':'line_segment','bbox':[40,40,100,100]}]}
    reverse={**record,'nodes':list(reversed(record['nodes']))}; affine=torch.tensor([1.,1.,0.,0.,160.,160.])
    old=targets(record,affine,160); old_reverse=targets(reverse,affine,160)
    assert not torch.equal(old['box_heat'],old_reverse['box_heat'])
    a=invariant_targets(record,affine,160); b=invariant_targets(reverse,affine,160)
    assert torch.equal(a['box_heat'],b['box_heat'])
    assert torch.equal(a['box_heat']==1,old['box_heat']==1)
    for key in ['box_extent','box_offset','box_mask','key_extent','key_offset','key_mask','valid']:
        assert torch.equal(a[key],old[key])
    assert not a['box_mask'].any() and a['counts']==old['counts'] and a['collisions']==old['collisions']


def test_noncolliding_targets_identical():
    record={'width':160,'height':160,'nodes':[{'id':'point','type':'point','bbox':[10,10,16,16]}]}; affine=torch.tensor([1.,1.,0.,0.,160.,160.])
    a=targets(record,affine,160); b=invariant_targets(record,affine,160)
    assert all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in a)
