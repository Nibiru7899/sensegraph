import torch
from sensegraph.full_visual_training import decode_maps,improved,atomic_save
from sensegraph.visual_training import decode


def test_vectorized_decoder_matches_original():
    torch.manual_seed(48)
    cfg={'decode_threshold':.1,'max_proposals':128,'nms_iou':.5}
    output={k:torch.randn(1,c,12,12) for k,c in [('center',5),('size',10),('offset',10),('keypoint',2),('key_extent',4),('key_offset',4)]}
    affine=torch.tensor([1.,1.,0.,0.,48.,48.])
    slow=decode(output,affine,cfg); fast=decode_maps(output,affine,cfg)
    for key in slow: assert torch.allclose(slow[key],fast[key],atol=1e-5)


def test_stopping_and_atomic_cursor_roundtrip(tmp_path):
    assert improved(.5026,.5) and not improved(.502,.5)
    state={'cursor':128,'optimizer_steps':16,'rng':torch.get_rng_state()}
    path=tmp_path/'last.pt'; atomic_save(state,path)
    loaded=torch.load(path,weights_only=True)
    assert loaded['cursor']==128 and loaded['optimizer_steps']==16 and torch.equal(loaded['rng'],state['rng'])
    assert not path.with_suffix('.tmp').exists()


def test_optimizer_rng_resume_does_not_replay_committed_steps(tmp_path):
    torch.manual_seed(19)
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    data=torch.arange(15,dtype=torch.float32).reshape(5,3)/10
    path=tmp_path/'committed.pt'
    for i in range(5):
        opt.zero_grad(); model(data[i]).square().mean().backward(); opt.step()
        if i==2:
            atomic_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'cursor':3,'steps':3,'rng':torch.get_rng_state()},path)
    reference={k:v.detach().clone() for k,v in model.state_dict().items()}
    restored=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
    new_opt=torch.optim.AdamW(restored.parameters(),lr=.001)
    state=torch.load(path,weights_only=True); restored.load_state_dict(state['model']); new_opt.load_state_dict(state['optimizer']); torch.set_rng_state(state['rng'])
    processed=[]
    for i in range(state['cursor'],5):
        new_opt.zero_grad(); restored(data[i]).square().mean().backward(); new_opt.step(); processed.append(i)
    assert processed==[3,4]
    assert all(torch.equal(reference[k],v) for k,v in restored.state_dict().items())
