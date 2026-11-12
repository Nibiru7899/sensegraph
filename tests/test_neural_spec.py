"""Random-weight CPU wiring tests, not fitting or scientific model evidence."""
import io
import torch
import pytest
from sensegraph.neural_spec import SenseGraphNet, CropCTCRecognizer, CharacterBiGRU, GraphStack128, QuestionNodeReasoner


def fixture(size=128):
    rgb = torch.rand(2, 3, size, size)
    boxes = [torch.tensor([[4., 5., 30., 40.], [35., 8., 60., 50.]], requires_grad=True), torch.empty(0, 4)]
    return dict(images=rgb, rgb=rgb, boxes=boxes, probabilities=torch.ones(2, 7)/7,
                confidence=torch.ones(2, 1)*.8, recognized_ids=torch.full((2, 8), 97, dtype=torch.long),
                available=torch.zeros(2), question_ids=torch.ones(2, 128, dtype=torch.long),
                lengths=torch.tensor([5, 7]), question_chart=torch.tensor([0, 1]))


def test_full_forward_shapes_and_checkpoint():
    torch.set_num_threads(1); torch.manual_seed(4)
    model = SenseGraphNet().eval()
    data = fixture()
    with torch.no_grad():
        output = model(**data)
        repeat = model(**data)
    assert output['nodes'].shape == (2,128) and output['relation_logits'].shape == (2,10)
    assert output['dense']['center'].shape == (2,5,32,32)
    r=output['reasoning']
    assert r['operation_logits'].shape == (2,7) and r['operand_a_logits'].shape == (2,3)
    assert r['evidence_logits'].shape == (2,2) and r['unsupported_empty_chart'].tolist() == [False,True]
    assert torch.isfinite(r['operation_logits']).all() and not torch.isnan(r['operand_a_logits']).any()
    assert torch.equal(output['relation_logits'],repeat['relation_logits'])
    storage=io.BytesIO(); torch.save(model.state_dict(),storage); storage.seek(0)
    model.load_state_dict(torch.load(storage,weights_only=True))
    with torch.no_grad():
        assert torch.equal(output['nodes'],model(**data)['nodes'])


def test_ocr_question_padding_and_ctc_shape():
    torch.set_num_threads(1)
    ocr=CropCTCRecognizer()
    assert ocr(torch.rand(2,1,32,128)).shape == (2,32,96)
    assert ocr(torch.empty(0,1,32,128)).shape == (0,32,96)
    q=CharacterBiGRU().eval(); ids=torch.ones(2,128,dtype=torch.long); lengths=torch.tensor([3,5])
    before=q(ids,lengths); ids[:,6:]=95
    assert torch.equal(before,q(ids,lengths))
    with pytest.raises(AssertionError):
        q(ids,torch.tensor([0,5]))


def test_gradients_freezes_and_discrete_boxes():
    torch.set_num_threads(1)
    m=SenseGraphNet(); m.set_stage('structural'); m.train(); data=fixture()
    result=m(**data); result['relation_logits'].square().mean().backward()
    assert data['boxes'][0].grad is None
    assert all(p.grad is None for p in m.backbone.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.graph.parameters())
    assert all(not layer.training for layer in m.backbone.modules() if isinstance(layer,torch.nn.BatchNorm2d))
    assert all(not p.requires_grad for p in m.heads['segmentation'].parameters())


@pytest.mark.parametrize('architecture',['gat','mlp'])
def test_graph_permutation_equivariance(architecture):
    torch.set_num_threads(1); torch.manual_seed(7)
    model=GraphStack128(architecture).eval(); h=torch.randn(4,128)
    edges=torch.tensor([[0,1,2,3,1],[1,2,3,0,3]]); attrs=torch.randn(5,8)
    permutation=torch.tensor([2,0,3,1]); inverse=torch.argsort(permutation)
    with torch.no_grad():
        actual=model(h,edges,attrs)
        permuted=model(h[permutation],inverse[edges],attrs)
    assert torch.allclose(actual[permutation],permuted,atol=1e-6)


def test_empty_chart_and_chart_isolation():
    torch.set_num_threads(1); r=QuestionNodeReasoner().eval()
    h=torch.randn(3,128); q=torch.randn(2,128); charts=torch.tensor([0,0,1]); qi=torch.tensor([0,1])
    a=r(q,h,charts,qi,2); h[2]*=100; b=r(q,h,charts,qi,2)
    assert torch.equal(a['operation_logits'][0],b['operation_logits'][0])
    empty=r(q,torch.empty(0,128),torch.empty(0,dtype=torch.long),qi,2)
    assert empty['unsupported_empty_chart'].all() and torch.isfinite(empty['operation_logits']).all()
