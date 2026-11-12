import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from sensegraph.io import read_json
from sensegraph.relation_models import RelationNet, RelationInputs, mask_features, message_graph, pack_samples, pair_features, rule_scores
from sensegraph.relation_metrics import average_precision, calibrate, metrics, paired_bootstrap
from sensegraph.relation_training import loss_weights, load_checkpoint, cached_split

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/relations.json")


def sample():
    x = torch.tensor([[.1,.1,.2,.2,.15,.15,.1,.1,.2,.4,.7],
                      [.3,.4,.5,.6,.4,.5,.2,.2,.7,.3,.1],
                      [.6,.3,.8,.4,.7,.35,.2,.1,.4,.4,.4]])
    candidate = torch.tensor([[0,0,1,1,2,2],[1,2,0,2,0,1]])
    labels = torch.zeros((6,10))
    labels[0,0] = labels[2,1] = 1
    return {"inputs": {"x": x, "candidate_index": candidate}, "targets": {"edge_y": labels}, "metadata": {"id":"fixture"}}


@pytest.mark.parametrize("architecture", ["mlp","gat"])
def test_shape_finite_gradients_and_checkpoint(architecture, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    cfg = {**CONFIG, "dropout": 0.}
    model = RelationNet(architecture, cfg)
    inputs, labels, _ = pack_samples([sample(), sample()])
    output = model(inputs)
    assert output.shape == (12,10)
    torch.nn.functional.binary_cross_entropy_with_logits(output, labels).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict":model.state_dict(),"variant":{"architecture":architecture},"config":cfg},path)
    restored, _ = load_checkpoint(path)
    assert torch.equal(model(inputs), restored(inputs))


def test_targets_and_metadata_cannot_change_forward_inputs():
    first, changed = sample(), sample()
    changed["targets"]["edge_y"].fill_(1)
    changed["metadata"].update(node_type_y=[999]*3, numeric_value=999, semantic_edges="forbidden")
    a, y1, _ = pack_samples([first])
    b, y2, _ = pack_samples([changed])
    assert not torch.equal(y1, y2)
    assert torch.equal(a.x,b.x) and torch.equal(a.message_index,b.message_index)
    model = RelationNet("gat",CONFIG).eval()
    assert torch.equal(model(a),model(b))
    assert set(a.__dict__) == {"x","candidate_index","message_index"}


def test_directed_candidates_and_no_cross_chart_messages():
    inputs, labels, slices = pack_samples([sample(),sample()])
    assert slices == [(0,6),(6,12)]
    source,target = inputs.message_index
    assert torch.equal(source//3,target//3)
    features = pair_features(inputs.x,inputs.candidate_index)
    assert torch.allclose(features[0,:2],-features[2,:2])
    assert not torch.equal(labels[0],labels[2])


def test_geometry_and_color_ablation_do_not_leak_derived_features():
    first = sample()
    other = copy.deepcopy(first)
    other["inputs"]["x"][:,:8] += 100
    a,_,_ = pack_samples([first],mode="no_geometry")
    b,_,_ = pack_samples([other],mode="no_geometry")
    assert torch.equal(a.x,b.x) and torch.equal(a.message_index,b.message_index)
    assert torch.equal(pair_features(a.x,a.candidate_index),pair_features(b.x,b.candidate_index))
    c = mask_features(first["inputs"]["x"],"no_color")
    assert torch.equal(c[:,8:],torch.zeros_like(c[:,8:]))
    assert torch.equal(pair_features(c,first["inputs"]["candidate_index"])[:,-1],torch.zeros(6))


def test_metrics_ap_ties_rare_classes_and_threshold_ties():
    assert average_precision([1,0],[.5,.5]) == .5
    assert average_precision([0,1],[.5,.5]) == .5
    assert average_precision([0,0],[.9,.1]) is None
    labels=np.zeros((2,10));labels[0,9]=1
    scores=np.zeros((2,10));scores[0,9]=.8
    thresholds=calibrate(labels,scores,[.1,.5,.7,.9])
    assert thresholds[9] == .7 and thresholds[0] == .5
    result=metrics(labels,scores,thresholds)
    assert result["per_class"]["greater_than"]["f1"] == 1
    assert result["macro_f1"] == .1 and result["micro_f1"] == 1
    assert result["per_class"]["equal_to"]["positives"] == 0
    assert result["ap_defined_classes"] == 1


def test_positive_weights_and_parameter_budget():
    assert torch.isfinite(loss_weights([sample()])).all()
    counts=[sum(p.numel() for p in RelationNet(name,CONFIG).parameters()) for name in ["mlp","gat"]]
    assert max(counts)/min(counts) <= 1.25


def test_paired_cluster_bootstrap_identity():
    counts=np.ones((2,5,10,4))
    result=paired_bootstrap(counts,counts,samples=50)
    assert result["macro_f1"]["ci95"] == [0.,0.]
    with pytest.raises(ValueError):
        paired_bootstrap(counts,counts[:,:4])


def test_test_cache_requires_explicit_sealed_caller(tmp_path):
    with pytest.raises(ValueError, match="Test tensors"):
        cached_split(tmp_path,"v2","test")


def test_rule_scores_are_finite_and_do_not_require_types():
    inputs,_,_=pack_samples([sample()])
    scores=rule_scores(inputs)
    assert scores.shape == (6,10) and torch.isfinite(scores).all()


def test_training_and_calibration_are_forbidden_after_sealing(tmp_path):
    from sensegraph.relation_experiment import safe_train, fit_v3
    path=tmp_path / "experiments/milestone3/sealed_inventory.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError,match="forbidden"):
        safe_train(tmp_path,"v3","gat",CONFIG)
    with pytest.raises(ValueError,match="forbidden"):
        fit_v3(tmp_path,CONFIG)
