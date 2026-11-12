import torch
import pytest
from PIL import Image
from sensegraph.perception import detection_target, predicted_graph, match_nodes, detection_metrics, relation_recovery, ocr
from sensegraph.relation_models import pack_samples


def target():
    return {"boxes":torch.tensor([[1.,1.,5.,5.],[10.,10.,14.,14.]]),"labels":torch.tensor([1,1])}


def pred():
    return {"boxes":torch.tensor([[1.,1.,5.,5.],[1.,1.,5.,5.],[18.,18.,20.,20.]]),
            "labels":torch.tensor([1,1,2]),"scores":torch.tensor([.9,.8,.7])}


def test_unique_matching_duplicates_and_wrong_type():
    matches,_=match_nodes(pred(),target())
    assert matches.tolist()==[0,-1,-1]
    metrics=detection_metrics([pred()],[target()])["per_class"]
    assert metrics["bar"]["tp"]==1 and metrics["bar"]["fp"]==1 and metrics["bar"]["fn"]==1
    assert metrics["bar"]["duplicates"]==1
    assert metrics["bar"]["ap"]==.5
    assert metrics["point"]["ap"] is None and metrics["point"]["fp"]==1


def test_iou_boundary_and_no_predictions():
    gt={"boxes":torch.tensor([[0.,0.,2.,2.]]),"labels":torch.tensor([1])}
    p={"boxes":torch.tensor([[0.,0.,1.,2.]]),"labels":torch.tensor([1]),"scores":torch.tensor([.9])}
    assert match_nodes(p,gt,.5)[0].tolist()==[0]
    assert match_nodes(p,gt,.5001)[0].tolist()==[-1]
    empty={k:v[:0] for k,v in p.items()}
    assert detection_metrics([empty],[gt])["per_class"]["bar"]["fn"]==1


def test_annotation_mapping_retains_invalid_and_unsupported():
    record={"width":20,"height":20,"nodes":[{"id":"s","type":"series","bbox":[1,1,10,10]},
            {"id":"b","type":"bar","bbox":[1,1,1,5]}, {"id":"t","type":"title","bbox":[1,1,10,5]}]}
    tensors,ids,rejected=detection_target(record)
    assert ids==["t"] and tensors["labels"].tolist()==[4]
    assert {r["reason"] for r in rejected}=={"unsupported_type","invalid_box"}


def test_image_only_graph_missing_duplicate_false_boxes_and_batching():
    image=torch.ones(3,20,20)
    empty=predicted_graph(image,[])
    assert empty.x.shape==(0,11) and empty.candidate_index.shape==(2,0)
    graph=predicted_graph(image,pred()["boxes"])
    assert graph.candidate_index.shape==(2,6)
    samples=[{"inputs":{"x":graph.x,"candidate_index":graph.candidate_index},"targets":{"edge_y":torch.zeros(6,10)}}]*2
    packed,_,_=pack_samples(samples)
    assert torch.equal(packed.message_index[0]//3,packed.message_index[1]//3)
    with pytest.raises(ValueError):
        predicted_graph(image,[[0,0,0,5]])


def test_end_to_end_denominator_and_conditional_universe():
    record={"edges":[{"source":"a","target":"b","relation":"less_than"},
                     {"source":"a","target":"s","relation":"belongs_to"}]}
    scores=relation_recovery(record,[(0,1,"less_than"),(0,2,"less_than")],{0:"a",1:"b",2:None})
    assert scores["less_than"]["tp"]==1 and scores["less_than"]["fp"]==1
    assert scores["less_than"]["conditional_fp"]==0
    assert scores["belongs_to"]["fn"]==1 and scores["belongs_to"]["conditional_fn"]==0
    assert len(scores)==10


def test_no_ocr_source_fallback(monkeypatch):
    monkeypatch.setattr("shutil.which",lambda _:None)
    result=ocr(Image.new("RGB",(20,20)))
    assert result["status"]=="unavailable" and result["words"]==[]
