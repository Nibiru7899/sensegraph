import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from sensegraph.io import read_json, write_jsonl
from sensegraph.synthetic import make_spec, render_chart, fingerprint
from sensegraph.structural_validation import validate_record
from sensegraph.graph_data import StructuralDataset, collate_charts, observable_graph
from sensegraph.leakage import leakage_report

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/structural.json")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("structural")
    rows = [render_chart(make_spec(CONFIG, i), root / f"{i}.png", root) for i in (0, 1, 2, 70, 85)]
    return root, rows


def test_renderer_geometry_all_types_and_splits(rendered):
    root, rows = rendered
    assert {row["chart_type"] for row in rows} == {"bar", "grouped_bar", "line"}
    for row in rows:
        assert validate_record(row, root) == []
        assert Image.open(root / row["image_path"]).size == tuple(row["spec"]["resolution"])
        matrix = np.array(row["axes"]["data_to_pixel"])
        for node in row["nodes"]:
            if node["type"] in ("bar", "point"):
                assert np.allclose((matrix @ [*node["data_xy"], 1])[:2], node["anchor_px"])


def test_resolutions_and_dpi_are_not_chart_type_shortcuts():
    specs = [make_spec(CONFIG, i) for i in range(CONFIG["count"])]
    for kind in CONFIG["chart_types"]:
        selected = [s for s in specs if s["chart_type"] == kind]
        assert len({tuple(s["resolution"]) for s in selected}) == len(CONFIG["resolutions"])
        assert len({s["dpi"] for s in selected}) == len(CONFIG["dpi"])


@pytest.mark.parametrize("corruption", ["bbox", "huge_box", "resolution", "transform", "value", "edge", "extra_edge", "answer", "evidence", "node_id", "numerical"])
def test_corrupt_annotations_are_rejected(rendered, corruption):
    root, rows = rendered
    row = copy.deepcopy(rows[0])
    mark = next(n for n in row["nodes"] if n["type"] == "bar")
    if corruption == "bbox":
        mark["bbox"][0] = -2
    elif corruption == "huge_box":
        mark["bbox"] = [0, 0, row["width"], row["height"]]
    elif corruption == "resolution":
        row["width"] += 10
    elif corruption == "transform":
        row["axes"]["data_to_pixel"][1][2] += 10
    elif corruption == "value":
        mark["numeric_value"] += 1
    elif corruption == "edge":
        row["edges"][0]["target"] = "missing"
    elif corruption == "extra_edge":
        row["edges"].append({"source": "title", "target": "axis:x", "relation": "labels"})
    elif corruption == "answer":
        row["questions"][0]["answer"] += 1
    elif corruption == "evidence":
        row["questions"][0]["evidence_boxes"][0]["bbox"] = [0, 0, 1, 1]
    elif corruption == "node_id":
        row["nodes"][0]["id"] = row["nodes"][1]["id"]
    else:
        row["numerical_relationships"] = []
    assert validate_record(row, root)


def test_zero_negative_tied_values(rendered):
    root, _ = rendered
    spec = make_spec(CONFIG, 0)
    spec["data"]["categories"] = ["C1", "C2", "C3"]
    spec["data"]["values"] = [[0, -5, -5]]
    spec["data_hash"] = fingerprint(spec["data"])
    row = render_chart(spec, root / "zero.png", root)
    assert validate_record(row, root) == []
    assert any(e["relation"] == "equal_to" for e in row["numerical_relationships"])
    assert next(q["answer"] for q in row["questions"] if q["operation"] == "minimum") == -5
    assert next(q["answer"] for q in row["questions"] if q["operation"] == "maximum") == 0


def test_render_repeat_is_deterministic(rendered):
    root, rows = rendered
    repeated = render_chart(rows[0]["spec"], root / "repeat.png", root)
    assert repeated["image_sha256"] == rows[0]["image_sha256"]
    assert repeated["nodes"] == rows[0]["nodes"]
    assert repeated["edges"] == rows[0]["edges"]


def test_pytorch_and_pyg_batch_no_label_leakage(rendered):
    root, rows = rendered
    manifest = root / "train.jsonl"
    write_jsonl(manifest, rows[:3])
    dataset = StructuralDataset(manifest, root)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_charts)))
    inputs, targets = batch["inputs"], batch["targets"]
    assert inputs["images"].shape[0] == 3
    assert torch.isfinite(inputs["graph"].x).all()
    assert set(inputs["graph"].keys()) <= {"x", "edge_index", "batch", "ptr", "num_nodes"}
    assert targets.node_type_y.numel() == targets.num_nodes
    assert targets.edge_y.shape[0] == targets.edge_index.shape[1]
    assert targets.evidence_index[0].max() < targets.question_op_y.numel()
    assert targets.evidence_index[1].max() < targets.num_nodes
    assert targets.value_mask.sum() > 0
    # Changing supervision alone cannot change observable features/candidate topology.
    image = dataset[0]["image"]
    boxes = [node["bbox"] for node in rows[0]["nodes"]]
    graph = observable_graph(image, boxes)
    assert torch.equal(graph.x, dataset[0]["graph"].x)
    assert graph.edge_index.shape[1] == len(boxes) * (len(boxes) - 1)


def test_split_leakage_and_duplicate_injection(rendered):
    root, rows = rendered
    assert leakage_report(rows, root)["violations"] == []
    duplicate = {**copy.deepcopy(rows[0]), "id": "leaked", "split": "test"}
    report = leakage_report([rows[0], duplicate], root)
    assert {r["kind"] for r in report["violations"]} >= {"template_hash", "data_hash", "exact_image"}
    assert report["near_duplicates"][0]["hamming_distance"] == 0
    duplicate["spec"]["data_hash"] = "renamed source"
    assert "numerical_configuration" in {r["kind"] for r in leakage_report([rows[0], duplicate], root)["violations"]}
