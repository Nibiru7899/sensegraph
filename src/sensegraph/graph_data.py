"""Oracle-proposal training datasets and explicitly label-free model inputs."""
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from .io import read_jsonl
from .synthetic import NODE_TYPES, RELATIONS, OPERATIONS
from .structural_validation import validate_record


class ChartGraph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "evidence_index":
            return torch.tensor([[self.question_op_y.numel()], [self.num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)


def observable_graph(image, boxes):
    """Accepts image pixels and region proposals only; all ordered pairs are candidates."""
    _, height, width = image.shape
    pixels = image.numpy() if image.device.type == "cpu" else None
    features = []
    for box in boxes:
        x0, y0, x1, y1 = box
        geometry = [x0 / width, y0 / height, x1 / width, y1 / height,
                    (x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height]
        left, top, right, bottom = max(0, math.floor(x0)), max(0, math.floor(y0)), min(width, math.ceil(x1)), min(height, math.ceil(y1))
        if right <= left or bottom <= top:
            raise ValueError("Empty region proposal")
        rgb = pixels[:, top:bottom, left:right].mean(axis=(1, 2)).tolist() if pixels is not None else image[:, top:bottom, left:right].mean(dim=(1, 2)).tolist()
        features.append(geometry + rgb)
    count = len(features)
    pairs = [(s, t) for s in range(count) for t in range(count) if s != t]
    return Data(x=torch.tensor(features, dtype=torch.float32).reshape(count, 11),
                edge_index=torch.tensor(pairs, dtype=torch.long).t().contiguous().reshape(2, -1), num_nodes=count)


class StructuralDataset(Dataset):
    def __init__(self, manifest, root="."):
        self.root = Path(root)
        self.records = read_jsonl(manifest)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        errors = validate_record(record, self.root)
        if errors:
            raise ValueError(f"Invalid supervision: {errors}")
        with Image.open(self.root / record["image_path"]) as source:
            array = np.asarray(source.convert("RGB")).copy()
        image = torch.from_numpy(array).permute(2, 0, 1).float() / 255
        nodes = record["nodes"]
        identity = {node["id"]: i for i, node in enumerate(nodes)}
        base = observable_graph(image, [node["bbox"] for node in nodes])
        graph = ChartGraph(x=base.x, edge_index=base.edge_index, num_nodes=base.num_nodes)
        graph.node_type_y = torch.tensor([NODE_TYPES.index(node["type"]) for node in nodes])
        graph.value_mask = torch.tensor([node.get("numeric_value") is not None for node in nodes])
        graph.value_y = torch.tensor([float(node.get("numeric_value") or 0.) for node in nodes], dtype=torch.float32)
        candidates = {(int(s), int(t)): i for i, (s, t) in enumerate(graph.edge_index.t())}
        graph.edge_y = torch.zeros((graph.edge_index.shape[1], len(RELATIONS)), dtype=torch.float32)
        for edge in record["edges"]:
            pair = (identity[edge["source"]], identity[edge["target"]])
            graph.edge_y[candidates[pair], RELATIONS.index(edge["relation"])] = 1
        questions = record["questions"]
        graph.question_op_y = torch.tensor([OPERATIONS.index(q["operation"]) for q in questions])
        graph.answer_y = torch.tensor([float(q["answer"]) for q in questions], dtype=torch.float32)
        graph.evidence_index = torch.tensor([(i, identity[key]) for i, q in enumerate(questions) for key in q["evidence_nodes"]], dtype=torch.long).t().contiguous()
        graph.validate(raise_on_error=True)
        return {"image": image, "graph": graph, "questions": [q["question"] for q in questions],
                "metadata": {"id": record["id"], "split": record["split"], "template_id": record["spec"]["template"]["id"],
                             "proposal_source": "renderer_ground_truth_ORACLE_training_only", "node_ids": list(identity)}}


def collate_charts(items):
    """Pad images without scaling coordinates; separate targets from inference-shaped inputs."""
    if not items:
        raise ValueError("Cannot batch zero charts")
    sizes = torch.tensor([[item["image"].shape[1], item["image"].shape[2]] for item in items])
    height, width = sizes.max(dim=0).values.tolist()
    images = torch.zeros((len(items), 3, height, width))
    image_mask = torch.zeros((len(items), height, width), dtype=torch.bool)
    for i, item in enumerate(items):
        h, w = sizes[i].tolist()
        images[i, :, :h, :w] = item["image"]
        image_mask[i, :h, :w] = True
    targets = Batch.from_data_list([item["graph"] for item in items])
    inputs = Batch.from_data_list([Data(x=item["graph"].x, edge_index=item["graph"].edge_index, num_nodes=item["graph"].num_nodes) for item in items])
    if not torch.equal(targets.batch[targets.edge_index[0]], targets.batch[targets.edge_index[1]]):
        raise ValueError("Candidate edges cross graph boundaries")
    question_batch = torch.repeat_interleave(torch.arange(len(items)), torch.tensor([len(item["questions"]) for item in items]))
    if not torch.equal(question_batch[targets.evidence_index[0]], targets.batch[targets.evidence_index[1]]):
        raise ValueError("Evidence links cross graph boundaries")
    return {"inputs": {"images": images, "image_mask": image_mask, "image_sizes": sizes, "graph": inputs,
                       "questions": [item["questions"] for item in items]},
            "targets": targets, "metadata": [item["metadata"] for item in items]}
