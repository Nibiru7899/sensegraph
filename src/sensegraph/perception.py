"""Image-only proposals/features and explicitly separate matching supervision."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .graph_data import observable_graph
from .relation_models import RelationInputs, message_graph
from .synthetic import RELATIONS

CLASSES = ("background", "bar", "point", "line_segment", "text", "legend_item", "axis", "tick")
TYPE_CLASS = {name: i for i, name in enumerate(CLASSES) if i}
TYPE_CLASS.update({name: 4 for name in ("title", "axis_label", "tick_label")})


def image_tensor(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255


def detection_target(record):
    boxes, labels, node_ids, rejected = [], [], [], []
    for node in record["nodes"]:
        label = TYPE_CLASS.get(node["type"])
        if label is None:
            rejected.append({"id": node["id"], "reason": "unsupported_type", "type": node["type"]})
            continue
        b = np.asarray(node["bbox"], dtype=float)
        if (b.shape != (4,) or not np.isfinite(b).all() or b[0] < 0 or b[1] < 0
                or b[2] > record["width"] or b[3] > record["height"] or b[2] <= b[0] or b[3] <= b[1]):
            rejected.append({"id": node["id"], "reason": "invalid_box"})
            continue
        boxes.append(b.tolist()); labels.append(label); node_ids.append(node["id"])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)}, node_ids, rejected


def detector(root, config):
    # The factory uses a local, verified official cache. It must never implicitly download.
    path = Path(root) / config["weights_path"]
    if not path.exists():
        raise FileNotFoundError("Run the budgeted acquire command first")
    from .io import sha256
    if not sha256(path).startswith("907ea3f9"):
        raise ValueError("Official weight hash prefix mismatch")
    os.environ["TORCH_HOME"] = str(Path(root).resolve() / ".cache/torch")
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
                                               box_score_thresh=0.01, box_detections_per_img=config["max_detections"])
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(CLASSES))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("rpn.", "roi_heads.box_predictor.")))
    return model


@torch.inference_mode()
def visual_embeddings(model, image, boxes):
    """Frozen pretrained ROI features, accepting pixels and proposals only."""
    if not len(boxes):
        return torch.empty((0, 1024), device=image.device)
    model.eval()
    transformed, targets = model.transform([image], [{"boxes": boxes}])
    maps = model.backbone(transformed.tensors)
    pooled = model.roi_heads.box_roi_pool(maps, [targets[0]["boxes"]], transformed.image_sizes)
    return model.roi_heads.box_head(pooled)


def predicted_graph(image, boxes, connectivity="spatial"):
    """No target inputs. Empty, duplicate and spurious positive-area boxes are legal."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    h, w = image.shape[-2:]
    if (not torch.isfinite(boxes).all() or (boxes[:, :2] < 0).any()
            or (boxes[:, 2:] > torch.tensor([w, h])).any() or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("Invalid predicted boxes")
    graph = observable_graph(image.cpu(), boxes.tolist())
    return RelationInputs(graph.x, graph.edge_index, message_graph(graph.x, connectivity))


def match_nodes(prediction, target, iou_threshold=0.5):
    """Confidence-ordered one-to-one matching; types compatible at detector granularity."""
    n, m = len(prediction["boxes"]), len(target["boxes"])
    match = torch.full((n,), -1, dtype=torch.long)
    overlap = box_iou(prediction["boxes"].cpu(), target["boxes"].cpu())
    used = set()
    for p in torch.argsort(prediction["scores"].cpu(), descending=True, stable=True).tolist():
        choices = [g for g in range(m) if g not in used and int(prediction["labels"][p]) == int(target["labels"][g])
                   and float(overlap[p, g]) >= iou_threshold]
        if choices:
            g = max(choices, key=lambda g: (float(overlap[p, g]), -g))
            match[p] = g; used.add(g)
    return match, overlap


def detection_metrics(predictions, targets, iou_threshold=0.5, score_threshold=0.25):
    result = {}
    for label, name in enumerate(CLASSES[1:], 1):
        scored, total, errors, duplicates = [], 0, [], 0
        for pred, gt in zip(predictions, targets):
            match, overlaps = match_nodes(pred, gt, iou_threshold)
            total += int((gt["labels"] == label).sum())
            for p in torch.where(pred["labels"] == label)[0].tolist():
                score, g = float(pred["scores"][p]), int(match[p])
                scored.append((score, g >= 0))
                if score >= score_threshold:
                    if g >= 0:
                        a, b = pred["boxes"][p], gt["boxes"][g]
                        scale = (b[2:] - b[:2]).square().sum().sqrt().clamp_min(1e-8)
                        errors.append(float(((a[:2]+a[2:]-b[:2]-b[2:])/2).norm()/scale))
                    elif any(float(overlaps[p, j]) >= iou_threshold for j in torch.where(gt["labels"] == label)[0].tolist()):
                        duplicates += 1
        scored.sort(key=lambda x: -x[0])
        y = np.array([x[1] for x in scored], dtype=int)
        tp = np.cumsum(y); fp = np.cumsum(1-y)
        precision = tp / np.maximum(1, tp+fp)
        recall = tp / max(1, total)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(np.sum(np.diff(np.r_[0., recall]) * envelope)) if total else None
        accepted = [x for x in scored if x[0] >= score_threshold]
        positives = sum(x[1] for x in accepted)
        result[name] = {"support": total, "tp": positives, "fp": len(accepted)-positives, "fn": total-positives,
                        "precision": positives/max(1, len(accepted)), "recall": positives/max(1, total), "ap": ap,
                        "duplicates": duplicates, "normalized_center_error": float(np.mean(errors)) if errors else None}
    return {"per_class": result, "macro_ap": float(np.mean([r["ap"] for r in result.values() if r["ap"] is not None])),
            "macro_recall": float(np.mean([r["recall"] for r in result.values()]))}


def relation_recovery(record, predicted_edges, proposal_to_node_id):
    """Scored triples after thresholds. Unmatched/duplicate predictions count as FP.

    Endpoint IDs exist only in evaluator matching output, never forward inputs.
    All GT edges, including unsupported detector types, remain in E2E denominators.
    """
    truth = {(e["source"], e["target"], e["relation"]) for e in record["edges"]}
    recovered, fp = set(), {r: 0 for r in RELATIONS}
    mapped = {v for v in proposal_to_node_id.values() if v is not None}
    for source, target, relation in predicted_edges:
        edge = (proposal_to_node_id.get(source), proposal_to_node_id.get(target), relation)
        if edge in truth and edge not in recovered:
            recovered.add(edge)
        else:
            fp[relation] += 1
    conditional = {e for e in truth if e[0] in mapped and e[1] in mapped}
    counts = {}
    for r in RELATIONS:
        tp = sum(e[2] == r for e in recovered)
        fn = sum(e[2] == r for e in truth - recovered)
        cf = sum(e[2] == r for e in conditional - recovered)
        # Conditional FP excludes relations with unmatched endpoints.
        cp = sum(rel == r and proposal_to_node_id.get(s) is not None and proposal_to_node_id.get(t) is not None
                 for s, t, rel in predicted_edges) - tp
        counts[r] = {"tp": tp, "fp": fp[r], "fn": fn, "f1": 2*tp/max(1,2*tp+fp[r]+fn),
                     "conditional_fp": cp, "conditional_fn": cf, "conditional_f1": 2*tp/max(1,2*tp+cp+cf)}
    return counts


def ocr(image, rotation=0, executable=None):
    """Local transcription only; no silent GT fallback. No implicit acquisition."""
    engine = executable or shutil.which("tesseract")
    if not engine:
        return {"status": "unavailable", "reason": "Tesseract executable not installed", "words": []}
    # Accept explicit rotations for inference-time hypotheses; never source annotations.
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Only explicit right-angle OCR rotations supported")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "crop.png"
        image.rotate(rotation, expand=True).save(path)
        done = subprocess.run([engine, str(path), "stdout", "--psm", "6", "tsv"], capture_output=True, text=True, check=True, timeout=30)
    import csv, io
    words = [{"text": row["text"], "confidence": float(row["conf"])} for row in csv.DictReader(io.StringIO(done.stdout), delimiter="\t") if row["text"].strip()]
    return {"status": "ok", "rotation": rotation, "words": words}
