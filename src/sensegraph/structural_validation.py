"""Independent consistency checks for renderer-derived supervision."""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .io import sha256
from .synthetic import NODE_TYPES, RELATIONS, OPERATIONS, fingerprint


def valid_box(box, width, height):
    return len(box) == 4 and all(math.isfinite(v) for v in box) and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height


def validate_record(record, root=None):
    errors = []
    width, height = record["width"], record["height"]
    nodes = {node["id"]: node for node in record["nodes"]}
    if len(nodes) != len(record["nodes"]):
        errors.append("duplicate node IDs")
    for node in nodes.values():
        if node["type"] not in NODE_TYPES or not valid_box(node["bbox"], width, height):
            errors.append(f"invalid node type or bbox: {node['id']}")
    edges = {(item["source"], item["target"], item["relation"]) for item in record["edges"]}
    if len(edges) != len(record["edges"]):
        errors.append("duplicate edges")
    for source, target, relation in edges:
        if source not in nodes or target not in nodes or relation not in RELATIONS or source == target:
            errors.append("invalid edge reference or relation")
    matrix = np.asarray(record["axes"]["data_to_pixel"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        return errors + ["invalid data-to-pixel transform"]
    inverse = np.linalg.inv(matrix)
    data = record["spec"]["data"]
    if fingerprint(data) != record["spec"]["data_hash"] or fingerprint({k: v for k, v in record["spec"]["template"].items() if k != "id"}) != record["spec"]["template_hash"]:
        errors.append("source data/template fingerprint mismatch")
    if any(record[key] != record["spec"][key] for key in ("id", "split", "chart_type")) or [width, height] != record["spec"]["resolution"]:
        errors.append("record/spec identity or resolution mismatch")
    expected_nodes = {"title", "axis:x", "axis:y", "axis_label:x", "axis_label:y"}
    for s in range(len(data["series"])):
        expected_nodes.update((f"series:{s}", f"legend:{s}"))
        expected_nodes.update(f"mark:{s}:{j}" for j in range(len(data["categories"])))
        if record["chart_type"] == "line":
            expected_nodes.update(f"segment:{s}:{j}" for j in range(len(data["categories"]) - 1))
    for axis, count in (("x", len(data["categories"])), ("y", len(record["axes"]["y"]["ticks"]))):
        for j in range(count):
            expected_nodes.update((f"tick:{axis}:{j}", f"tick_label:{axis}:{j}"))
    if set(nodes) != expected_nodes:
        errors.append("node identities do not match chart structure")
    prefixes = {"mark": "point" if record["chart_type"] == "line" else "bar", "segment": "line_segment", "legend": "legend_item"}
    for key, node in nodes.items():
        prefix = key.split(":")[0]
        if node["type"] != prefixes.get(prefix, prefix):
            errors.append("node type disagrees with semantic identity")
    semantic = {("axis_label:x", "axis:x", "labels"), ("axis_label:y", "axis:y", "labels")}
    for axis, count in (("x", len(data["categories"])), ("y", len(record["axes"]["y"]["ticks"]))):
        for j in range(count):
            semantic.add((f"tick:{axis}:{j}", f"axis:{axis}", "belongs_to"))
            semantic.add((f"tick_label:{axis}:{j}", f"tick:{axis}:{j}", "labels"))
            tick = nodes.get(f"tick:{axis}:{j}")
            label = nodes.get(f"tick_label:{axis}:{j}")
            expected_value = j if axis == "x" else record["axes"]["y"]["ticks"][j]
            if tick is None or tick.get("axis_value") != expected_value or label is None:
                errors.append("missing or inconsistent tick")
                continue
            if axis == "x" and label.get("text") != data["categories"][j]:
                errors.append("categorical tick label mismatch")
            if axis == "y" and float(label.get("text", "nan")) != expected_value:
                errors.append("numeric tick display mismatch")
            xy = [expected_value, record["axes"]["y"]["limits"][0], 1.] if axis == "x" else [record["axes"]["x"]["limits"][0], expected_value, 1.]
            if not np.allclose((matrix @ xy)[:2], tick.get("anchor_px"), atol=1e-6):
                errors.append("tick transform mismatch")
    marks = {key: node for key, node in nodes.items() if node["type"] in ("bar", "point")}
    if len(marks) != len(data["series"]) * len(data["categories"]):
        errors.append("missing data marks")
    for key, node in marks.items():
        s, j = node["series"], node["category"]
        value = data["values"][s][j]
        if node["numeric_value"] != value or node["data_xy"][1] != value:
            errors.append("mark numerical value disagrees with source data")
        mapped = matrix @ [*node["data_xy"], 1.]
        anchor = np.array([*node["anchor_px"], 1.])
        if not np.allclose(mapped, anchor, atol=1e-6) or not np.allclose((inverse @ anchor)[:2], node["data_xy"], atol=1e-6):
            errors.append("axis-to-value transform mismatch")
        x0, y0, x1, y1 = node["bbox"]
        if not (x0 - 1e-6 <= anchor[0] <= x1 + 1e-6 and y0 - 1e-6 <= anchor[1] <= y1 + 1e-6):
            errors.append("data anchor outside mark box")
        if node["type"] == "bar":
            zero = (matrix @ [node["data_xy"][0], 0, 1])[1]
            if not (y0 - 1 <= zero <= y1 + 1):
                errors.append("bar does not reach zero baseline")
        expected = {(key, f"series:{s}", "belongs_to"), (key, f"axis:x", "maps_to_axis"),
                    (key, f"axis:y", "maps_to_axis"), (key, f"tick:x:{j}", "maps_to_tick")}
        if not expected.issubset(edges):
            errors.append("missing mark semantic relationship")
        semantic.update(expected)
        if j:
            semantic.add((f"mark:{s}:{j-1}", key, "next_in_series"))
        count = len(data["series"])
        x = j if record["chart_type"] == "line" else j + (s - (count - 1) / 2) * 0.72 / count
        if not np.allclose(node["data_xy"], [x, value], atol=1e-6):
            errors.append("source data position mismatch")
        dpi = record["spec"]["dpi"]
        if node["type"] == "bar":
            half = 0.72 / count * 0.94 / 2
            corners = np.array([matrix @ [x - half, 0, 1], matrix @ [x + half, value, 1]])
            pad = 0.4 * dpi / 72
        else:
            corners = np.array([mapped, mapped])
            pad = 2.8 * dpi / 72
        expected_box = [max(0., corners[:, 0].min() - pad), max(0., corners[:, 1].min() - pad),
                        min(width, corners[:, 0].max() + pad), min(height, corners[:, 1].max() + pad)]
        if not np.allclose(node["bbox"], expected_box, atol=1e-5):
            errors.append("mark box disagrees with rendered geometry")
        if j and (f"mark:{s}:{j-1}", key, "next_in_series") not in edges:
            errors.append("missing ordered-series relation")
    for node in nodes.values():
        key = node["id"]
        if node["type"] == "tick":
            if (key, f"axis:{node['axis']}", "belongs_to") not in edges:
                errors.append("tick mapped to wrong axis")
        if node["type"] == "legend_item":
            if (key, f"series:{key.split(':')[1]}", "represents_series") not in edges:
                errors.append("legend series relationship missing")
        if node["type"] == "line_segment":
            s, j = node["series"], node["category"]
            if any((key, f"mark:{s}:{k}", "connected_to") not in edges for k in (j, j + 1)):
                errors.append("segment endpoints incorrect")
            semantic.add((key, f"series:{s}", "belongs_to"))
            semantic.update((key, f"mark:{s}:{k}", "connected_to") for k in (j, j + 1))
            ends = np.array([matrix @ [k, data["values"][s][k], 1.] for k in (j, j + 1)])
            pad = 0.8 * record["spec"]["dpi"] / 72
            expected_box = [max(0., ends[:, 0].min() - pad), max(0., ends[:, 1].min() - pad),
                            min(width, ends[:, 0].max() + pad), min(height, ends[:, 1].max() + pad)]
            if not np.allclose(node["bbox"], expected_box, atol=1e-5):
                errors.append("segment bbox inconsistent with endpoints")
    for s in range(len(data["series"])):
        semantic.add((f"legend:{s}", f"series:{s}", "represents_series"))
    expected_numerical = set()
    for a, first in marks.items():
        for b, second in marks.items():
            if a == b:
                continue
            av = data["values"][first["series"]][first["category"]]
            bv = data["values"][second["series"]][second["category"]]
            relation = "less_than" if av < bv else "greater_than" if av > bv else "equal_to"
            expected_numerical.add((a, b, relation))
    actual = {(r["source"], r["target"], r["relation"]) for r in record["numerical_relationships"]}
    if actual != expected_numerical or not expected_numerical.issubset(edges):
        errors.append("numerical relationships inconsistent")
    if edges != semantic | expected_numerical:
        errors.append("semantic edge set does not match chart structure")
    # Reject incorrect positive edges as well as missing expected edges.
    for source, target, relation in edges:
        if source not in nodes or target not in nodes:
            continue
        a, b = nodes[source], nodes[target]
        if relation in ("less_than", "equal_to", "greater_than") and (source, target, relation) not in expected_numerical:
            errors.append("incorrect numerical edge")
        if relation == "belongs_to" and a["type"] in ("bar", "point", "line_segment") and target != f"series:{a['series']}":
            errors.append("incorrect series membership")
        if relation == "maps_to_axis" and (a["type"] not in ("bar", "point") or target not in ("axis:x", "axis:y")):
            errors.append("incorrect axis mapping")
    qids = set()
    for question in record["questions"]:
        if question["id"] in qids:
            errors.append("duplicate question ID")
        qids.add(question["id"])
        operands = question["operands"]
        if not operands or any(key not in marks for key in operands):
            errors.append("invalid QA operands")
            continue
        vals = [data["values"][marks[key]["series"]][marks[key]["category"]] for key in operands]
        op = question["operation"]
        expected_answer = None
        if op == "lookup" and len(vals) == 1:
            expected_answer = vals[0]
        elif op in ("maximum", "minimum"):
            expected_answer = (max if op == "maximum" else min)(vals)
            if set(operands) != set(marks):
                errors.append("extremum omits comparison operands")
        elif op == "difference" and len(vals) == 2:
            expected_answer = abs(vals[0] - vals[1])
        elif op == "comparison" and len(vals) == 2:
            expected_answer = vals[0] > vals[1]
        elif op == "count":
            expected_answer = len(operands)
            if set(operands) != {key for key, node in marks.items() if node["series"] == 0}:
                errors.append("count series operands incorrect")
        elif op == "series_difference":
            expected_answer = sum(data["values"][0]) - sum(data["values"][1])
            if set(operands) != {key for key, node in marks.items() if node["series"] in (0, 1)}:
                errors.append("series difference operands incomplete")
        if op not in OPERATIONS or expected_answer is None or question["answer"] != expected_answer:
            errors.append("ground-truth answer incorrect")
        evidence = question["evidence_nodes"]
        if not set(operands).issubset(evidence) or any(key not in nodes for key in evidence):
            errors.append("evidence omits operands or has dangling references")
        boxes = question["evidence_boxes"]
        if [item["node_id"] for item in boxes] != evidence:
            errors.append("evidence labels and boxes disagree")
        for item in boxes:
            if item["node_id"] not in nodes or item["bbox"] != nodes[item["node_id"]]["bbox"] or not valid_box(item["bbox"], width, height):
                errors.append("evidence coordinate mismatch")
    if root is not None:
        image = Path(root) / record["image_path"]
        with Image.open(image) as im:
            if im.size != (width, height):
                errors.append("image resolution mismatch")
        if sha256(image) != record["image_sha256"]:
            errors.append("image identity changed")
    return sorted(set(errors))


def draw_overlay(image_path, nodes, destination, types=None):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"bar": "red", "point": "red", "line_segment": "orange", "tick_label": "blue", "axis": "green"}
    for node in nodes:
        if types is not None and node["type"] not in types:
            continue
        if node.get("bbox") is not None:
            draw.rectangle(node["bbox"], outline=colors.get(node["type"], "purple"), width=1)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
