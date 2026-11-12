"""Renderer-derived chart supervision. No OCR, tight crop, or image rescaling."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
from PIL import Image

from .io import sha256, write_json, write_jsonl

NODE_TYPES = ("axis", "tick", "tick_label", "title", "axis_label", "legend_item", "series", "bar", "point", "line_segment")
RELATIONS = ("belongs_to", "maps_to_axis", "maps_to_tick", "represents_series", "labels", "connected_to", "next_in_series", "less_than", "equal_to", "greater_than")
OPERATIONS = ("lookup", "maximum", "minimum", "difference", "comparison", "count", "series_difference")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def comparison(a, b):
    return "less_than" if a < b else "greater_than" if a > b else "equal_to"


def make_spec(config, index):
    if sum(config["splits"].values()) != config["count"]:
        raise ValueError("Split counts must sum to count")
    start = 0
    for split, count in config["splits"].items():
        if start <= index < start + count:
            break
        start += count
    else:
        raise ValueError("Sample index outside configuration")
    seed = int(fingerprint([config["seed"], split, index])[:16], 16)
    rng = np.random.default_rng(seed)
    chart_type = config["chart_types"][index % len(config["chart_types"])]
    template = config["templates"][split][(index - start) % len(config["templates"][split])]
    n = int(rng.choice(config["category_counts"]))
    series_count = 1 if chart_type == "bar" else int(rng.choice(config["series_counts"]))
    low, high = config["value_ranges"][int(rng.integers(len(config["value_ranges"])))]
    values = rng.integers(low, high + 1, (series_count, n)).tolist()
    source = {"categories": [f"C{i + 1}" for i in range(n)], "series": [f"Series {i + 1}" for i in range(series_count)], "values": values}
    return {"id": f"synthetic-{index:05d}", "split": split, "seed": seed, "chart_type": chart_type,
            "template": template, "template_hash": fingerprint({k: v for k, v in template.items() if k != "id"}),
            "data": source, "data_hash": fingerprint(source),
            "resolution": config["resolutions"][int(rng.integers(len(config["resolutions"])))],
            "dpi": config["dpi"][int(rng.integers(len(config["dpi"])))]}


def render_chart(spec, destination, root):
    """All coordinates are continuous xyxy on the exact saved canvas, origin top-left."""
    width, height = spec["resolution"]
    dpi, template = spec["dpi"], spec["template"]
    with matplotlib.rc_context({"font.family": template["font"], "font.size": 9, "axes.unicode_minus": False}):
        fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=template["background"])
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes(template["layout"], facecolor=template["background"])
        ax.spines[["top", "right"]].set_visible(False)
        if template["grid"]:
            ax.grid(True, axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        data = spec["data"]
        values = np.array(data["values"], dtype=float)
        series_count, n = values.shape
        lower = min(0., float(values.min()))
        upper = max(0., float(values.max()))
        span = max(upper - lower, 10.)
        low, high = math.floor((lower - span * 0.15) / 10) * 10., math.ceil((upper + span * 0.20) / 10) * 10.
        ax.set_ylim(low, high)
        y_ticks = np.linspace(low, high, 5)
        ax.set_yticks(y_ticks, [f"{v:.2f}" for v in y_ticks])
        ax.set_xlim(-0.65, n - 0.35)
        ax.set_xticks(range(n), data["categories"], rotation=template["rotation"])
        ax.set_xlabel("Category")
        ax.set_ylabel("Value (units)")
        ax.set_title(f"Synthetic {spec['chart_type'].replace('_', ' ')}")
        artists, points, segments = {}, {}, []
        for s, label in enumerate(data["series"]):
            color = template["palette"][s % len(template["palette"])]
            if spec["chart_type"] != "line":
                bar_width = 0.72 / series_count
                xs = np.arange(n) + (s - (series_count - 1) / 2) * bar_width
                bars = ax.bar(xs, values[s], width=bar_width * 0.94, color=color, edgecolor=color, linewidth=0.8, label=label)
                for j, artist in enumerate(bars):
                    identity = f"mark:{s}:{j}"
                    artists[identity] = (artist, 0.4 * dpi / 72)
                    points[identity] = [float(xs[j]), float(values[s, j])]
            else:
                # Separate artists provide exact per-marker and per-segment extents.
                for j in range(n):
                    artist, = ax.plot(j, values[s, j], marker="o", markersize=5, markeredgewidth=0.6,
                                      color=color, linestyle="none")
                    identity = f"mark:{s}:{j}"
                    artists[identity] = (artist, 0.3 * dpi / 72)
                    points[identity] = [float(j), float(values[s, j])]
                for j in range(n - 1):
                    artist, = ax.plot([j, j + 1], values[s, j:j + 2], color=color, linewidth=1.6)
                    identity = f"segment:{s}:{j}"
                    artists[identity] = (artist, 0.8 * dpi / 72)
                    segments.append((identity, s, j))
        handles = [Line2D([], [], color=template["palette"][s % len(template["palette"])], lw=3, label=label) for s, label in enumerate(data["series"])]
        legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.01), ncol=series_count, fontsize=7, framealpha=0.9)
        canvas.draw()
        actual_width, actual_height = canvas.get_width_height()
        if (actual_width, actual_height) != (width, height):
            raise ValueError("Canvas dimensions differ from requested resolution")
        renderer = canvas.get_renderer()
        pixels = np.asarray(canvas.buffer_rgba()).copy()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).convert("RGB").save(destination)
        nodes, edges = [], []

        def box(artist, pad=0.):
            extent = artist.get_window_extent(renderer)
            return [max(0., float(extent.x0 - pad)), max(0., float(height - extent.y1 - pad)),
                    min(float(width), float(extent.x1 + pad)), min(float(height), float(height - extent.y0 + pad))]

        def add(identity, kind, bounds, **attributes):
            nodes.append({"id": identity, "type": kind, "bbox": bounds, **attributes})

        def edge(source, target, relation):
            edges.append({"source": source, "target": target, "relation": relation})

        for axis, spine in (("x", "bottom"), ("y", "left")):
            add(f"axis:{axis}", "axis", box(ax.spines[spine], 0.5 * dpi / 72), axis=axis)
            label = ax.xaxis.label if axis == "x" else ax.yaxis.label
            add(f"axis_label:{axis}", "axis_label", box(label), text=label.get_text())
            edge(f"axis_label:{axis}", f"axis:{axis}", "labels")
            ticks = ax.xaxis.get_major_ticks() if axis == "x" else ax.yaxis.get_major_ticks()
            for j, tick in enumerate(ticks):
                identity = f"tick:{axis}:{j}"
                value = float(j) if axis == "x" else float(y_ticks[j])
                location = ax.transData.transform([value, low] if axis == "x" else [-0.65, value])
                add(identity, "tick", box(tick.tick1line, 0.5 * dpi / 72), axis=axis, axis_value=value,
                    anchor_px=[float(location[0]), float(height - location[1])],
                    numeric_value=value if axis == "y" else None)
                add(f"tick_label:{axis}:{j}", "tick_label", box(tick.label1), text=tick.label1.get_text())
                edge(identity, f"axis:{axis}", "belongs_to")
                edge(f"tick_label:{axis}:{j}", identity, "labels")
        add("title", "title", box(ax.title), text=ax.title.get_text())
        for identity, (artist, pad) in artists.items():
            if identity.startswith("mark:"):
                _, s, j = identity.split(":")
                s, j = int(s), int(j)
                location = ax.transData.transform(points[identity])
                add(identity, "point" if spec["chart_type"] == "line" else "bar", box(artist, pad),
                    series=s, category=j, numeric_value=float(values[s, j]), data_xy=points[identity],
                    anchor_px=[float(location[0]), float(height - location[1])])
                edge(identity, f"series:{s}", "belongs_to")
                for axis in ("x", "y"):
                    edge(identity, f"axis:{axis}", "maps_to_axis")
                edge(identity, f"tick:x:{j}", "maps_to_tick")
                if j:
                    edge(f"mark:{s}:{j-1}", identity, "next_in_series")
            else:
                _, s, j = identity.split(":")
                add(identity, "line_segment", box(artist, pad), series=int(s), category=int(j))
                edge(identity, f"series:{s}", "belongs_to")
                for k in (int(j), int(j) + 1):
                    edge(identity, f"mark:{s}:{k}", "connected_to")
        for s in range(series_count):
            bounds = np.array([node["bbox"] for node in nodes if node.get("series") == s])
            add(f"series:{s}", "series", [float(bounds[:, 0].min()), float(bounds[:, 1].min()), float(bounds[:, 2].max()), float(bounds[:, 3].max())], text=data["series"][s])
            add(f"legend:{s}", "legend_item", box(legend.get_texts()[s]), text=data["series"][s])
            edge(f"legend:{s}", f"series:{s}", "represents_series")
        marks = [node for node in nodes if node["type"] in ("bar", "point")]
        numerical = []
        for a in marks:
            for b in marks:
                if a["id"] != b["id"]:
                    relation = comparison(a["numeric_value"], b["numeric_value"])
                    item = {"source": a["id"], "target": b["id"], "relation": relation}
                    numerical.append(item)
                    edges.append(item)
        # The affine transform maps data coordinates directly to top-left canvas pixels.
        affine = ax.transData.get_affine().get_matrix()
        flip = np.array([[1, 0, 0], [0, -1, height], [0, 0, 1]])
        matrix = flip @ affine
        record = {"schema_version": 2, "source": "synthetic", "id": spec["id"], "split": spec["split"],
                  "chart_type": spec["chart_type"], "spec": spec, "image_path": destination.relative_to(Path(root)).as_posix(),
                  "image_sha256": sha256(destination), "width": width, "height": height,
                  "coordinate_convention": "continuous xyxy, top-left final PNG pixels; no crop or rescale",
                  "font": {"family": template["font"], "sha256": sha256(findfont(FontProperties(family=template["font"])))},
                  "axes": {"x": {"kind": "categorical", "categories": data["categories"], "limits": list(ax.get_xlim())},
                           "y": {"kind": "linear", "limits": [low, high], "ticks": y_ticks.tolist()},
                           "data_to_pixel": matrix.tolist(), "plot_bbox": box(ax)},
                  "nodes": nodes, "edges": edges, "numerical_relationships": numerical}
        record["questions"] = make_questions(record)
        fig.clear()
        return record


def make_questions(record):
    nodes = {node["id"]: node for node in record["nodes"]}
    marks = [key for key, node in nodes.items() if node["type"] in ("bar", "point")]
    first, second = marks[:2]
    values = lambda operands: [nodes[key]["numeric_value"] for key in operands]
    definitions = [
        ("lookup", [first], values([first])[0], "What is the value of Series 1 at C1?"),
        ("maximum", marks, max(values(marks)), "What is the maximum data value?"),
        ("minimum", marks, min(values(marks)), "What is the minimum data value?"),
        ("difference", [first, second], abs(values([first])[0] - values([second])[0]), "What is the absolute difference between Series 1 at C1 and C2?"),
        ("comparison", [first, second], values([first])[0] > values([second])[0], "Is Series 1 at C1 greater than at C2?"),
        ("count", [key for key in marks if nodes[key]["series"] == 0], len(record["spec"]["data"]["categories"]), "How many observations belong to Series 1?")]
    if len(record["spec"]["data"]["series"]) > 1:
        operands = [key for key in marks if nodes[key]["series"] in (0, 1)]
        data = record["spec"]["data"]["values"]
        definitions.append(("series_difference", operands, sum(data[0]) - sum(data[1]), "What is the sum of Series 1 minus the sum of Series 2?"))
    calibration = [key for key in nodes if key.startswith(("tick:y:", "tick_label:y:", "axis:", "axis_label:"))]
    result = []
    for index, (operation, operands, answer, question) in enumerate(definitions):
        context = [f"tick_label:x:{nodes[key]['category']}" for key in operands]
        context += [f"legend:{nodes[key]['series']}" for key in operands]
        evidence = list(dict.fromkeys(operands + calibration + context))
        result.append({"id": f"{record['id']}:q{index}", "operation": operation, "question": question,
                       "operands": operands, "answer": answer, "evidence_nodes": evidence,
                       "evidence_boxes": [{"node_id": key, "bbox": nodes[key]["bbox"]} for key in evidence]})
    return result


def generate_dataset(root, config):
    root = Path(root).resolve()
    output = root / config["output"]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already contains files; choose a new versioned output")
    from .structural_validation import validate_record
    records = []
    for index in range(config["count"]):
        spec = make_spec(config, index)
        record = render_chart(spec, output / "images" / f"{spec['id']}.png", root)
        errors = validate_record(record, root)
        if errors:
            raise ValueError(f"Invalid generated chart {spec['id']}: {errors}")
        write_json(output / "annotations" / f"{spec['id']}.json", record)
        records.append(record)
    for split in config["splits"]:
        write_jsonl(output / f"{split}.jsonl", [record for record in records if record["split"] == split])
    write_json(output / "manifest.json", {"schema_version": 2, "config": config, "config_sha256": fingerprint(config),
                                         "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                                         "records": [{"id": r["id"], "split": r["split"], "image_sha256": r["image_sha256"],
                                                      "annotation_sha256": sha256(output / "annotations" / f"{r['id']}.json")} for r in records]})
    return records
