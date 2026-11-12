"""Source-supported real annotation mappings; no inferred raster transforms."""
from pathlib import Path
import zipfile

from PIL import Image

from .acquisition import Downloader, RemoteZipReader
from .io import read_json, read_jsonl, sha256, write_json

CHARTQA_FULL_REVISION = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"


def acquire_chartqa_training_annotations(root):
    root = Path(root).resolve()
    downloader = Downloader(root)
    metadata = downloader.json("https://huggingface.co/api/datasets/ahmed-masry/ChartQA", "m2-chartqa-metadata")
    if metadata["sha"] != CHARTQA_FULL_REVISION:
        raise ValueError("Full ChartQA source revision differs from the inspected release")
    if metadata.get("cardData", {}).get("license") != "gpl-3.0":
        raise ValueError("Unexpected full-dataset license metadata")
    revision = metadata["sha"]
    url = f"https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/{revision}/ChartQA%20Dataset.zip"
    reader = RemoteZipReader(downloader, url)
    records = read_jsonl(root / "data/normalized/training.jsonl")
    names = sorted({r["image_name"] for r in records})
    files = []
    with zipfile.ZipFile(reader) as archive:
        for name in names:
            for directory, filename in (("annotations", Path(name).stem + ".json"), ("png", name)):
                member = f"ChartQA Dataset/train/{directory}/{filename}"
                destination = root / "data/real_annotations/chartqa/train" / directory / filename
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(member))
                files.append({"member": member, "path": destination.relative_to(root).as_posix(),
                              "sha256": sha256(destination), "bytes": destination.stat().st_size,
                              "crc32": archive.getinfo(member).CRC})
    manifest = {"dataset": "ChartQA full", "revision": revision, "license": "gpl-3.0 author metadata; third-party image rights retained",
                "archive": reader.metadata, "split": "train", "files": files,
                "note": "Only the existing 20 training identities; no benchmark test payloads fetched"}
    write_json(root / "data/real_annotations/chartqa/source_manifest.json", manifest)
    return manifest


def vistext_local_nodes(scenegraph):
    """Accumulate explicit group translations, preserving local geometry only."""
    result = []
    def walk(mark, offset, path):
        kind = mark.get("marktype")
        if kind == "group":
            for i, item in enumerate(mark.get("items", [])):
                if item.get("angle", 0) != 0 or item.get("scaleX", 1) != 1 or item.get("scaleY", 1) != 1 or item.get("transform") is not None:
                    result.append({"id": f"sg:{path}/{i}", "type": "unsupported_group_transform", "role": mark.get("role"),
                                   "point_chart_xy": None, "bbox_chart_xyxy": None, "bbox": None,
                                   "raster_mapping": "unavailable", "source_attributes": {k: v for k, v in item.items() if k != "items"}})
                    continue
                origin = [offset[0] + item.get("x", 0), offset[1] + item.get("y", 0)]
                for j, child in enumerate(item.get("items", [])):
                    walk(child, origin, f"{path}/{i}/{j}")
        else:
            for i, item in enumerate(mark.get("items", [])):
                x, y = item.get("x"), item.get("y")
                point = [offset[0] + x, offset[1] + y] if x is not None and y is not None else None
                bounds = None
                if kind == "rect" and point is not None and item.get("width", 0) > 0 and item.get("height", 0) > 0:
                    bounds = [*point, point[0] + item["width"], point[1] + item["height"]]
                result.append({"id": f"sg:{path}/{i}", "type": kind or "unknown", "role": mark.get("role"),
                               "point_chart_xy": point, "bbox_chart_xyxy": bounds,
                               "bbox": None, "raster_mapping": "unavailable", "source_attributes": item})
    walk(scenegraph, [0, 0], "root")
    return result


def chartqa_pixel_annotations(annotation, image_size):
    """Convert the paired full-release xywh fields; never zip segment boxes to points."""
    width, height = image_size
    if annotation.get("type") not in ("h_bar", "v_bar", "line"):
        raise ValueError("Unsupported ChartQA chart type")
    nodes, edges, issues = [], [], []
    def add(identity, kind, raw_box=None, **attrs):
        bounds = None
        if raw_box is not None:
            x, y, w, h = [float(raw_box[key]) for key in ("x", "y", "w", "h")]
            import math
            if not all(math.isfinite(v) for v in (x, y, w, h)) or x < 0 or y < 0 or w < 0 or h < 0 or x + w > width or y + h > height:
                issues.append({"node": identity, "reason": "out-of-image or malformed source box", "source_bbox": raw_box})
            elif w == 0 or h == 0:
                issues.append({"node": identity, "reason": "zero-area source box retained without invented padding", "source_bbox": raw_box})
            else:
                bounds = [x, y, x + w, y + h]
        nodes.append({"id": identity, "type": kind, "bbox": bounds, "source_bbox": raw_box, **attrs})
    info = annotation["general_figure_info"]
    add("plot_area", "plot_area", info.get("figure_info", {}).get("bbox"))
    if "title" in info:
        add("title", "title", info["title"].get("bbox"), text=info["title"].get("text"))
    for axis in ("x_axis", "y_axis"):
        labels = info.get(axis, {}).get("major_labels", {})
        boxes, values = labels.get("bboxes", []), labels.get("values", [])
        if len(boxes) != len(values):
            raise ValueError("Unaligned axis labels and boxes")
        for i, (box, value) in enumerate(zip(boxes, values)):
            add(f"{axis}:label:{i}", "tick_label", box, text=str(value), axis_key=axis)
    for s, model in enumerate(annotation["models"]):
        add(f"series:{s}", "series", name=model.get("name"))
        boxes, xs, ys = model.get("bboxes", []), model.get("x", []), model.get("y", [])
        line = annotation["type"] == "line"
        if len(xs) != len(ys) or len(boxes) != len(xs) - int(line):
            raise ValueError("Ambiguous point/segment alignment in source model")
        for j, box in enumerate(boxes):
            attrs = {"series": s, "source_ordinal": j}
            if line:
                attrs["data_endpoints"] = {"x": xs[j:j + 2], "y": ys[j:j + 2]}
                attrs["numeric_value"] = None
            else:
                attrs.update(category=xs[j], source_value=ys[j])
            add(f"mark:{s}:{j}", "line_segment" if line else "bar", box, **attrs)
            edges.append({"source": f"mark:{s}:{j}", "target": f"series:{s}", "relation": "belongs_to", "provenance": "source models grouping"})
    names = [model.get("name") for model in annotation["models"]]
    for i, item in enumerate(info.get("legend", {}).get("items", [])):
        label = item.get("label", {})
        add(f"legend:{i}", "legend_item", label.get("bbox"), text=label.get("text"))
        if names.count(item.get("model")) == 1:
            edges.append({"source": f"legend:{i}", "target": f"series:{names.index(item['model'])}", "relation": "represents_series", "provenance": "source legend model name"})
    return {"schema_version": "real-partial-1", "chart_type": annotation["type"], "nodes": nodes, "edges": edges,
            "coordinate_mapping": "paired source xywh -> top-left pixel xyxy, identity scale",
            "issues": issues, "missing": ["axis lines/tick marks where absent", "point boxes for line segments", "QA-specific evidence", "validated numerical calibration"]}


def map_real_annotations(root):
    root = Path(root).resolve()
    from .structural_validation import draw_overlay
    report = {"chartqa": [], "vistext": [], "limitations": ["Real mappings are partial and are not fed into the synthetic supervised dataset", "No invented QA evidence or inferred VisText pixel transform"]}
    for path in sorted((root / "data/real_annotations/chartqa/train/annotations").glob("*.json")):
        image = root / "data/real_annotations/chartqa/train/png" / (path.stem + ".png")
        try:
            with Image.open(image) as im:
                converted = chartqa_pixel_annotations(read_json(path), im.size)
        except ValueError as error:
            report["chartqa"].append({"id": path.stem, "status": "unsupported", "reason": str(error)})
            continue
        converted.update(image_path=image.relative_to(root).as_posix(), image_sha256=sha256(image), source_annotation_sha256=sha256(path))
        destination = root / "data/real_annotations/chartqa/converted" / path.name
        write_json(destination, converted)
        draw_overlay(image, converted["nodes"], root / f"experiments/milestone2/real_overlays/chartqa_{path.stem}.png",
                     {"bar", "line_segment", "tick_label", "title", "legend_item"})
        old = root / "data/raw/chartqa/train/png" / image.name
        report["chartqa"].append({"id": path.stem, "chart_type": converted["chart_type"], "nodes": len(converted["nodes"]),
                                   "pixel_boxes": sum(n["bbox"] is not None for n in converted["nodes"]),
                                   "issues": converted["issues"], "matches_baseline_image": sha256(old) == sha256(image)})
    for row in read_jsonl(root / "data/normalized/vistext_training.jsonl"):
        nodes = vistext_local_nodes(row["scenegraph"])
        write_json(root / f"data/real_annotations/vistext/{Path(row['image_name']).stem}.json", {
            "id": row["id"], "schema_version": "vistext-local-1", "image_path": row["image_path"], "nodes": nodes,
            "raster_transform": None, "source_image_sha256": row["image_sha256"],
            "note": "Explicit group translations only; origin/scale/padding to raster are unavailable"})
        # A coordinate preview deliberately has no real raster background.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure = Figure(figsize=(6, 5), dpi=100)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot()
        points = [n["point_chart_xy"] for n in nodes if n["point_chart_xy"] is not None]
        if points:
            axes.scatter([p[0] for p in points], [p[1] for p in points], s=8)
        axes.invert_yaxis()
        axes.set_title(f"{row['id']}: LOCAL coordinates only", fontsize=10)
        axes.set_xlabel("chart-local x; NOT raster pixels")
        axes.set_ylabel("chart-local y")
        destination = root / f"experiments/milestone2/real_overlays/vistext_{Path(row['image_name']).stem}_local.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination)
        figure.clear()
        descriptions = [n["source_attributes"].get("description", "") for n in nodes]
        report["vistext"].append({"id": row["id"], "nodes": len(nodes), "local_rectangles": sum(n["bbox_chart_xyxy"] is not None for n in nodes),
                                  "pixel_boxes": 0, "unsupported_group_transforms": sum(n["type"] == "unsupported_group_transform" for n in nodes),
                                  "temporal_description_examples": [d for d in descriptions if "Dec 31" in d][:2]})
    write_json(root / "experiments/milestone2/real_mapping.json", report)
    return report
