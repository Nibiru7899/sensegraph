import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .io import read_json, read_jsonl, sha256, write_json


def baseline_integrity(root):
    guard = read_json(root / "experiments/milestone2/baseline_guard.json")
    allowed = {"experiments/download_ledger.json"}
    changes = [path for path, digest in guard.items() if path not in allowed and (not (root / path).exists() or sha256(root / path) != digest)]
    from .evaluation import score_records
    rows = read_jsonl(root / "data/normalized/pilot.jsonl")
    original = read_json(root / "research/pilot_results.json")
    scores = {}
    for model in ("unichart", "deplot"):
        current = score_records(rows, read_jsonl(root / f"experiments/{model}/predictions.jsonl"))
        scores[model] = {"overall": current["overall"], "matches_baseline": current["overall"] == original["metrics"][model]["overall"]}
    return {"changed_protected_files": changes, "protected_files": len(guard) - len(allowed),
            "allowed_bookkeeping_updates": sorted(allowed), "recomputed_baseline_scores": scores}


def validate_dataset(root, config):
    from .structural_validation import validate_record, draw_overlay
    from .graph_data import StructuralDataset, collate_charts
    from .leakage import leakage_report
    from torch.utils.data import DataLoader
    import torch

    output = root / config["output"]
    records = sum([read_jsonl(output / f"{split}.jsonl") for split in config["splits"]], [])
    invalid = []
    for record in records:
        errors = validate_record(record, root)
        if errors:
            invalid.append({"id": record["id"], "errors": errors})
    manifest = read_json(output / "manifest.json")
    indexed = {r["id"]: r for r in records}
    for item in manifest["records"]:
        annotation = output / "annotations" / f"{item['id']}.json"
        if item["id"] not in indexed or sha256(annotation) != item["annotation_sha256"] or read_json(annotation) != indexed[item["id"]]:
            invalid.append({"id": item["id"], "errors": ["annotation file/manifest/partition record disagreement"]})
    real = sum([read_jsonl(root / f"data/normalized/{name}.jsonl") for name in ("training", "pilot", "vistext_training")], [])
    leakage = leakage_report(records, root, config["near_duplicate_hamming_threshold"], real)
    write_json(root / "experiments/milestone2/leakage.json", leakage)
    # Choose one example of each chart type in each split for reproducible visual QA.
    representatives = []
    for split in config["splits"]:
        for chart_type in config["chart_types"]:
            record = next(r for r in records if r["split"] == split and r["chart_type"] == chart_type)
            for layer, types in (("marks", {"bar", "point", "line_segment"}), ("context", {"axis", "tick", "tick_label", "title", "legend_item", "axis_label"})):
                destination = root / f"experiments/milestone2/overlays/{record['id']}_{layer}.png"
                draw_overlay(root / record["image_path"], record["nodes"], destination, types)
            representatives.append(record["id"])
            evidence_ids = set(record["questions"][0]["evidence_nodes"])
            draw_overlay(root / record["image_path"], [node for node in record["nodes"] if node["id"] in evidence_ids],
                         root / f"experiments/milestone2/overlays/{record['id']}_evidence.png")
    dataset = StructuralDataset(output / "train.jsonl", root)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_charts)))
    inputs, targets = batch["inputs"], batch["targets"]
    batch_report = {"images": list(inputs["images"].shape), "valid_pixels": int(inputs["image_mask"].sum()),
                    "node_features": list(inputs["graph"].x.shape), "candidate_edges": list(inputs["graph"].edge_index.shape),
                    "edge_labels": list(targets.edge_y.shape), "node_types": list(targets.node_type_y.shape),
                    "numerical_labels": list(targets.value_y.shape), "evidence_index": list(targets.evidence_index.shape),
                    "questions": targets.question_op_y.numel(), "finite_features": bool(torch.isfinite(inputs["graph"].x).all()),
                    "input_graph_keys": sorted(inputs["graph"].keys()), "metadata": batch["metadata"], "device": "cpu"}
    write_json(root / "experiments/milestone2/batch_smoke.json", batch_report)
    loaded = {}
    for split in config["splits"]:
        count, batches = 0, 0
        split_data = StructuralDataset(output / f"{split}.jsonl", root)
        for combined in DataLoader(split_data, batch_size=4, collate_fn=collate_charts):
            if not torch.isfinite(combined["inputs"]["graph"].x).all() or not torch.isfinite(combined["targets"].value_y).all():
                raise ValueError("Nonfinite training tensors")
            count += len(combined["metadata"])
            batches += 1
        loaded[split] = {"loaded_charts": count, "batches": batches}
    counts = [len(r["nodes"]) for r in records]
    qas = [q for r in records for q in r["questions"]]
    report = {"schema_version": 2, "count": len(records), "splits": dict(Counter(r["split"] for r in records)),
              "chart_types": dict(Counter(r["chart_type"] for r in records)),
              "resolutions_by_chart_type": {kind: dict(Counter(f"{r['width']}x{r['height']}" for r in records if r["chart_type"] == kind)) for kind in config["chart_types"]},
              "node_counts": {"min": min(counts), "max": max(counts), "mean": statistics.mean(counts), "total": sum(counts)},
              "node_types": dict(Counter(n["type"] for r in records for n in r["nodes"])),
              "relations": dict(Counter(e["relation"] for r in records for e in r["edges"])),
              "qa_operations": dict(Counter(q["operation"] for q in qas)), "question_count": len(qas),
              "evidence_coverage": sum(bool(q["evidence_boxes"]) for q in qas) / len(qas),
              "annotation_fields": {"bbox": "all synthetic nodes", "numeric_value": "data marks and numeric y ticks", "numeric_missing": "masked, never treated as zero labels"},
              "invalid_samples": invalid, "visual_review_ids": representatives, "leakage": leakage,
              "batch_smoke": batch_report, "all_split_loading": loaded, "baseline_integrity": baseline_integrity(root),
              "limitations": ["100-chart pipeline validation, not training or generalization evidence", "Renderer GT proposals are oracle training inputs, not a detector", "No cached learned visual features or trained GNN"]}
    write_json(root / "experiments/milestone2/dataset_statistics.json", report)
    if invalid or leakage["violations"] or report["baseline_integrity"]["changed_protected_files"]:
        raise ValueError("Structural validation or baseline integrity failed; inspect statistics")
    return report


def main():
    parser = argparse.ArgumentParser(prog="sensegraph-structure")
    parser.add_argument("command", choices=("generate", "validate", "acquire-real", "map-real"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/structural.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        config = read_json(root / args.config)
        if args.command == "generate":
            from .synthetic import generate_dataset
            rows = generate_dataset(root, config)
            result = {"validated_generated_charts": len(rows), "output": config["output"]}
        elif args.command == "validate":
            import torch
            torch.set_num_threads(1)
            result = validate_dataset(root, config)
        elif args.command == "acquire-real":
            from .real_mapping import acquire_chartqa_training_annotations
            result = acquire_chartqa_training_annotations(root)
        else:
            from .real_mapping import map_real_annotations
            result = map_real_annotations(root)
        print(json.dumps(result, indent=2))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
