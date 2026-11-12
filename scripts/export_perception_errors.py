"""CPU-only export of the actual frozen detector validation errors; no refitting."""
from pathlib import Path
import sys
import csv
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensegraph.io import read_json, read_jsonl, write_json, sha256
from sensegraph.perception import match_nodes, detection_target, CLASSES


def main():
    directory = ROOT / "experiments/milestone4"
    result = read_json(directory / "result.json")
    config = read_json(ROOT / "configs/perception.json")
    saved = torch.load(directory / "validation.pt", weights_only=True, map_location="cpu")
    records = read_jsonl(ROOT / "data/synthetic/v3/val.jsonl")[:result["validation_images"]]
    errors = []
    for record, raw, target in zip(records, saved["predictions"], saved["targets"]):
        pred = {key: value[raw["scores"] >= config["score_threshold"]] for key, value in raw.items()}
        matches, overlaps = match_nodes(pred, target, .5)
        _, ids, rejected = detection_target(record)
        matched = set(matches[matches >= 0].tolist())
        missed = [{"node_id": ids[i], "class": CLASSES[int(target["labels"][i])], "bbox": target["boxes"][i].tolist()}
                  for i in range(len(ids)) if i not in matched]
        false = []
        for i in torch.where(matches < 0)[0].tolist():
            label = int(pred["labels"][i])
            duplicate = any(float(overlaps[i, j]) >= .5 for j in torch.where(target["labels"] == label)[0].tolist())
            false.append({"prediction_index_after_threshold": i, "class": CLASSES[label], "score": float(pred["scores"][i]),
                          "bbox": pred["boxes"][i].tolist(), "reason": "duplicate" if duplicate else "unmatched_detection"})
        errors.append({"id": record["id"], "matched": len(matched), "supported_gt": len(ids),
                       "missed_supported_annotations": missed, "false_positive_predictions": false,
                       "unsupported_or_invalid_annotations": rejected})
    write_json(directory / "detection_errors.json", {"iou": .5, "score_threshold": config["score_threshold"], "images": errors})
    rows = [{"iou": iou, "class": name, **values} for iou, data in result["metrics"].items() for name, values in data["per_class"].items()]
    with (directory / "detection_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    weights = read_json(directory / "weights.json")
    write_json(directory / "checkpoint_provenance.json", {"official_source": weights,
               "trained_checkpoint_sha256": sha256(directory / "detector.pt"),
               "trained_checkpoint_bytes": (directory / "detector.pt").stat().st_size,
               "frozen_sources": read_json(directory / "freeze.json")["sources"],
               "training_steps": result["steps"], "validation_images": result["validation_images"],
               "protected_test_evaluated": result["test_evaluated"]})
    print({"images_exported": len(errors), "matched": sum(e["matched"] for e in errors),
           "supported_gt": sum(e["supported_gt"] for e in errors)})


if __name__ == "__main__":
    main()
