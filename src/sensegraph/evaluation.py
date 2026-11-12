"""Answer scoring, failed-run denominators, and paired candidate comparisons."""
import math
import statistics
from pathlib import Path

from .io import read_json, read_jsonl, sha256, write_json
from .reasoning import intent, number


def exact_match(answer, reference):
    return str(answer).strip() == str(reference).strip()


def chartqa_relaxed(answer, reference):
    # ChartQA official evaluation: numeric conversion, percentage scaling,
    # <=5% relative error for nonzero targets; zero falls back to string match.
    def numeric(text):
        try:
            text = str(text)
            value = float(text.rstrip("%")) / (100 if text.endswith("%") else 1)
            return value
        except ValueError:
            return None
    pred, target = numeric(answer), numeric(reference)
    if pred is not None and target is not None and target != 0:
        return abs(pred - target) / abs(target) <= 0.05
    return str(answer).lower() == str(reference).lower()


def score_records(records, predictions):
    known = {r["id"] for r in records}
    ids = [p["id"] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) - known:
        raise ValueError("Duplicate or unknown prediction IDs")
    indexed = {p["id"]: p for p in predictions}
    def group(rows):
        exact, relaxed, answered = 0, 0, 0
        for row in rows:
            pred = indexed.get(row["id"], {})
            if pred.get("status") != "ok" or pred.get("answer") is None:
                continue
            answered += 1
            exact += any(exact_match(pred["answer"], ref) for ref in row["answers"])
            relaxed += any(chartqa_relaxed(pred["answer"], ref) for ref in row["answers"])
        return {"denominator": len(rows), "answered": answered, "exact_correct": exact, "relaxed_correct": relaxed,
                "exact_accuracy": exact / len(rows) if rows else None,
                "relaxed_accuracy": relaxed / len(rows) if rows else None,
                "coverage": answered / len(rows) if rows else None}
    return {"overall": group(records), "partitions": {partition: group([r for r in records if r["partition"] == partition]) for partition in sorted({r["partition"] for r in records})},
            "missing_predictions": sorted(known - set(ids))}


def score_run(root, model):
    root = Path(root)
    records = read_jsonl(root / "data/normalized/pilot.jsonl")
    predictions = read_jsonl(root / f"experiments/{model}/predictions.jsonl")
    summary = read_json(root / f"experiments/{model}/run.json")
    if summary["records_sha256"] != sha256(root / "data/normalized/pilot.jsonl"):
        raise ValueError("Evaluation data changed since inference")
    report = {"model": model, "run": summary, **score_records(records, predictions)}
    report["scoring_sha256"] = sha256(Path(__file__))
    report["metric_definition"] = {
        "exact": "case-sensitive equality after trimming outer whitespace",
        "relaxed": "Pix2Struct official ChartQA relaxed_correctness: percent-to-fraction conversion, <=5% relative error, lowercase string fallback for zero/nonnumeric targets",
        "source": "https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py"
    }
    latencies = sorted(p["latency_seconds"] for p in predictions if "latency_seconds" in p)
    report["latency"] = {"count": len(latencies), "median_seconds": statistics.median(latencies) if latencies else None,
                         "p95_seconds": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else None,
                         "includes": "per-question preprocessing, decoding, and executor; cached extraction explicitly marked"}
    if model == "deplot":
        indexed = {r["id"]: r for r in records}
        per_image = {}
        for prediction in predictions:
            record = indexed[prediction["id"]]
            digest = record["image_sha256"]
            if digest not in per_image:
                per_image[digest] = {"record_id": record["id"], **aligned_numeric_error(prediction.get("predicted_table"), record.get("table"))}
        errors = list(per_image.values())
        aligned = [item for item in errors if item["status"] == "aligned"]
        cells = sum(item["cells"] for item in aligned)
        report["numeric_extraction"] = {"alignment": "exact case-insensitive headers and row order; percentages in percentage points",
                                        "aligned_tables": len(aligned), "evaluated_tables": len(errors), "aligned_numeric_cells": cells,
                                        "missing_numeric_cells": sum(item.get("missing_numeric_cells", 0) for item in errors),
                                        "mae": sum(item["mae"] * item["cells"] for item in aligned) / cells if cells else None,
                                        "units": "mixed source units; conditional on parsed cells; descriptive only, not a normalized cross-chart score",
                                        "aggregation": "unique image SHA256, each chart once", "per_image": per_image}
    other = "deplot" if model == "unichart" else "unichart"
    other_path = root / f"experiments/{other}/predictions.jsonl"
    if other_path.exists():
        other_summary = read_json(root / f"experiments/{other}/run.json")
        if other_summary["records_sha256"] != summary["records_sha256"]:
            raise ValueError("Paired runs used different records")
        if other_summary["rules_sha256"] != summary["rules_sha256"]:
            raise ValueError("Paired runs used different executor rules")
        # Fixed question-only lexical candidates, independent of model success and labels.
        candidates = [r for r in records if intent(r["question"]) is not None]
        report["paired"] = {"selection": "rule-v1 question-only operation candidates; unresolved operands and parsing failures remain in denominator",
                            "candidate_count": len(candidates), "manifest_count": len(records),
                            model: score_records(candidates, [p for p in predictions if p["id"] in {r["id"] for r in candidates}]),
                            other: score_records(candidates, [p for p in read_jsonl(other_path) if p["id"] in {r["id"] for r in candidates}])}
    write_json(root / f"experiments/{model}/metrics.json", report)
    return report


def aligned_numeric_error(predicted, reference):
    # Strict labels and order prevent optimistic matching of unrelated cells.
    if not predicted or not reference or len(predicted) != len(reference):
        return {"status": "unaligned", "mae": None, "cells": 0}
    if [c.casefold().strip() for c in predicted[0]] != [c.casefold().strip() for c in reference[0]]:
        return {"status": "unaligned", "mae": None, "cells": 0}
    errors, missing = [], 0
    for p, r in zip(predicted[1:], reference[1:]):
        if len(p) != len(r) or p[0].casefold().strip() != r[0].casefold().strip():
            return {"status": "unaligned", "mae": None, "cells": 0}
        for pv, rv in zip(p[1:], r[1:]):
            a, b = number(pv), number(rv)
            if a is not None and b is not None:
                errors.append(float(abs(a - b)))
            elif b is not None:
                missing += 1
    return {"status": "aligned" if errors else "no_numeric_cells", "mae": sum(errors) / len(errors) if errors else None,
            "cells": len(errors), "missing_numeric_cells": missing}
