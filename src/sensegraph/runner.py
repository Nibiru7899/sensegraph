import json
from pathlib import Path
import time

from .io import read_json, read_jsonl, sha256, write_json
from .models import LocalModel
from .reasoning import execute, parse_table, RULE_VERSION


def run_baseline(root, config, name, smoke=False):
    root = Path(root)
    records_path = root / "data/normalized/pilot.jsonl"
    records = read_jsonl(records_path)
    if not records:
        raise ValueError("Empty pilot")
    run_dir = root / "experiments" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = "smoke" if smoke else "run"
    summary_path = run_dir / f"{prefix}.json"
    prediction_path = run_dir / ("smoke_predictions.jsonl" if smoke else "predictions.jsonl")
    if prediction_path.exists():
        raise ValueError(f"Output already exists: {prediction_path}; preserve it and choose a new run directory")
    if not smoke:
        smoke_path = run_dir / "smoke.json"
        if not smoke_path.exists() or not read_json(smoke_path).get("complete") or read_json(smoke_path).get("error"):
            raise ValueError("Complete a real-image smoke check before the pilot")
        smoke_predictions = read_jsonl(run_dir / "smoke_predictions.jsonl")
        if not smoke_predictions or not smoke_predictions[0].get("raw_output") or smoke_predictions[0].get("error"):
            raise ValueError("Smoke inference failed; inspect its saved output before running the pilot")
    budget_path = root / "experiments/gpu_budget.json"
    budget = read_json(budget_path) if budget_path.exists() else {"charged_seconds": 0, "sessions": []}
    allowance = min(300 if smoke else config["model_seconds"], config["total_gpu_seconds"] - budget["charged_seconds"])
    if allowance < 120:
        raise RuntimeError("Insufficient GPU budget for another bounded session")
    # Reserve before loading so interrupted/killed sessions cannot silently reset the budget.
    session = {"model": name, "kind": prefix, "reserved_seconds": allowance, "actual_seconds": None}
    budget["sessions"].append(session)
    budget["charged_seconds"] += allowance
    write_json(budget_path, budget)
    started = time.monotonic()
    summary = {"model": name, "checkpoint": config["models"][name], "config": config,
               "records_sha256": sha256(records_path), "manifest_sha256": sha256(root / "data/manifests/pilot.json"),
               "rule_version": RULE_VERSION, "rules_sha256": sha256(Path(__file__).with_name("reasoning.py")),
               "complete": False, "processed": 0, "expected": 1 if smoke else len(records),
               "evidence_localization": None, "calibrated_confidence": None, "kind": "real_model_inference"}
    summary["implementation_sha256"] = {filename: sha256(Path(__file__).with_name(filename)) for filename in ("models.py", "runner.py", "reasoning.py")}
    write_json(summary_path, summary)
    model = None
    try:
        model = LocalModel(root, name, config)
        if name == "deplot":
            summary["prompt_font"] = {"source": str(model.font_path), "sha256": model.font_sha256,
                                      "note": "Locally installed font; not downloaded or redistributed"}
        torch = model.torch
        torch.cuda.reset_peak_memory_stats()
        cache = {}
        with prediction_path.open("w", encoding="utf-8") as stream:
            for record in records[:1] if smoke else records:
                remaining = allowance - (time.monotonic() - started)
                if remaining < 90:
                    summary["stop_reason"] = "time_budget"
                    break
                # Deliberate inference boundary: image path and question only.
                image_path = root / record["image_path"]
                if sha256(image_path) != record["image_sha256"]:
                    raise ValueError("Input image changed after normalization")
                prediction = {"id": record["id"], "answer": None, "status": "error", "confidence": None,
                              "evidence_boxes": None, "graph_nodes": None}
                torch.cuda.synchronize()
                before = time.monotonic()
                try:
                    cached = name == "deplot" and record["image_sha256"] in cache
                    if cached:
                        text, generation = cache[record["image_sha256"]]
                    else:
                        text = model.predict(image_path, record["question"], min(60, remaining - 20))
                        generation = dict(model.last_generation)
                    prediction["raw_output"] = text
                    prediction["cached_extraction"] = cached
                    prediction["generation"] = generation
                    if not generation["completed"]:
                        raise ValueError("Generation stopped before EOS; output retained but not scored as an answer")
                    if name == "deplot":
                        cache[record["image_sha256"]] = (text, generation)
                        table = parse_table(text)
                        prediction["predicted_table"] = table
                        prediction.update(execute(record["question"], table))
                    else:
                        prediction.update(answer=text, status="ok")
                except (ValueError, RuntimeError) as error:
                    prediction["error"] = f"{type(error).__name__}: {error}"
                    if "out of memory" in str(error).lower():
                        torch.cuda.empty_cache()
                        summary["stop_reason"] = "cuda_out_of_memory"
                torch.cuda.synchronize()
                prediction.update(latency_seconds=time.monotonic() - before, precision=str(model.dtype),
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                stream.flush()
                summary["processed"] += 1
                print(json.dumps({"model": name, "processed": summary["processed"], "status": prediction["status"]}), flush=True)
                write_json(summary_path, summary)
                if summary.get("stop_reason") == "cuda_out_of_memory":
                    break
        summary["complete"] = summary["processed"] == summary["expected"]
        summary["precision"] = str(model.dtype)
        summary["precision_fallback"] = model.precision_fallback
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        elapsed = time.monotonic() - started
        summary["elapsed_seconds"] = elapsed
        write_json(summary_path, summary)
        session["actual_seconds"] = elapsed
        budget["charged_seconds"] += elapsed - allowance
        write_json(budget_path, budget)
    return summary
