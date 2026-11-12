"""Protocol-first orchestration. Test access requires a sealed experiment inventory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

from .io import read_json, read_jsonl, sha256, write_json
from .synthetic import fingerprint, generate_dataset, RELATIONS
from .relation_models import RelationNet, pack_samples
from .relation_metrics import confusion, metrics, paired_bootstrap
from .relation_training import cached_split, train_run, rule_validation, load_checkpoint, infer, GPUSession, seed_everything


def history_check(root):
    guard = read_json(root / "experiments/milestone3/history_guard.json")
    changed = [name for name, digest in guard.items() if not (root / name).is_file() or sha256(root / name) != digest]
    if changed:
        raise ValueError(f"Historical files changed: {changed[:5]}")
    return {"protected_files": len(guard), "changed": changed}


def source_hashes(root):
    names = [f"src/sensegraph/{name}.py" for name in ("relation_models", "relation_metrics", "relation_training", "relation_experiment")]
    names += ["configs/relations.json", "configs/structural_v3.json", "research/milestone3_protocol.md"]
    return {name: sha256(root / name) for name in names}


def freeze(root, config):
    directory = root / "experiments/milestone3"
    target = directory / "protocol_freeze.json"
    hashes = source_hashes(root)
    counts = {name: sum(p.numel() for p in RelationNet(name, config).parameters()) for name in ("mlp", "gat")}
    if max(counts.values()) / min(counts.values()) > 1.25:
        raise ValueError("Main neural model parameter counts differ by >25%")
    payload = {"source_sha256": hashes, "config": config, "parameter_counts": counts,
               "v2_split_sha256": {s: sha256(root / f"data/synthetic/v2/{s}.jsonl") for s in ("train", "val", "test")},
               "hardware": {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
                            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
               "history_integrity": history_check(root), "test_evaluation_permitted": False}
    if target.exists() and read_json(target) != payload:
        if (directory / "sealed_inventory.json").exists():
            raise ValueError("Protocol cannot change after test inventory sealing")
        old = read_json(target)
        write_json(directory / f"protocol_history/{fingerprint(old)}.json", old)
    write_json(target, payload)
    snapshot = directory / "source_snapshots" / fingerprint(hashes)
    for name in hashes:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    return payload


def require_frozen(root):
    record = read_json(root / "experiments/milestone3/protocol_freeze.json")
    if record["source_sha256"] != source_hashes(root):
        raise ValueError("Implementation/protocol changed; re-freeze before test access")
    history_check(root)
    return record


def safe_train(root, dataset, architecture, config, seed=42, variant="main", overfit=False):
    if (root / "experiments/milestone3/sealed_inventory.json").exists():
        raise ValueError("Training is forbidden after final-test inventory sealing")
    require_frozen(root)
    name = f"{dataset}_{architecture}_{variant}_s{seed}" + ("_overfit" if overfit else "")
    result_path = root / f"experiments/milestone3/runs/{name}/result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result.get("complete"):
            return result
        raise ValueError(f"Prior incomplete run must be preserved and investigated: {name}")
    try:
        result = train_run(root, dataset, architecture, config, seed, variant, overfit)
        result["protocol_sha256"] = sha256(root / "experiments/milestone3/protocol_freeze.json")
        result["implementation_sha256"] = source_hashes(root)
        write_json(result_path, result)
        return result
    except Exception as error:
        failure = {"name": name, "error": f"{type(error).__name__}: {error}", "complete": False}
        write_json(result_path.parent / "failure.json", failure)
        raise


def verify_v2(root, config):
    results = []
    for architecture in ("mlp", "gat"):
        smoke = safe_train(root, "v2", architecture, config, config["smoke_seed"], overfit=True)
        if not smoke["gradients_finite"] or smoke["final_fixed_batch_loss"] >= smoke["initial_fixed_batch_loss"]:
            raise ValueError("One-batch gradient/loss verification failed")
        if not smoke["reload_exact"] or not smoke["repeated_inference_exact"]:
            raise ValueError("One-batch reload/inference repeat check failed")
        results.append(safe_train(root, "v2", architecture, config, config["smoke_seed"]))
    results.append(rule_validation(root, "v2", config))
    write_json(root / "experiments/milestone3/v2_verification.json", {"verified": all(r["complete"] for r in results), "runs": [r["name"] for r in results], "kind": "synthetic small-pipeline verification; not real-world evidence"})
    return results


def expand_v3(root):
    require_frozen(root)
    if not read_json(root / "experiments/milestone3/v2_verification.json")["verified"]:
        raise ValueError("v2 verification must pass before expansion")
    config = read_json(root / "configs/structural_v3.json")
    output = root / config["output"]
    if not (output / "manifest.json").exists():
        generate_dataset(root, config)
    from .structural_validation import validate_record, draw_overlay
    from .leakage import leakage_report
    records = sum([read_jsonl(output / f"{split}.jsonl") for split in config["splits"]], [])
    invalid = [{"id": r["id"], "errors": validate_record(r, root)} for r in records]
    invalid = [r for r in invalid if r["errors"]]
    manifest = read_json(output / "manifest.json")
    indexed = {r["id"]: r for r in records}
    for record in manifest["records"]:
        path = output / "annotations" / f"{record['id']}.json"
        if sha256(path) != record["annotation_sha256"] or read_json(path) != indexed[record["id"]]:
            invalid.append({"id": record["id"], "errors": ["annotation/manifest/split mismatch"]})
    real = sum([read_jsonl(root / f"data/normalized/{name}.jsonl") for name in ("training", "pilot", "vistext_training")], [])
    leakage = leakage_report(records, root, config["near_duplicate_hamming_threshold"], real)
    prior = sum([read_jsonl(root / f"data/synthetic/v2/{split}.jsonl") for split in config["splits"]], [])
    numeric_v2 = {fingerprint(r["spec"]["data"]["values"]) for r in prior}
    overlap = [r["id"] for r in records if fingerprint(r["spec"]["data"]["values"]) in numeric_v2]
    for split in config["splits"]:
        for kind in config["chart_types"]:
            row = next(r for r in records if r["split"] == split and r["chart_type"] == kind)
            draw_overlay(root / row["image_path"], row["nodes"], root / f"experiments/milestone3/v3_overlays/{row['id']}.png", {"bar", "point", "line_segment", "tick_label"})
    report = {"count": len(records), "splits": dict(Counter(r["split"] for r in records)),
              "chart_types": dict(Counter(r["chart_type"] for r in records)), "invalid_samples": invalid,
              "leakage": leakage, "v2_numerical_overlap": overlap,
              "node_types": dict(Counter(n["type"] for r in records for n in r["nodes"])),
              "relations": dict(Counter(e["relation"] for r in records for e in r["edges"])),
              "v3_split_sha256": {s: sha256(output / f"{s}.jsonl") for s in config["splits"]}}
    write_json(root / "experiments/milestone3/v3_integrity.json", report)
    if invalid or leakage["violations"] or overlap:
        raise ValueError("Expansion integrity/leakage check failed; no v3 fitting permitted")
    return report


def fit_v3(root, config):
    if (root / "experiments/milestone3/sealed_inventory.json").exists():
        raise ValueError("Fitting/calibration is forbidden after final-test inventory sealing")
    report = read_json(root / "experiments/milestone3/v3_integrity.json")
    if report["invalid_samples"] or report["leakage"]["violations"]:
        raise ValueError("v3 integrity failed")
    results = [rule_validation(root, "v3", config)]
    for seed in config["seeds"]:
        for architecture in ("mlp", "gat"):
            results.append(safe_train(root, "v3", architecture, config, seed))
    for variant in config["ablations"]:
        results.append(safe_train(root, "v3", "gat", config, config["ablation_seed"], variant))
    return results


def seal(root):
    require_frozen(root)
    runs = []
    for path in sorted((root / "experiments/milestone3/runs").glob("*/result.json")):
        result = read_json(path)
        if "overfit" in result["name"]:
            continue
        entry = {"name": result["name"], "result_sha256": sha256(path), "complete": result["complete"]}
        if result["variant"]["architecture"] != "rule" and result["complete"]:
            for filename in ("checkpoint.pt", "thresholds.json"):
                entry[filename] = sha256(path.parent / filename)
        runs.append(entry)
    config = read_json(root / "configs/relations.json")
    expected = {"v2_rule", "v3_rule"}
    expected.update(f"v2_{name}_main_s{config['smoke_seed']}" for name in ("mlp", "gat"))
    expected.update(f"v3_{name}_main_s{seed}" for seed in config["seeds"] for name in ("mlp", "gat"))
    expected.update(f"v3_gat_{variant}_s{config['ablation_seed']}" for variant in config["ablations"])
    if expected != {entry["name"] for entry in runs}:
        raise ValueError("Planned experiment inventory is incomplete; document missing runs before test access")
    target = root / "experiments/milestone3/sealed_inventory.json"
    payload = {"protocol_sha256": sha256(root / "experiments/milestone3/protocol_freeze.json"), "runs": runs,
               "test_split_sha256": {v: sha256(root / f"data/synthetic/{v}/test.jsonl") for v in ("v2", "v3")}}
    if target.exists() and read_json(target) != payload:
        raise ValueError("Sealed experiment inventory cannot change")
    write_json(target, payload)
    return payload


def evaluate_test(root, config):
    require_frozen(root)
    sealed = read_json(root / "experiments/milestone3/sealed_inventory.json")
    for version, digest in sealed["test_split_sha256"].items():
        if sha256(root / f"data/synthetic/{version}/test.jsonl") != digest:
            raise ValueError("Sealed test split changed")
    results = []
    for entry in sealed["runs"]:
        if not entry["complete"]:
            continue
        directory = root / "experiments/milestone3/runs" / entry["name"]
        if sha256(directory / "result.json") != entry["result_sha256"]:
            raise ValueError("Selected run changed after seal")
        output = directory / "test.json"
        if output.exists():
            results.append(read_json(output))
            continue
        trained = read_json(directory / "result.json")
        dataset, variant = trained["dataset"], trained["variant"]
        samples = cached_split(root, dataset, "test", allow_test=True)
        architecture = variant["architecture"]
        scope = GPUSession(root, entry["name"] + "_final_test", 90) if architecture != "rule" else __import__("contextlib").nullcontext()
        with scope:
            device = "cpu" if architecture == "rule" else "cuda"
            if architecture == "rule":
                model, thresholds = None, trained["thresholds"]
            else:
                for filename in ("checkpoint.pt", "thresholds.json"):
                    if sha256(directory / filename) != entry[filename]:
                        raise ValueError("Selected checkpoint/threshold artifact changed")
                model, _ = load_checkpoint(directory / "checkpoint.pt", device)
                thresholds = read_json(directory / "thresholds.json")["values"]
                torch.cuda.reset_peak_memory_stats()
            y, p, chart_sizes, latencies = infer(model, samples, variant, config, device, measured=True)
            start = 0
            charts, failures = [], []
            for identity, count in chart_sizes:
                labels, scores = y[start:start + count], p[start:start + count]
                charts.append({"id": identity, "counts": confusion(labels, scores, thresholds).tolist()})
                predicted = scores >= np.asarray(thresholds)
                for label, relation in enumerate(RELATIONS):
                    for kind, selection in (("false_positive", predicted[:, label] & ~labels[:, label].astype(bool)), ("false_negative", ~predicted[:, label] & labels[:, label].astype(bool))):
                        if sum(f["relation"] == relation and f["kind"] == kind for f in failures) >= 5:
                            continue
                        for index in np.flatnonzero(selection)[:2]:
                            failures.append({"chart": identity, "candidate_ordinal": int(index), "relation": relation, "kind": kind, "score": float(scores[index, label])})
                start += count
            result = {"name": entry["name"], "dataset": dataset, "device": device, "metrics": metrics(y, p, thresholds),
                      "charts": charts, "errors": failures, "latency_median_seconds": float(np.median(latencies)), "latency_p95_seconds": float(np.quantile(latencies, .95)),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
                      "sealed_inventory_sha256": sha256(root / "experiments/milestone3/sealed_inventory.json")}
            np.savez_compressed(directory / "test_predictions.npz", scores=p, labels=y, thresholds=np.array(thresholds), chart_counts=np.array([c for _, c in chart_sizes]))
            write_json(output, result)
            results.append(result)
            if model is not None:
                del model
                torch.cuda.empty_cache()
    by_name = {r["name"]: r for r in results}
    pairs = []
    for seed in config["seeds"]:
        a, b = f"v3_gat_main_s{seed}", f"v3_mlp_main_s{seed}"
        if a in by_name and b in by_name:
            if [c["id"] for c in by_name[a]["charts"]] != [c["id"] for c in by_name[b]["charts"]]:
                raise ValueError("Paired test chart identities differ")
            pairs.append((by_name[a], by_name[b]))
    interval = paired_bootstrap([[c["counts"] for c in a["charts"]] for a, _ in pairs], [[c["counts"] for c in b["charts"]] for _, b in pairs], config["bootstrap_samples"], config["bootstrap_seed"]) if pairs else None
    write_json(root / "experiments/milestone3/test_comparison.json", {"runs": [r["name"] for r in results], "paired_seeds": len(pairs), "paired_chart_bootstrap": interval})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify-v2", "expand", "fit-v3", "seal", "test", "check-history"))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = read_json(root / "configs/relations.json")
    seed_everything(config["smoke_seed"])
    try:
        commands = {"freeze": lambda: freeze(root, config), "verify-v2": lambda: verify_v2(root, config),
                    "expand": lambda: expand_v3(root), "fit-v3": lambda: fit_v3(root, config),
                    "seal": lambda: seal(root), "test": lambda: evaluate_test(root, config), "check-history": lambda: history_check(root)}
        result = commands[args.command]()
        print(json.dumps({"command": args.command, "status": "complete", "result": result if args.command in ("freeze", "seal", "check-history", "expand") else [r["name"] for r in result]}, indent=2))
    except Exception as error:
        write_json(root / f"experiments/milestone3/{args.command}_failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
