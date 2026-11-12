"""Bounded relation training, independent input caches, and validation-only selection."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import contextlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from .graph_data import StructuralDataset
from .io import read_json, read_jsonl, sha256, write_json
from .relation_models import RelationNet, pack_samples, rule_scores
from .relation_metrics import calibrate, metrics, confusion
from .synthetic import fingerprint, RELATIONS


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


class GPUSession:
    def __init__(self, root, name, seconds=240):
        self.path = Path(root) / "experiments/milestone3/gpu_budget.json"
        historical = read_json(Path(root) / "experiments/gpu_budget.json")["charged_seconds"]
        self.ledger = read_json(self.path) if self.path.exists() else {"historical_seconds": historical, "limit_seconds": 3600, "charged_seconds": 0, "sessions": []}
        available = self.ledger["limit_seconds"] - self.ledger["historical_seconds"] - self.ledger["charged_seconds"]
        self.seconds = min(seconds, available)
        if self.seconds < 20:
            raise RuntimeError("Lifetime GPU budget exhausted")
        self.entry = {"name": name, "reserved_seconds": self.seconds, "actual_seconds": None}

    def __enter__(self):
        self.started = time.monotonic()
        self.ledger["charged_seconds"] += self.seconds
        self.ledger["sessions"].append(self.entry)
        write_json(self.path, self.ledger)
        return self

    @property
    def remaining(self):
        return self.seconds - (time.monotonic() - self.started)

    def __exit__(self, *args):
        elapsed = time.monotonic() - self.started
        self.entry["actual_seconds"] = elapsed
        self.ledger["charged_seconds"] += elapsed - self.seconds
        write_json(self.path, self.ledger)


def cached_split(root, dataset, split, allow_test=False):
    root = Path(root)
    if split == "test" and not allow_test:
        raise ValueError("Test tensors are unavailable before sealed final evaluation")
    manifest = root / f"data/synthetic/{dataset}/{split}.jsonl"
    path = root / f"experiments/milestone3/cache/{dataset}_{split}.pt"
    digest = sha256(manifest)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["manifest_sha256"] != digest:
            raise ValueError("Cached source split changed")
        return saved["samples"]
    torch.set_num_threads(1)
    data = StructuralDataset(manifest, root)
    samples = []
    for i in range(len(data)):
        item = data[i]
        graph, identity = item["graph"], item["metadata"]["id"]
        generator = torch.Generator().manual_seed(int(fingerprint(["node-order", dataset, identity])[:15], 16))
        perm = torch.randperm(graph.num_nodes, generator=generator)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(len(perm))
        samples.append({"inputs": {"x": graph.x[perm].contiguous(), "candidate_index": inverse[graph.edge_index]},
                        "targets": {"edge_y": graph.edge_y.clone()},
                        "metadata": {"id": identity, "split": split, "node_count": graph.num_nodes,
                                     "proposal_source": "renderer ground truth; oracle structural prediction only"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest_sha256": digest, "samples": samples}, path)
    write_json(path.with_suffix(".json"), {"manifest_sha256": digest, "cache_sha256": sha256(path), "count": len(samples),
                                         "inputs": ["x", "candidate_index"], "targets": ["edge_y"], "dtype": "float32"})
    return samples


def run_variant(architecture, variant="main"):
    return {"architecture": architecture, "mode": variant if variant in ("no_geometry", "no_color") else "all",
            "connectivity": "complete" if variant in ("no_geometry", "complete_connectivity") else "spatial", "variant": variant}


def loss_weights(samples, cap=50.):
    positives = torch.stack([s["targets"]["edge_y"].sum(0) for s in samples]).sum(0)
    total = sum(s["targets"]["edge_y"].shape[0] for s in samples)
    return torch.where(positives > 0, ((total - positives) / positives.clamp_min(1)).clamp(1, cap), torch.ones_like(positives))


def pack(samples, device, variant, config):
    return pack_samples(samples, device, variant["mode"], variant["connectivity"], config["spatial_neighbors"])


def infer(model, samples, variant, config, device, measured=False):
    scores, labels, counts, latency = [], [], [], []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), 1 if measured or model is None else config["batch_size"]):
            selected = samples[start:start + (1 if measured or model is None else config["batch_size"])]
            if device == "cuda":
                torch.cuda.synchronize()
            before = time.perf_counter()
            inputs, targets, slices = pack(selected, device, variant, config)
            probability = rule_scores(inputs) if model is None else torch.sigmoid(model(inputs))
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            p, y = probability.cpu().numpy(), targets.cpu().numpy()
            scores.append(p)
            labels.append(y)
            counts.extend((samples[start + j]["metadata"]["id"], end - begin) for j, (begin, end) in enumerate(slices))
            latency.extend([elapsed / len(selected)] * len(selected))
    return np.concatenate(labels), np.concatenate(scores), counts, latency


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = RelationNet(saved["variant"]["architecture"], saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved


def train_run(root, dataset, architecture, config, seed=42, variant_name="main", overfit=False):
    root = Path(root)
    name = f"{dataset}_{architecture}_{variant_name}_s{seed}" + ("_overfit" if overfit else "")
    directory = root / "experiments/milestone3/runs" / name
    if (directory / "result.json").exists():
        raise ValueError("Run result already exists; choose a new protocol/run version")
    directory.mkdir(parents=True, exist_ok=True)
    train = cached_split(root, dataset, "train")
    validation = cached_split(root, dataset, "val") if not overfit else train[:config["batch_size"]]
    selected_config = dict(config)
    if overfit:
        selected_config.update(dropout=0., lr=config["overfit_lr"])
    steps = config["overfit_steps"] if overfit else config[f"{dataset}_steps"]
    variant = run_variant(architecture, variant_name)
    weights = loss_weights(train, config["positive_weight_cap"])
    result = {"name": name, "dataset": dataset, "seed": seed, "variant": variant, "config": selected_config,
              "requested_steps": steps, "completed_steps": 0, "complete": False, "kind": "one_batch_wiring_verification" if overfit else "synthetic_oracle_proposal_training",
              "manifest_hashes": {split: sha256(root / f"data/synthetic/{dataset}/{split}.jsonl") for split in ("train", "val")},
              "train_positive_weights": weights.tolist(), "deterministic_algorithms": "enabled, warn_only=True; unsupported CUDA kernels may remain nondeterministic"}
    write_json(directory / "result.json", result)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU experiment not silently replaced")
    with GPUSession(root, name, config["gpu_session_seconds"]) as budget:
        started = time.perf_counter()
        seed_everything(seed)
        torch.cuda.reset_peak_memory_stats()
        model = RelationNet(architecture, selected_config).cuda()
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=selected_config["lr"], weight_decay=config["weight_decay"])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.cuda())
        rng = np.random.default_rng(seed)
        best_ap, best_step = -1., None
        logs = []
        first = train[:config["batch_size"]]
        fixed_inputs, fixed_targets, _ = pack(first, "cuda", variant, selected_config)
        model.eval()
        with torch.no_grad():
            result["initial_fixed_batch_loss"] = float(criterion(model(fixed_inputs), fixed_targets))
        for step in range(1, steps + 1):
            if budget.remaining < 20:
                result["stop_reason"] = "GPU session time budget"
                break
            selected = first if overfit else [train[int(i)] for i in rng.choice(len(train), min(config["batch_size"], len(train)), replace=False)]
            inputs, targets, _ = pack(selected, "cuda", variant, selected_config)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if logits.shape != targets.shape:
                raise ValueError("Relation logits/targets shape mismatch")
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            result["completed_steps"] = step
            logs.append({"step": step, "train_loss": float(loss.detach()), "gradient_norm": float(norm), "elapsed_seconds": time.perf_counter() - started})
            if step % config["validation_interval"] == 0 or step == steps:
                y, p, _, _ = infer(model, validation, variant, selected_config, "cuda")
                validation_metrics = metrics(y, p, [0.5] * len(RELATIONS))
                ap = validation_metrics["macro_ap"] or 0.
                logs[-1].update(validation_macro_ap=ap, validation_macro_f1_at_05=validation_metrics["macro_f1"])
                if ap > best_ap:
                    best_ap, best_step = ap, step
                    torch.save({"state_dict": model.state_dict(), "config": selected_config, "variant": variant,
                                "step": step, "seed": seed, "dataset": dataset}, directory / "checkpoint.pt")
                print(f"{name}: step {step}/{steps}, loss={float(loss):.4f}, val AP={ap:.4f}", flush=True)
        if best_step is None:
            raise RuntimeError("No validation-selected checkpoint produced before budget limit")
        selected_model, checkpoint = load_checkpoint(directory / "checkpoint.pt", "cuda")
        y, p, _, _ = infer(selected_model, validation, variant, selected_config, "cuda")
        _, repeated, _, _ = infer(selected_model, validation[:1], variant, selected_config, "cuda")
        count = validation[0]["targets"]["edge_y"].shape[0]
        # Re-evaluate the first chart with identical batching for a meaningful exact check.
        _, single, _, _ = infer(selected_model, validation[:1], variant, selected_config, "cuda")
        with torch.inference_mode():
            original = selected_model(fixed_inputs).cpu()
        restored, _ = load_checkpoint(directory / "checkpoint.pt", "cuda")
        with torch.inference_mode():
            reloaded = restored(fixed_inputs).cpu()
        thresholds = calibrate(y, p, config["threshold_grid"])
        write_json(directory / "thresholds.json", {"values": thresholds, "source_split": "val" if not overfit else "one_batch_verification_only",
                                                   "checkpoint_sha256": sha256(directory / "checkpoint.pt")})
        result.update(complete=result["completed_steps"] == steps, best_step=best_step, validation=metrics(y, p, thresholds),
                      checkpoint_sha256=sha256(directory / "checkpoint.pt"), checkpoint_bytes=(directory / "checkpoint.pt").stat().st_size,
                      reload_exact=bool(torch.equal(original, reloaded)), repeated_inference_exact=bool(np.array_equal(single, repeated)),
                      reload_max_abs_difference=float((original - reloaded).abs().max()), repeated_max_abs_difference=float(np.max(np.abs(single - repeated))),
                      final_fixed_batch_loss=float(criterion(selected_model(fixed_inputs), fixed_targets).detach()),
                      elapsed_seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gradients_finite=True, train_log=logs)
        write_json(directory / "result.json", result)
        del model, selected_model, restored, optimizer
        torch.cuda.empty_cache()
    return result


def rule_validation(root, dataset, config):
    samples = cached_split(root, dataset, "val")
    variant = run_variant("rule")
    started = time.perf_counter()
    y, p, _, latency = infer(None, samples, variant, config, "cpu")
    thresholds = calibrate(y, p, config["threshold_grid"])
    result = {"name": f"{dataset}_rule", "dataset": dataset, "variant": variant, "parameter_count": 0,
              "thresholds": thresholds, "validation": metrics(y, p, thresholds), "elapsed_seconds": time.perf_counter() - started,
              "complete": True, "device": "cpu", "training": "none; fixed deterministic rules"}
    write_json(Path(root) / f"experiments/milestone3/runs/{dataset}_rule/result.json", result)
    return result
