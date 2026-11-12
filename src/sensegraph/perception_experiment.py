"""Budgeted, protocol-frozen perception feasibility experiment. No test-set reader."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .acquisition import Downloader
from .io import read_json, read_jsonl, write_json, sha256
from .perception import detector, detection_target, image_tensor, predicted_graph, visual_embeddings, detection_metrics, ocr, CLASSES
from .relation_models import RelationNet
from .relation_training import seed_everything


def sources(root):
    names = ["configs/perception.json", "research/milestone4_protocol.md", "src/sensegraph/perception.py", "src/sensegraph/perception_experiment.py"]
    return {name: sha256(root/name) for name in names}


def verify(root):
    frozen = read_json(root/"experiments/milestone4/freeze.json")
    if frozen["sources"] != sources(root):
        raise ValueError("Perception protocol/source changed after freeze")
    changed = [p for p, digest in frozen["history"].items() if not (root/p).exists() or sha256(root/p) != digest]
    if changed:
        raise ValueError(f"Historical artifacts changed: {changed[:5]}")
    return {"protected_files": len(frozen["history"]), "changed": []}


def freeze(root, config):
    directory = root/"experiments/milestone4"
    if (directory/"freeze.json").exists():
        return verify(root)
    files = []
    for folder in ("data", "experiments", "src", "tests", "configs", "research"):
        files.extend(p for p in (root/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and "milestone4" not in p.parts and p.name not in ("download_ledger.json", "perception.py", "perception_experiment.py", "test_perception.py", "perception.json", "milestone4_protocol.md"))
    legacy = read_json(root/"experiments/gpu_budget.json")["charged_seconds"]
    m3 = read_json(root/"experiments/milestone3/gpu_budget.json")["charged_seconds"]
    history = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    train = read_jsonl(root/"data/synthetic/v3/train.jsonl")[:config["train_count"]]
    val = read_jsonl(root/"data/synthetic/v3/val.jsonl")[:config["val_count"]]
    write_json(directory/"freeze.json", {"sources": sources(root), "history": history, "config": config,
                "train_ids": [r["id"] for r in train], "val_ids": [r["id"] for r in val],
                "historical_gpu_seconds": legacy+m3, "download_snapshot": read_json(root/"experiments/download_ledger.json"),
                "test_access": False, "torch": torch.__version__, "cuda": torch.version.cuda})
    return {"protected_files": len(history), "historical_gpu_seconds": legacy+m3}


class Session:
    def __init__(self, root, seconds):
        self.path = root/"experiments/milestone4/gpu_budget.json"
        historical = read_json(root/"experiments/milestone4/freeze.json")["historical_gpu_seconds"]
        self.ledger = read_json(self.path) if self.path.exists() else {"historical_seconds": historical, "charged_seconds": 0., "limit_seconds": 3600, "sessions": []}
        self.seconds = min(seconds, 3600-historical-self.ledger["charged_seconds"])
        if self.seconds < seconds:
            raise RuntimeError("Insufficient budget for the predeclared bounded session")

    def __enter__(self):
        self.start = time.monotonic()
        self.entry = {"reserved_seconds": self.seconds, "actual_seconds": None}
        self.ledger["sessions"].append(self.entry); self.ledger["charged_seconds"] += self.seconds
        write_json(self.path, self.ledger)
        return self

    @property
    def remaining(self):
        return self.seconds - (time.monotonic()-self.start)

    def __exit__(self, *args):
        actual = time.monotonic()-self.start
        self.entry["actual_seconds"] = actual
        self.ledger["charged_seconds"] += actual-self.seconds
        write_json(self.path, self.ledger)


def train_step(model, image, target, optimizer):
    model.train(); model.backbone.eval(); model.roi_heads.box_head.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = model([image], [target]); loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite detection loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite detection gradient")
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm": float(norm), **{k: float(v.detach()) for k,v in losses.items()}}


def overlay(root, record, output, path, threshold):
    image = Image.open(root/record["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for b, s, label in zip(output["boxes"], output["scores"], output["labels"]):
        if float(s) >= threshold:
            box = b.tolist(); draw.rectangle(box, outline="red", width=2)
            draw.text((box[0],box[1]), f"{CLASSES[int(label)]} {float(s):.2f}", fill="black")
    image.save(path)


def run(root, config):
    integrity = verify(root)
    directory = root/"experiments/milestone4"
    if (directory/"attempt.json").exists():
        raise ValueError("Attempt exists; preserve it and create a new protocol for any refit")
    write_json(directory/"attempt.json", {"started": time.time(), "complete": False})
    seed_everything(config["seed"])
    train = read_jsonl(root/"data/synthetic/v3/train.jsonl")[:config["train_count"]]
    val = read_jsonl(root/"data/synthetic/v3/val.jsonl")[:config["val_count"]]
    device = "cuda"
    with Session(root, config["session_seconds"]) as session:
        torch.cuda.reset_peak_memory_stats()
        model = detector(root, config).to(device)
        initial = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        parameters = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        image = image_tensor(root/train[0]["image_path"]).to(device)
        target = {k:v.to(device) for k,v in detection_target(train[0])[0].items()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"])
        model.eval()
        with torch.inference_mode():
            first = model([image])[0]; repeated = model([image])[0]
        exact = all(torch.equal(first[k],repeated[k]) for k in first)
        overlay(root, train[0], {k:v.cpu() for k,v in first.items()}, directory/"smoke_before.png",config["score_threshold"])
        # Pause here for visual review before full fitting.
        smoke = [train_step(model,image,target,optimizer) for _ in range(config["one_batch_steps"])]
        smoke_ok = exact and np.mean([r["loss"] for r in smoke[-5:]]) < np.mean([r["loss"] for r in smoke[:5]])
        model.eval()
        with torch.inference_mode():
            before_reload = model([image])[0]
        torch.save({"state_dict":model.state_dict()}, directory/"smoke.pt")
        model.load_state_dict(torch.load(directory/"smoke.pt",map_location=device,weights_only=True)["state_dict"])
        model.eval()
        with torch.inference_mode():
            after = {k:v.cpu() for k,v in model([image])[0].items()}
        reload_exact = all(torch.equal(before_reload[k].cpu(), after[k]) for k in after)
        smoke_ok = smoke_ok and reload_exact
        write_json(directory/"smoke.json", {"repeated_inference_exact":exact,"checkpoint_reload_exact":reload_exact,
                   "loss_decreased":bool(smoke_ok),"log":smoke})
        overlay(root,train[0],after,directory/"smoke_after.png",config["score_threshold"])
        if not smoke_ok:
            raise RuntimeError("Smoke checks failed; no development fitting")
        write_json(directory/"smoke_ready.json", {"status":"awaiting_visual_review", "remaining_seconds":session.remaining})
        # Full development fitting is a separate command after the smoke overlay review.
        torch.save({"state_dict":initial}, directory/"initial.pt")
        write_json(directory/"model_size.json", {"parameters":parameters,"trainable_parameters":trainable,"device":torch.cuda.get_device_name()})
    return {"smoke":"passed", "next":"Review smoke_after.png; then run fit"}


def fit(root, config):
    integrity = verify(root)
    directory = root/"experiments/milestone4"
    if not (directory/"visual_approval.json").exists():
        raise ValueError("Smoke overlay review must be recorded first")
    if (directory/"fit_attempt.json").exists():
        raise ValueError("Development fitting already attempted")
    write_json(directory/"fit_attempt.json", {"started":time.time()})
    seed_everything(config["seed"])
    train = read_jsonl(root/"data/synthetic/v3/train.jsonl")[:config["train_count"]]
    val = read_jsonl(root/"data/synthetic/v3/val.jsonl")[:config["val_count"]]
    with Session(root,config["session_seconds"]) as session:
        torch.cuda.reset_peak_memory_stats()
        model = detector(root,config).cuda()
        model.load_state_dict(torch.load(directory/"initial.pt",weights_only=True,map_location="cuda")["state_dict"])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config["lr"])
        log = []
        rng = np.random.default_rng(config["seed"])
        indices = []
        while len(indices) < config["steps"]:
            indices.extend(rng.permutation(len(train)).tolist())
        for step,index in enumerate(indices[:config["steps"]],1):
            if session.remaining < 60:
                break
            record = train[index]
            image = image_tensor(root/record["image_path"]).cuda()
            target = {k:v.cuda() for k,v in detection_target(record)[0].items()}
            log.append({"step":step,**train_step(model,image,target,optimizer)})
            if step % 40 == 0:
                write_json(directory/"training_log.json",log)
                print(f"step {step}, loss {log[-1]['loss']:.4f}, remaining {session.remaining:.1f}s",flush=True)
        torch.save({"state_dict":model.state_dict(),"config":config,"steps":len(log)},directory/"detector.pt")
        write_json(directory/"training_log.json",log)
        model.eval()
        predictions, targets, timings, rejected, graph_checks = [], [], [], [], []
        config3 = read_json(root/"configs/relations.json")
        nets = {name:RelationNet(name,config3).eval() for name in ("mlp","gat")}
        with torch.inference_mode():
            for index,record in enumerate(val):
                if session.remaining < 8:
                    break
                started=time.perf_counter()
                image=image_tensor(root/record["image_path"])
                output={k:v.cpu() for k,v in model([image.cuda()])[0].items()}
                boxes=output["boxes"][output["scores"]>=config["score_threshold"]]
                embeddings=visual_embeddings(model,image.cuda(),boxes.cuda()).cpu()
                graph=predicted_graph(image,boxes)
                shape_checks={name:list(net(graph).shape) for name,net in nets.items()}
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-started)
                graph_checks.append({"id":record["id"],"nodes":len(boxes),"embedding_shape":list(embeddings.shape),"untrained_wiring_output_shapes":shape_checks})
                target,_,invalid=detection_target(record)
                rejected.extend({"image":record["id"],**r} for r in invalid)
                predictions.append(output); targets.append(target)
                if index<6:
                    overlay(root,record,output,directory/f"validation_{index:02}.png",config["score_threshold"])
        # Reload the actual final checkpoint and compare, without retraining.
        image=image_tensor(root/val[0]["image_path"]).cuda()
        a=model([image])[0]
        model.load_state_dict(torch.load(directory/"detector.pt",map_location="cuda",weights_only=True)["state_dict"])
        model.eval()
        b=model([image])[0]
        reload_exact=all(torch.equal(a[k],b[k]) for k in a)
        measured={str(t):detection_metrics(predictions,targets,t,config["score_threshold"]) for t in config["iou_thresholds"]}
        metric=measured["0.5"]
        passed=(len(log)==config["steps"] and len(predictions)==len(val) and reload_exact
                and metric["macro_recall"]>=config["gate_macro_recall50"] and metric["macro_ap"]>=config["gate_macro_ap50"]
                and all(metric["per_class"][c]["recall"]>=config["gate_point_segment_recall50"] for c in ("point","line_segment")))
        torch.save({"predictions":predictions,"targets":targets,"graph_checks":graph_checks},directory/"validation.pt")
        result={"gate_passed":passed,"steps":len(log),"validation_images":len(predictions),"metrics":measured,
                "checkpoint_reload_exact":reload_exact,"rejected_supervision":rejected,"graph_checks":graph_checks,
                "untrained_relation_wiring_only":True,"latency_seconds":timings,
                "median_pipeline_seconds":float(np.median(timings)),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"history_integrity":integrity,
                "ocr_probe":ocr(Image.open(root/val[0]["image_path"])),"test_evaluated":False}
        write_json(directory/"result.json",result)
    return {"gate_passed":passed,"macro_recall50":metric["macro_recall"],"macro_ap50":metric["macro_ap"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("freeze","acquire","smoke","fit","verify"))
    parser.add_argument("--root",default=".")
    args=parser.parse_args(); root=Path(args.root).resolve()
    config=read_json(root/"configs/perception.json")
    if args.command=="freeze":
        result=freeze(root,config)
    elif args.command=="acquire":
        verify(root)
        path=Downloader(root).fetch(config["weights_url"],config["weights_path"])
        if not sha256(path).startswith("907ea3f9"):
            raise ValueError("Official hash prefix failed")
        write_json(root/"experiments/milestone4/weights.json",{"url":config["weights_url"],"sha256":sha256(path),"bytes":path.stat().st_size})
        result={"path":str(path)}
    elif args.command=="verify":
        result=verify(root)
    else:
        result=run(root,config) if args.command=="smoke" else fit(root,config)
    print(result,flush=True)


if __name__=="__main__":
    main()
