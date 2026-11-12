import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from .io import read_json, write_json


def environment(root):
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": {}}
    for name in ("torch", "torchvision", "transformers", "numpy", "Pillow", "sentencepiece", "pytest"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
        if torch.cuda.is_available():
            tensor = torch.ones(8, device="cuda")
            result["cuda_tensor_check"] = tensor.sum().item() == 8
    except ImportError:
        result["cuda_available"] = False
    result["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True).strip()
    write_json(Path(root) / "experiments/environment.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(prog="sensegraph")
    parser.add_argument("command", choices=["environment", "acquire", "normalize", "inspect", "infer", "score", "oracle"])
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", choices=["chartqa", "vistext"], default="chartqa")
    parser.add_argument("--model", choices=["unichart", "deplot"], default="unichart")
    parser.add_argument("--models", action="store_true", help="Acquire selected model instead of data")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = read_json(root / args.config)
    try:
        if args.command == "environment":
            result = environment(root)
        elif args.command == "acquire":
            if args.models:
                from .models import acquire_model
                result = acquire_model(root, config, args.model)
            else:
                from .data import acquire_chartqa, acquire_vistext
                result = (acquire_chartqa if args.dataset == "chartqa" else acquire_vistext)(root, config)
        elif args.command == "normalize":
            from .data import normalize, normalize_vistext
            result = (normalize if args.dataset == "chartqa" else normalize_vistext)(root, config)
        elif args.command == "inspect":
            from .data import inspect
            result = inspect(root, config)
        elif args.command == "infer":
            from .runner import run_baseline
            result = run_baseline(root, config, args.model, args.smoke)
            if args.smoke:
                from .io import read_jsonl
                checks = read_jsonl(root / f"experiments/{args.model}/smoke_predictions.jsonl")
                if not checks or checks[0].get("error") or not checks[0].get("raw_output"):
                    raise RuntimeError("Real-image smoke inference failed; inspect saved smoke artifacts")
        elif args.command == "score":
            from .evaluation import score_run
            result = score_run(root, args.model)
        else:
            from .io import read_jsonl, write_jsonl
            from .reasoning import execute
            from .evaluation import score_records
            rows = read_jsonl(root / "data/normalized/pilot.jsonl")
            predictions = [{"id": row["id"], **execute(row["question"], row["table"])} if row["table"] else {"id": row["id"], "status": "missing_reference_table"} for row in rows]
            write_jsonl(root / "experiments/oracle/predictions.jsonl", predictions)
            result = {"kind": "oracle_reference_table_diagnostic_not_model_performance", **score_records(rows, predictions)}
            write_json(root / "experiments/oracle/metrics.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
