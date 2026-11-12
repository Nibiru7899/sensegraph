from types import SimpleNamespace

from PIL import Image

from sensegraph.io import read_jsonl, sha256, write_json, write_jsonl
from sensegraph.runner import run_baseline


def test_cached_extraction_keeps_own_generation_metadata(tmp_path, monkeypatch):
    records = []
    for i, color in enumerate(("red", "blue", "red")):
        image = tmp_path / f"{i}.png"
        Image.new("RGB", (10, 10), color).save(image)
        records.append({"id": str(i), "image_path": image.name, "image_sha256": sha256(image),
                        "question": "What is the maximum value?"})
    write_jsonl(tmp_path / "data/normalized/pilot.jsonl", records)
    write_json(tmp_path / "data/manifests/pilot.json", {"fixture": "wiring_only"})
    write_json(tmp_path / "experiments/deplot/smoke.json", {"complete": True})
    write_jsonl(tmp_path / "experiments/deplot/smoke_predictions.jsonl", [{"raw_output": "fixture"}])

    class FakeModel:
        def __init__(self, root, name, config):
            cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                                   max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
            self.torch = SimpleNamespace(cuda=cuda)
            self.dtype, self.precision_fallback = "fixture", None
            self.font_path = tmp_path / "0.png"
            self.font_sha256 = sha256(self.font_path)

        def predict(self, path, question, max_seconds):
            self.last_generation = {"completed": path.name != "1.png"}
            return "Category | Value\nAlpha | 20"

    monkeypatch.setattr("sensegraph.runner.LocalModel", FakeModel)
    config = {"models": {"deplot": {"revision": "fixture"}}, "model_seconds": 1500, "total_gpu_seconds": 3600}
    run_baseline(tmp_path, config, "deplot")
    predictions = read_jsonl(tmp_path / "experiments/deplot/predictions.jsonl")
    assert [p["status"] for p in predictions] == ["ok", "error", "ok"]
    assert predictions[2]["cached_extraction"] is True
    assert predictions[2]["generation"]["completed"] is True
