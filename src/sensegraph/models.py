"""Pinned, local-only model adapters; no reference answers/tables enter this interface."""
import os
import time
from pathlib import Path

from .acquisition import Downloader
from .io import read_json, sha256, write_json


def acquire_model(root, config, name):
    downloader = Downloader(root, config["download_limit_bytes"])
    model = config["models"][name]
    metadata = downloader.json(f"https://huggingface.co/api/models/{model['id']}/revision/{model['revision']}?blobs=true", f"model-{name}")
    if metadata["sha"] != model["revision"] or metadata.get("gated"):
        raise ValueError("Unexpected model revision or gated checkpoint")
    license_name = metadata.get("cardData", {}).get("license")
    if license_name != model["license"]:
        raise ValueError(f"Model license changed: {license_name}")
    siblings = metadata["siblings"]
    safe = any(item["rfilename"] == "model.safetensors" for item in siblings)
    manifest = {**model, "files": []}
    for item in siblings:
        filename = item["rfilename"]
        if "/" in filename:
            continue
        wanted = filename.endswith((".json", ".model")) or filename == "README.md"
        wanted |= filename == ("model.safetensors" if safe else "pytorch_model.bin")
        if not wanted:
            continue
        url = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{filename}"
        digest = item.get("lfs", {}).get("sha256")
        path = downloader.fetch(url, f"data/models/{name}/{filename}", digest, item.get("size"))
        manifest["files"].append(downloader.ledger["files"][path.relative_to(Path(root)).as_posix()])
    write_json(Path(root) / f"data/models/{name}/source_manifest.json", manifest)
    return {"model": name, "revision": model["revision"], "files": len(manifest["files"])}


class LocalModel:
    def __init__(self, root, name, config):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoProcessor, Pix2StructForConditionalGeneration

        self.torch, self.name, self.config = torch, name, config
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; real GPU pilot cannot run")
        self.device, self.dtype = "cuda", torch.float16
        path = Path(root) / "data/models" / name
        manifest = read_json(path / "source_manifest.json")
        if manifest["revision"] != config["models"][name]["revision"]:
            raise ValueError("Model files differ from configured revision")
        for artifact in manifest["files"]:
            filename = artifact["url"].rsplit("/", 1)[1]
            if sha256(path / filename) != artifact["sha256"]:
                raise ValueError(f"Checkpoint integrity failed: {filename}")
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        if name == "unichart":
            self.processor = DonutProcessor.from_pretrained(path, local_files_only=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(path, local_files_only=True, torch_dtype=self.dtype, weights_only=True)
        else:
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self.model = Pix2StructForConditionalGeneration.from_pretrained(path, local_files_only=True, torch_dtype=self.dtype, weights_only=True)
            # Pix2Struct otherwise downloads Arial from the Hub while rendering its prompt.
            self.font_path = Path(config.get("deplot_font_path", Path(os.environ.get("WINDIR", "")) / "Fonts/arial.ttf"))
            if not self.font_path.is_file():
                raise ValueError("Set deplot_font_path to a locally installed TrueType font")
            self.font_sha256 = sha256(self.font_path)
        self.model.eval().to(self.device)
        self.precision_fallback = None
        self.last_generation = None

    def predict(self, image_path, question, max_seconds=60):
        from PIL import Image
        torch = self.torch
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if self.name == "unichart":
            prompt = f"<chartqa> {question} <s_answer>"
            inputs = {"pixel_values": self.processor(image, return_tensors="pt").pixel_values,
                      "decoder_input_ids": self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids}
        else:
            # The composite 4.57 processor drops font_path; call its image preprocessor
            # directly. VQA Pix2Struct renders the prompt into the image, not tokens.
            inputs = self.processor.image_processor.preprocess(images=image, header_text="Generate underlying data table of the figure below:", return_tensors="pt", font_path=str(self.font_path))
        inputs = {key: value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device) for key, value in inputs.items()}
        kwargs = {"max_new_tokens": self.config[f"{self.name}_max_new_tokens"], "num_beams": self.config["num_beams"],
                  "do_sample": False, "use_cache": True, "max_time": max_seconds}
        if self.name == "unichart":
            kwargs.update(pad_token_id=self.processor.tokenizer.pad_token_id, eos_token_id=self.processor.tokenizer.eos_token_id,
                          bad_words_ids=[[self.processor.tokenizer.unk_token_id]])
        started = time.monotonic()
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
        except RuntimeError as error:
            if self.dtype != torch.float16 or not any(word in str(error).lower() for word in ("nan", "inf", "probability tensor")):
                raise
            self.dtype = torch.float32
            self.precision_fallback = str(error)
            self.model.float()
            inputs = {key: value.float() if value.is_floating_point() else value for key, value in inputs.items()}
            kwargs["max_time"] = max(1, max_seconds - (time.monotonic() - started))
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
        prefix_length = inputs["decoder_input_ids"].shape[-1] if self.name == "unichart" else 1
        tokens = output[0, prefix_length:].tolist()
        self.last_generation = {"generated_tokens": len(tokens), "completed": self.processor.tokenizer.eos_token_id in tokens,
                                "max_new_tokens": kwargs["max_new_tokens"], "max_seconds": max_seconds}
        text = self.processor.batch_decode(output, skip_special_tokens=self.name != "unichart")[0]
        if self.name == "unichart":
            if "<s_answer>" not in text:
                raise ValueError("UniChart output has no answer delimiter")
            text = text.split("<s_answer>", 1)[1]
            for token in (self.processor.tokenizer.eos_token, self.processor.tokenizer.pad_token):
                text = text.replace(token, "")
        return text.strip()
