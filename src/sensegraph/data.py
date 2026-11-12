"""Official source adapters. Raw assets are immutable; normalized records are versioned."""
import csv
import hashlib
from pathlib import Path
import urllib.error
import urllib.parse
import zipfile

from PIL import Image

from .acquisition import Downloader, RemoteZipReader
from .io import read_json, read_jsonl, sha256, write_json, write_jsonl


def stable_selection(rows, count, seed):
    return sorted(rows, key=lambda x: hashlib.sha256(f"{seed}:{x['id']}".encode()).hexdigest())[:count]


def question_records(raw, split, partition):
    records = []
    for index, item in enumerate(raw):
        if not all(k in item for k in ("imgname", "query", "label")):
            raise ValueError(f"Malformed ChartQA question {split}/{partition}/{index}")
        image = item["imgname"]
        if Path(image).name != image or "\\" in image:
            raise ValueError("Unsafe image filename")
        records.append({"id": f"chartqa:{split}:{partition}:{index}", "image_name": image,
                        "question": item["query"], "answers": [str(item["label"])],
                        "split": split, "partition": partition})
    return records


def acquire_chartqa(root, config):
    root = Path(root)
    downloader = Downloader(root, config["download_limit_bytes"])
    revision = config["chartqa_revision"]
    base = f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{revision}"
    downloader.fetch(f"{base}/LICENSE", "data/raw/chartqa/LICENSE")
    downloader.fetch(f"{base}/README.md", "data/raw/chartqa/README.md")
    selections = []
    train_questions = []
    for split in ("train", "val"):
        split_rows = []
        for partition in ("human", "augmented"):
            path = f"ChartQA Dataset/{split}/{split}_{partition}.json"
            file = downloader.fetch(f"{base}/{urllib.parse.quote(path)}", f"data/raw/chartqa/{split}/{split}_{partition}.json")
            rows = question_records(read_json(file), split, partition)
            split_rows.extend(rows)
            if split == "val":
                selections.extend(stable_selection(rows, config["questions_per_partition"], config["seed"]))
        if split == "train":
            names = sorted({r["image_name"] for r in split_rows})[:config["training_charts"]]
            train_questions = [r for r in split_rows if r["image_name"] in names]
            split_rows = train_questions
        else:
            split_rows = selections
        for image in sorted({r["image_name"] for r in split_rows}):
            for folder, name in (("png", image), ("tables", Path(image).stem + ".csv")):
                path = f"ChartQA Dataset/{split}/{folder}/{name}"
                try:
                    downloader.fetch(f"{base}/{urllib.parse.quote(path)}", f"data/raw/chartqa/{split}/{folder}/{name}")
                except urllib.error.HTTPError as error:
                    if folder == "tables" and error.code == 404:
                        continue
                    raise
    # Write once: changing the selected IDs requires an explicit new manifest location.
    manifest = {"schema_version": 1, "source_revision": revision, "seed": config["seed"],
                "selection": "sha256(seed:source_question_id), 50 per author partition by default",
                "questions": selections}
    manifest_path = root / "data/manifests/pilot.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Frozen pilot manifest differs; use a new workspace/configuration")
    write_json(manifest_path, manifest)
    write_jsonl(root / "data/manifests/training_questions.jsonl", train_questions)
    write_json(root / "data/manifests/chartqa_source.json", {
        "repository": "https://github.com/vis-nlp/ChartQA", "revision": revision,
        "license": "GPL-3.0 repository license; third-party chart provenance retained",
        "annotations": "Not distributed in this GitHub subset; missing, not empty ground truth",
        "files": {k: v for k, v in downloader.ledger["files"].items() if k.startswith("data/raw/chartqa/")}
    })
    return {"validation_questions": len(selections), "training_questions": len(train_questions)}


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes destination")
        zipped.extractall(destination)


def acquire_vistext(root, config):
    root = Path(root)
    downloader = Downloader(root, config["download_limit_bytes"])
    revision = config["vistext_revision"]
    base = f"https://raw.githubusercontent.com/mitvis/vistext/{revision}"
    for name in ("LICENSE", "README.md", "download_data.sh", "data/splits.json", "data/statista_mappings.json"):
        downloader.fetch(f"{base}/{name}", f"data/raw/vistext/source/{name}")
    results, artifacts = {}, []
    selected = sorted(read_json(root / "data/raw/vistext/source/data/splits.json")["train"], key=int)[:config["training_charts"]]
    for name in ("tabular", "scenegraphs", "images"):
        try:
            reader = RemoteZipReader(downloader, f"https://vis.csail.mit.edu/vistext/{name}.zip")
            with zipfile.ZipFile(reader) as archive:
                if name == "tabular":
                    wanted = ["data_train.json"]
                else:
                    wanted = [member for member in archive.namelist() if Path(member).stem in selected and member.endswith(".json" if name == "scenegraphs" else ".png")]
                    if len(wanted) != len(selected):
                        raise ValueError(f"Only {len(wanted)} selected training members found in {name}")
                for member in wanted:
                    destination = root / "data/raw/vistext" / (member if name != "images" else f"images/{Path(member).name}")
                    if not destination.resolve().is_relative_to((root / "data/raw/vistext").resolve()):
                        raise ValueError("Unsafe ZIP member")
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(archive.read(member))
                    artifacts.append({"archive": reader.metadata, "member": member, "sha256": sha256(destination), "bytes": destination.stat().st_size})
            results[name] = {"status": "selected_training_members_only", "members": len(wanted)}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            results[name] = {"status": "blocked", "error": str(error)}
            break
    write_json(root / "data/manifests/vistext_source.json", {
        "repository": "https://github.com/mitvis/vistext", "revision": revision,
        "license": "GPL-3.0 repository license; underlying Statista material has separate provenance",
        "selected_training_ids": selected, "archives": results, "members": artifacts,
        "files": {k: v for k, v in downloader.ledger["files"].items() if k.startswith("data/raw/vistext/")}
    })
    return results


def chartqa_record(root, question, revision):
    root = Path(root)
    split, name = question["split"], question["image_name"]
    image = Path("data/raw/chartqa") / split / "png" / name
    with Image.open(root / image) as im:
        im.verify()
    with Image.open(root / image) as im:
        width, height = im.size
    table_path = root / "data/raw/chartqa" / split / "tables" / (Path(name).stem + ".csv")
    table = None
    if table_path.exists():
        with table_path.open(encoding="utf-8-sig", newline="") as stream:
            table = list(csv.reader(stream))
    return {"schema_version": 1, "source": "chartqa", "source_revision": revision,
            **question, "image_path": image.as_posix(), "width": width, "height": height,
            "image_sha256": sha256(root / image), "table": table,
            "elements": None, "relationships": None, "chart_type": None,
            "coordinate_convention": "top-left pixel xyxy when annotated; unavailable in this subset"}


def normalize(root, config):
    root = Path(root)
    pilot = read_json(root / "data/manifests/pilot.json")
    if pilot["source_revision"] != config["chartqa_revision"]:
        raise ValueError("Configuration source revision differs from frozen manifest")
    for name, questions in (("pilot", pilot["questions"]), ("training", read_jsonl(root / "data/manifests/training_questions.jsonl"))):
        records = [chartqa_record(root, q, config["chartqa_revision"]) for q in questions]
        write_jsonl(root / f"data/normalized/{name}.jsonl", records)
    return {"pilot": len(pilot["questions"])}


def normalize_vistext(root, config):
    root = Path(root)
    source = root / "data/raw/vistext"
    manifest = read_json(root / "data/manifests/vistext_source.json")
    selected = set(manifest["selected_training_ids"])
    raw = read_json(source / "data_train.json")
    grouped = {}
    for row in raw:
        if str(row["img_id"]) in selected:
            if row["split"] != "train":
                raise ValueError("VisText training member has inconsistent split")
            grouped.setdefault(str(row["img_id"]), []).append(row)
    records, missing = [], []
    for identity in sorted(selected, key=int):
        image = source / f"images/{identity}.png"
        graph = source / f"scenegraphs/{identity}.json"
        if not image.exists() or not graph.exists() or identity not in grouped:
            missing.append(identity)
            continue
        with Image.open(image) as im:
            width, height = im.size
            im.verify()
        rows = grouped[identity]
        records.append({"schema_version": 1, "id": f"vistext:train:{identity}", "source": "vistext", "source_revision": config["vistext_revision"],
                        "split": "train", "partition": None, "image_name": image.name,
                        "image_path": image.relative_to(root).as_posix(), "image_sha256": sha256(image),
                        "width": width, "height": height, "question": None, "answers": None,
                        "table": None, "table_text": rows[0]["datatable"], "scenegraph": read_json(graph),
                        "linearized_scenegraph": rows[0]["scenegraph"],
                        "captions": [{"id": row["caption_id"], "l1": row["caption_L1"], "l2l3": row["caption_L2L3"]} for row in rows],
                        "elements": None, "relationships": None,
                        "coordinate_convention": "Original hierarchical Vega scenegraph coordinates; transforms to raster pixels not validated"})
    write_jsonl(root / "data/normalized/vistext_training.jsonl", records)
    report = {"selected": len(selected), "normalized": len(records), "missing_ids": missing,
              "table_format": "Source linearized text preserved; not silently parsed into rectangular cells",
              "pixel_grounding": "UNVERIFIED; hierarchical local coordinates require renderer transform validation"}
    write_json(root / "experiments/vistext_inspection.json", report)
    return report


def audit_records(records):
    ids, images, duplicates, problems = set(), {}, [], []
    for record in records:
        if record["id"] in ids:
            problems.append({"id": record["id"], "issue": "duplicate question ID"})
        ids.add(record["id"])
        key = record["image_sha256"]
        identity = f"{record['source']}:{record['split']}:{record['image_name']}"
        images.setdefault(key, set()).add(identity)
        table = record.get("table")
        if table and any(len(row) != len(table[0]) for row in table):
            problems.append({"id": record["id"], "issue": "ragged table"})
        if not table and not record.get("table_text"):
            problems.append({"id": record["id"], "issue": "missing table"})
        if record["width"] <= 0 or record["height"] <= 0:
            problems.append({"id": record["id"], "issue": "invalid dimensions"})
    for digest, identities in images.items():
        if len(identities) > 1:
            duplicates.append({"sha256": digest, "identities": sorted(identities)})
    return {"records": len(records), "unique_images": len(images), "duplicate_images": duplicates,
            "unparsed_linearized_tables": sum(bool(r.get("table_text")) and not r.get("table") for r in records),
            "problems": problems, "missing_elements": sum(r.get("elements") is None for r in records)}


def inspect(root, config):
    root = Path(root)
    train = read_jsonl(root / "data/normalized/training.jsonl")
    pilot = read_jsonl(root / "data/normalized/pilot.jsonl")
    report = {"training": audit_records(train), "validation": audit_records(pilot),
              "combined": audit_records(train + pilot),
              "annotation_correspondence": "Unavailable in the acquired GitHub release; not validated",
              "cross_dataset_overlap": "UNKNOWN until VisText image normalization succeeds"}
    vistext_path = root / "data/normalized/vistext_training.jsonl"
    if vistext_path.exists():
        vistext = read_jsonl(vistext_path)
        report["vistext_training"] = audit_records(vistext)
        chart_hashes = {r["image_sha256"] for r in train + pilot}
        report["cross_dataset_overlap"] = {"method": "exact image SHA256 over acquired samples only",
                                            "vistext_images": len(vistext), "chartqa_images": len(chart_hashes),
                                            "overlap": [r["id"] for r in vistext if r["image_sha256"] in chart_hashes],
                                            "limits": "Does not rule out re-rendered charts or shared source data"}
    write_json(root / "experiments/data_inspection.json", report)
    return report
