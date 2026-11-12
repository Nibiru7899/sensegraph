"""Sample-level exact/near-image checks, plus template/data split identity checks."""
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from .synthetic import fingerprint


def difference_hash(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (array[:, 1:] > array[:, :-1]).reshape(-1)
    return sum(int(bit) << i for i, bit in enumerate(bits))


def leakage_report(records, root, threshold=4, real_records=()):
    root = Path(root)
    violations, exact, near, cross_source = [], [], [], []
    identities = {}
    for record in records:
        if record["id"] in identities:
            violations.append({"kind": "duplicate_id", "id": record["id"]})
        identities[record["id"]] = record
    hashes = {r["id"]: difference_hash(root / r["image_path"]) for r in records}
    numeric = {r["id"]: fingerprint(r["spec"]["data"]["values"]) for r in records}
    for a, b in combinations(records, 2):
        pair = {"ids": [a["id"], b["id"]], "splits": [a["split"], b["split"]]}
        if a["image_sha256"] == b["image_sha256"]:
            exact.append(pair)
        distance = (hashes[a["id"]] ^ hashes[b["id"]]).bit_count()
        if distance <= threshold:
            near.append({**pair, "hamming_distance": distance})
        if a["split"] != b["split"]:
            if numeric[a["id"]] == numeric[b["id"]]:
                violations.append({**pair, "kind": "numerical_configuration"})
            for field in ("template_hash", "data_hash"):
                if a["spec"][field] == b["spec"][field]:
                    violations.append({**pair, "kind": field})
            if a["image_sha256"] == b["image_sha256"]:
                violations.append({**pair, "kind": "exact_image"})
    for real in real_records:
        perceptual = difference_hash(root / real["image_path"])
        for synthetic in records:
            distance = (perceptual ^ hashes[synthetic["id"]]).bit_count()
            if distance <= threshold or real["image_sha256"] == synthetic["image_sha256"]:
                cross_source.append({"synthetic": synthetic["id"], "real": real["id"], "hamming_distance": distance,
                                     "exact": real["image_sha256"] == synthetic["image_sha256"]})
    return {"violations": violations, "exact_duplicates": exact, "near_duplicates": near,
            "cross_source_candidates": cross_source, "near_duplicate_threshold": threshold,
            "method": "64-bit grayscale difference hash; candidates require visual review, not proof of common source",
            "limits": "Only acquired samples checked; hash distance misses some semantic/rerendered overlap"}
