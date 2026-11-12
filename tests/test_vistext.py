from PIL import Image
import pytest

from sensegraph.data import normalize_vistext
from sensegraph.io import read_jsonl, write_json


def prepare(tmp_path, split="train"):
    write_json(tmp_path / "data/manifests/vistext_source.json", {"selected_training_ids": ["1", "2"]})
    source = tmp_path / "data/raw/vistext"
    write_json(source / "data_train.json", [{"img_id": "1", "split": split, "datatable": "Title <s> Year Value 2020 1",
                                           "scenegraph": "source local coordinates", "caption_id": "1_01",
                                           "caption_L1": "A chart", "caption_L2L3": "A trend"}])
    write_json(source / "scenegraphs/1.json", {"marktype": "group", "items": [{"x": -10}]})
    image = source / "images/1.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (20, 30)).save(image)


def test_vistext_preserves_source_and_missing_ids(tmp_path):
    prepare(tmp_path)
    result = normalize_vistext(tmp_path, {"vistext_revision": "fixture"})
    assert result["normalized"] == 1 and result["missing_ids"] == ["2"]
    record = read_jsonl(tmp_path / "data/normalized/vistext_training.jsonl")[0]
    assert record["split"] == "train"
    assert record["elements"] is None and record["relationships"] is None
    assert record["table"] is None and record["table_text"].startswith("Title")
    assert record["scenegraph"]["items"][0]["x"] == -10


def test_vistext_rejects_split_mismatch(tmp_path):
    prepare(tmp_path, "test")
    with pytest.raises(ValueError, match="inconsistent split"):
        normalize_vistext(tmp_path, {"vistext_revision": "fixture"})
