from PIL import Image
import pytest

from sensegraph.acquisition import BudgetExceeded, Downloader
from sensegraph.data import audit_records, chartqa_record, normalize, question_records, stable_selection
from sensegraph.evaluation import aligned_numeric_error, chartqa_relaxed, score_records, score_run
from sensegraph.io import read_jsonl, sha256, write_json, write_jsonl
from sensegraph.reasoning import execute, number, parse_table


TABLE = [["Category", "Value"], ["Alpha", "-10"], ["Beta", "20%"], ["Gamma", "0"]]


@pytest.mark.parametrize("raw,expected", [("-1,200", -1200), ("20%", 20), ("−2.5", -2.5), ("0", 0)])
def test_numbers(raw, expected):
    assert float(number(raw)) == expected


@pytest.mark.parametrize("raw", ["NaN", "inf", "2m", "12 dollars", ""])
def test_ambiguous_numbers(raw):
    assert number(raw) is None


@pytest.mark.parametrize("question,answer", [
    ("What is the value for Alpha?", "-10"),
    ("What is the difference between Alpha and Beta?", "30"),
    ("Which category has the highest value?", "Beta"),
    ("What is the minimum value?", "-10"),
    ("Which is lower, Alpha or Gamma?", "Alpha"),
    ("How many data points are above 0?", "1"),
    ("How many categories are there?", "3"),
])
def test_executor(question, answer):
    assert execute(question, TABLE)["answer"] == answer


def test_unsupported_and_ties():
    assert execute("What color is the chart?", TABLE)["status"] == "unsupported"
    assert execute("What is the average value?", TABLE)["status"] == "unsupported"
    assert execute("Which blue bar is highest?", TABLE)["status"] == "unsupported"
    tied = [["Category", "Value"], ["A", "2"], ["B", "2"]]
    assert execute("Which category is highest?", tied)["answer"] is None
    assert execute("What is the value for Alpha above 10?", TABLE)["answer"] is None
    assert execute("How many data points are above 0 and below 10?", TABLE)["answer"] is None
    assert execute("How many data points are above or equal to 0?", TABLE)["answer"] is None


def test_parse_table():
    assert parse_table("TITLE | ignored title\nCategory | Value<0x0A>Alpha | 2") == [["Category", "Value"], ["Alpha", "2"]]
    with pytest.raises(ValueError, match="Ragged"):
        parse_table("Category | A | B\nX | 2")
    with pytest.raises(ValueError, match="Duplicate row"):
        parse_table("Category | A\nX | 2\nX | 3")


@pytest.mark.parametrize("pred,target,correct", [
    ("105", "100", True), ("105.01", "100", False),
    ("-105", "-100", True), ("0.01", "0", False),
    ("0", "0", True), ("0.5", "50%", True), ("Blue", "blue", True),
    ("NaN", "100", False), ("infinity", "100", False), ("NaN", "NaN", False), ("0.0", "0", False)
])
def test_metric(pred, target, correct):
    assert chartqa_relaxed(pred, target) == correct


def test_selection_independent_of_input_order():
    rows = [{"id": str(i)} for i in range(20)]
    assert stable_selection(rows, 5, 42) == stable_selection(list(reversed(rows)), 5, 42)


def test_malformed_record():
    with pytest.raises(ValueError, match="Malformed"):
        question_records([{"imgname": "a.png"}], "train", "human")
    with pytest.raises(ValueError, match="Unsafe"):
        question_records([{"imgname": "../a.png", "query": "q", "label": "a"}], "train", "human")


def fixture_root(tmp_path):
    question = {"id": "chartqa:val:human:0", "image_name": "fixture.png", "question": "What is the maximum value?",
                "answers": ["20"], "split": "val", "partition": "human"}
    path = tmp_path / "data/raw/chartqa/val/png/fixture.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (30, 20), "white").save(path)
    write_json(tmp_path / "data/manifests/pilot.json", {"source_revision": "revision", "questions": [question]})
    write_jsonl(tmp_path / "data/manifests/training_questions.jsonl", [])
    return question


def test_offline_pipeline_wiring_only(tmp_path):
    fixture_root(tmp_path)
    normalize(tmp_path, {"chartqa_revision": "revision"})
    rows = read_jsonl(tmp_path / "data/normalized/pilot.jsonl")
    assert rows[0]["split"] == "val"
    assert rows[0]["elements"] is None and rows[0]["table"] is None
    assert rows[0]["width"] == 30
    # Stub establishes serialization and scoring wiring, not model behavior.
    def stub(image, question):
        assert image.name == "fixture.png"
        assert isinstance(question, str)
        return "20"
    predictions = [{"id": row["id"], "status": "ok", "answer": stub(tmp_path / row["image_path"], row["question"])} for row in rows]
    destination = tmp_path / "stub_predictions.jsonl"
    write_jsonl(destination, predictions)
    assert score_records(rows, read_jsonl(destination))["overall"]["relaxed_accuracy"] == 1
    failed = [{"id": rows[0]["id"], "status": "unsupported", "answer": None}]
    assert score_records(rows, failed)["overall"]["denominator"] == 1
    assert score_records(rows, failed)["overall"]["relaxed_accuracy"] == 0
    assert score_records(rows, [])["overall"]["denominator"] == 1


def test_duplicates_and_missing_annotations(tmp_path):
    q = fixture_root(tmp_path)
    record = chartqa_record(tmp_path, q, "revision")
    copy = {**record, "id": "other", "split": "train"}
    audit = audit_records([record, copy])
    assert len(audit["duplicate_images"]) == 1
    assert audit["missing_elements"] == 2


def test_prediction_identity_validation():
    rows = [{"id": "a", "partition": "human", "answers": ["1"]}]
    with pytest.raises(ValueError):
        score_records(rows, [{"id": "unknown"}])
    with pytest.raises(ValueError):
        score_records(rows, [{"id": "a"}, {"id": "a"}])


def test_download_budget_persists(tmp_path):
    downloader = Downloader(tmp_path, 100)
    downloader.charge(75)
    with pytest.raises(BudgetExceeded):
        Downloader(tmp_path, 100).charge(26)
    assert Downloader(tmp_path, 100).remaining == 25


def test_alignment_requires_labels():
    assert aligned_numeric_error(TABLE, TABLE)["mae"] == 0
    assert aligned_numeric_error(TABLE, [["Other", "Value"], *TABLE[1:]])["status"] == "unaligned"


def test_extraction_metrics_count_unique_images_and_missing_values(tmp_path):
    rows = [{"id": str(i), "partition": "human", "answers": ["1"], "image_sha256": "same-image", "table": TABLE} for i in range(2)]
    data_path = tmp_path / "data/normalized/pilot.jsonl"
    write_jsonl(data_path, rows)
    write_json(tmp_path / "experiments/deplot/run.json", {"records_sha256": sha256(data_path)})
    predicted = [TABLE[0], ["Alpha", "unknown"], *TABLE[2:]]
    write_jsonl(tmp_path / "experiments/deplot/predictions.jsonl", [{"id": str(i), "status": "unsupported", "predicted_table": predicted} for i in range(2)])
    result = score_run(tmp_path, "deplot")["numeric_extraction"]
    assert result["evaluated_tables"] == 1
    assert result["aligned_numeric_cells"] == 2
    assert result["missing_numeric_cells"] == 1
