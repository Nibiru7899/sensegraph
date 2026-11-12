"""Regenerate the reviewable milestone report from completed local evidence."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensegraph.evaluation import score_run
from sensegraph.io import read_json, read_jsonl, write_json


def main():
    for model in ("unichart", "deplot"):
        run = read_json(ROOT / f"experiments/{model}/run.json")
        if not run["complete"] or run["processed"] != 100:
            raise RuntimeError(f"{model} pilot is incomplete; report was not finalized")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp=.cache/pytest", "--junitxml=experiments/test_results.xml"], cwd=ROOT, check=True)
    tests = len(ET.parse(ROOT / "experiments/test_results.xml").findall(".//testcase"))
    metrics = {name: score_run(ROOT, name) for name in ("unichart", "deplot")}
    downloads = read_json(ROOT / "experiments/download_ledger.json")
    gpu = read_json(ROOT / "experiments/gpu_budget.json")
    environment = read_json(ROOT / "experiments/environment.json")
    inspection = read_json(ROOT / "experiments/data_inspection.json")
    oracle = read_json(ROOT / "experiments/oracle/metrics.json")
    comparisons = {}
    for name in metrics:
        before = read_jsonl(ROOT / f"experiments/{name}/initial_run/predictions.jsonl")
        after = read_jsonl(ROOT / f"experiments/{name}/predictions.jsonl")
        comparisons[name] = {"records": len(after),
                             "raw_outputs_equal": sum(p.get("raw_output") == q.get("raw_output") for p, q in zip(before, after)),
                             "answers_equal": sum(p.get("answer") == q.get("answer") for p, q in zip(before, after)),
                             "statuses_equal": sum(p["status"] == q["status"] for p, q in zip(before, after))}
    predictions = read_jsonl(ROOT / "experiments/deplot/predictions.jsonl")
    statuses = dict(Counter(p["status"] for p in predictions))
    errors = dict(Counter(p.get("error") for p in predictions if p.get("error")))
    paired = metrics["deplot"]["paired"]
    now = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds")
    results = {"created_at": now, "kind": "executed_local_validation_pilot_not_research_validation", "tests_passed": tests,
               "metrics": metrics, "oracle": oracle, "repeat_comparison": comparisons,
               "download_accounted_bytes": downloads["transferred_bytes"], "session_seconds": gpu["charged_seconds"],
               "environment": environment, "deplot_statuses": statuses, "deplot_errors": errors}
    write_json(ROOT / "research/pilot_results.json", results)
    table = []
    for name, item in metrics.items():
        overall, timing, run = item["overall"], item["latency"], item["run"]
        table.append(f"| {name} | {overall['exact_correct']}/100 | {overall['relaxed_correct']}/100 | {overall['answered']}/100 | {timing['median_seconds']:.3f} s | {run['peak_allocated_bytes']/2**20:.1f} MiB |")
    breakdown = []
    for name, item in metrics.items():
        breakdown.append(f"- {name}: human {item['partitions']['human']['relaxed_correct']}/50; augmented {item['partitions']['augmented']['relaxed_correct']}/50.")
    candidate_count = paired["candidate_count"]
    paired_lines = [f"- {name}: {paired[name]['overall']['relaxed_correct']}/{candidate_count} relaxed-correct; {paired[name]['overall']['answered']}/{candidate_count} answered." for name in metrics]
    numeric = metrics["deplot"]["numeric_extraction"]
    report = f"""# SenseGraph baseline pilot

Generated {now} from executed local artifacts. This is a one-seed validation pilot, not a full benchmark reproduction, a trained SenseGraph model, or evidence for the graph hypothesis.

## Executed results

| Baseline | Exact correct | Relaxed correct | Answered | Median per-question latency | Peak PyTorch allocated memory |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

{chr(10).join(breakdown)}

All 100 selected questions remain in each overall denominator. DePlot statuses: `{statuses}`. Errors: `{errors}`. Unsupported forms and malformed/truncated generations do not count as correct answers. Latency includes preprocessing, decoding, and execution, with four repeated-image extraction cache opportunities; it is not pure GPU kernel latency. Memory is PyTorch allocated memory, not total device usage.

## Paired comparison and oracle

The fixed question-only operation-candidate subset contains {candidate_count}/100 questions. It was not selected using prediction correctness, reference tables, or answer labels. Ambiguous operands, unsupported visual semantics, and extraction failures remain in this denominator. This is lexical eligibility, not a claim that every candidate can be executed.

{chr(10).join(paired_lines)}

The separate reference-table oracle answered {oracle['overall']['answered']}/100 and was relaxed-correct on {oracle['overall']['relaxed_correct']}/100. It is not image-only inference. Its low coverage and remaining errors show that the frozen rule executor is a limited language/intent baseline. DePlot-vs-oracle differences do not isolate perception because generated and reference table labels can differ.

Strict table alignment matched {numeric['aligned_tables']}/{numeric['evaluated_tables']} unique images and {numeric['aligned_numeric_cells']} parsed numeric cells; {numeric['missing_numeric_cells']} reference-numeric cells in aligned tables could not be parsed from predictions. Repeated questions do not count the same chart again. Raw extraction MAE over parsed cells: `{numeric['mae']}`. This mixes source units and is a descriptive diagnostic, not a normalized cross-chart score; per-image errors are retained. No error metric is asserted for unaligned tables. Supporting predicted table-cell indices are not visual evidence; image boxes and calibrated confidence are unavailable.

## Reproducibility and resource use

- Python {environment['python'].split()[0]}, PyTorch {environment['packages']['torch']}, Transformers {environment['packages']['transformers']}, CUDA {environment['torch_cuda']}; {environment['gpu']}.
- Frozen pilot: 50 human plus 50 augmented ChartQA validation questions, seed 42, 96 images. Manifest SHA256: `{metrics['unichart']['run']['manifest_sha256']}`.
- {tests} tests passed in the final project-environment run; JUnit evidence: `experiments/test_results.xml`.
- Both checkpoints passed real-image smoke inference with socket connections disabled. DePlot uses the locally installed Arial font, whose hash is recorded in its run metadata.
- Total accounted downloads: {downloads['transferred_bytes']:,} bytes ({downloads['transferred_bytes']/1e9:.3f} decimal GB), including 200 MB conservative metadata/retry allowances. Cap: 10 GB.
- Aggregate charged model-session time: {gpu['charged_seconds']:.3f} seconds ({gpu['charged_seconds']/60:.2f} minutes), including loading, smoke failures, initial pilots, and verification reruns. Cap: 60 minutes. No sessions remain reserved/incomplete.
- Initial runs are preserved under each model's `initial_run`; corrected runs capture implementation hashes. Repeat comparisons: `{comparisons}`. A cache metadata regression was fixed without changing the frozen executor rules.

## Evidence limits and next milestone

UniChart is ChartQA-finetuned and was run on Transformers 4.57.3, while its authors recommend 4.28.1. This is a modern compatibility baseline, not exact reproduction of published scores. DePlot uses deterministic execution rather than its original downstream LLM reasoning. The narrow executor is not a strong general reasoning baseline, and the two systems have different answer coverage. No matched-coverage reliability improvement is claimed.

Inspected 20 ChartQA and 20 VisText training charts. Their sampled images had no exact cross-dataset overlap. VisText linearized tables and scenegraph local coordinates are preserved, not validated raster-box supervision. Some table dates appear offset relative to chart labels. ChartQA element annotations are absent from the acquired GitHub subset. See `research/data_inspection.md` for limits.

No model was trained. Graph relation F1, evidence IoU, calibration, robustness gains, multi-seed confidence intervals, and BLV usability remain unmeasured. Do not interpret unavailable measurements as passes or compare this small pilot directly with full-test leaderboard scores.

The next milestone should validate synthetic rendering/graph ground truth and obtain a reliable annotation/supervision mapping before graph learning. Improve the executor using separate development/training examples and freeze a new protocol before future evaluation; do not tune version 1 on this pilot's failures. UI work remains deferred.

## Re-run

From the repository root: `python scripts/bootstrap.py`, then the CLI sequence in `README.md`, and `.venv/Scripts/python.exe scripts/report_pilot.py`. Existing predictions are intentionally protected: preserve prior model output files in a new archive directory before another inference run. Do not reset resource ledgers to bypass the agreed limits. Full machine-readable results are in `research/pilot_results.json`; local raw predictions and artifact hashes remain under `experiments/` and `data/manifests/`.
"""
    (ROOT / "research/baseline_report.md").write_text(report, encoding="utf-8")
    status = f"""# SenseGraph project status

Updated {now}. **Foundation and bounded baseline pilot complete.** No training or graph-model claims.

## Implemented

- Isolated Python 3.11.9 runtime and environment, resolved dependency lock, official CUDA PyTorch wheels.
- Persistent bounded acquisition with artifact hashes, pinned sources/checkpoints, resumable files, selective VisText ZIP members.
- ChartQA/VisText normalization and inspection, immutable pilot selection, explicit missing annotations and coordinate limits.
- Local-only UniChart and DePlot inference, bounded GPU ledger, conservative numerical executor, official relaxed scorer, paired/oracle diagnostics.
- Primary-source literature review, interface documentation, reproducible commands and generated baseline report.

## Executed verification

- `pip check`: no broken requirements.
- Final `pytest`: **{tests} passed**, recorded in `experiments/test_results.xml`.
- CUDA tensor execution and model imports passed.
- Both real-image smoke checks succeeded with socket connections disabled; earlier DePlot font failures are preserved separately.
- Both final pilots processed **100/100** questions. Verification repeat agreement: `{comparisons}`.
- Large runtime, data and checkpoint files were verified ignored by Git. Repository initialized; no commit or remote publication performed.

## Actual results

- UniChart relaxed accuracy: **{metrics['unichart']['overall']['relaxed_correct']}/100**; DePlot plus frozen executor: **{metrics['deplot']['overall']['relaxed_correct']}/100**, answering {metrics['deplot']['overall']['answered']}/100.
- Paired lexical candidates: {candidate_count}/100; detailed denominators and failures are in `research/baseline_report.md`.
- Acquired/inspected 20 training charts per dataset; no exact cross-dataset image overlap in these samples. ChartQA pilot contains 96 validation images.
- Accounted transfers **{downloads['transferred_bytes']/1e9:.3f}/10 GB**. Charged model sessions **{gpu['charged_seconds']/60:.2f}/60 minutes** including verification runs.

## Limitations and next engineering tasks

- No remaining environment/download blocker for this milestone. WSL was inaccessible and was not used.
- Missing ChartQA element annotations, unvalidated VisText raster transforms/date conventions, possible shared-source overlap, and narrow executor semantics limit later research use.
- UniChart library version differs from the authors' recommended reproduction environment; DePlot reasoning is a modified deterministic baseline.
- Trainable graph modules, the large synthetic generator, grounding, uncertainty calibration, robustness studies, and the accessible application remain deferred.
- Stop at this milestone. Next: synthetic ground-truth validation and annotation mapping, then controlled graph-model experiments with proper development/evaluation separation.
"""
    (ROOT / "PROJECT_STATUS.md").write_text(status, encoding="utf-8")
    print(f"Wrote baseline report and project status from {tests} passing tests and two complete pilots.")


if __name__ == "__main__":
    main()
