# SenseGraph baseline pilot

Generated 2026-11-03T15:14:29+05:30 from executed local artifacts. This is a one-seed validation pilot, not a full benchmark reproduction, a trained SenseGraph model, or evidence for the graph hypothesis.

## Executed results

| Baseline | Exact correct | Relaxed correct | Answered | Median per-question latency | Peak PyTorch allocated memory |
| --- | --- | --- | --- | --- | --- |
| unichart | 58/100 | 65/100 | 100/100 | 0.140 s | 651.7 MiB |
| deplot | 12/100 | 12/100 | 15/100 | 3.742 s | 1250.2 MiB |

- unichart: human 24/50; augmented 41/50.
- deplot: human 6/50; augmented 6/50.

All 100 selected questions remain in each overall denominator. DePlot statuses: `{'unsupported': 77, 'ok': 15, 'error': 8}`. Errors: `{'ValueError: Ragged predicted table': 6, 'ValueError: Generation stopped before EOS; output retained but not scored as an answer': 2}`. Unsupported forms and malformed/truncated generations do not count as correct answers. Latency includes preprocessing, decoding, and execution, with four repeated-image extraction cache opportunities; it is not pure GPU kernel latency. Memory is PyTorch allocated memory, not total device usage.

## Paired comparison and oracle

The fixed question-only operation-candidate subset contains 56/100 questions. It was not selected using prediction correctness, reference tables, or answer labels. Ambiguous operands, unsupported visual semantics, and extraction failures remain in this denominator. This is lexical eligibility, not a claim that every candidate can be executed.

- unichart: 36/56 relaxed-correct; 56/56 answered.
- deplot: 12/56 relaxed-correct; 15/56 answered.

The separate reference-table oracle answered 14/100 and was relaxed-correct on 11/100. It is not image-only inference. Its low coverage and remaining errors show that the frozen rule executor is a limited language/intent baseline. DePlot-vs-oracle differences do not isolate perception because generated and reference table labels can differ.

Strict table alignment matched 58/96 unique images and 757 parsed numeric cells; 0 reference-numeric cells in aligned tables could not be parsed from predictions. Repeated questions do not count the same chart again. Raw extraction MAE over parsed cells: `15.495660155848958`. This mixes source units and is a descriptive diagnostic, not a normalized cross-chart score; per-image errors are retained. No error metric is asserted for unaligned tables. Supporting predicted table-cell indices are not visual evidence; image boxes and calibrated confidence are unavailable.

## Reproducibility and resource use

- Python 3.11.9, PyTorch 2.8.0+cu128, Transformers 4.57.3, CUDA 12.8; NVIDIA GeForce RTX 4060 Laptop GPU.
- Frozen pilot: 50 human plus 50 augmented ChartQA validation questions, seed 42, 96 images. Manifest SHA256: `134639355f93393a9006f36fdb290100f1639a185681e53be2ebd44be79d3cc1`.
- 44 tests passed in the final project-environment run; JUnit evidence: `experiments/test_results.xml`.
- Both checkpoints passed real-image smoke inference with socket connections disabled. DePlot uses the locally installed Arial font, whose hash is recorded in its run metadata.
- Total accounted downloads: 5,712,639,877 bytes (5.713 decimal GB), including 200 MB conservative metadata/retry allowances. Cap: 10 GB.
- Aggregate charged model-session time: 1038.673 seconds (17.31 minutes), including loading, smoke failures, initial pilots, and verification reruns. Cap: 60 minutes. No sessions remain reserved/incomplete.
- Initial runs are preserved under each model's `initial_run`; corrected runs capture implementation hashes. Repeat comparisons: `{'unichart': {'records': 100, 'raw_outputs_equal': 100, 'answers_equal': 100, 'statuses_equal': 100}, 'deplot': {'records': 100, 'raw_outputs_equal': 100, 'answers_equal': 100, 'statuses_equal': 100}}`. A cache metadata regression was fixed without changing the frozen executor rules.

## Evidence limits and next milestone

UniChart is ChartQA-finetuned and was run on Transformers 4.57.3, while its authors recommend 4.28.1. This is a modern compatibility baseline, not exact reproduction of published scores. DePlot uses deterministic execution rather than its original downstream LLM reasoning. The narrow executor is not a strong general reasoning baseline, and the two systems have different answer coverage. No matched-coverage reliability improvement is claimed.

Inspected 20 ChartQA and 20 VisText training charts. Their sampled images had no exact cross-dataset overlap. VisText linearized tables and scenegraph local coordinates are preserved, not validated raster-box supervision. Some table dates appear offset relative to chart labels. ChartQA element annotations are absent from the acquired GitHub subset. See `research/data_inspection.md` for limits.

No model was trained. Graph relation F1, evidence IoU, calibration, robustness gains, multi-seed confidence intervals, and BLV usability remain unmeasured. Do not interpret unavailable measurements as passes or compare this small pilot directly with full-test leaderboard scores.

The next milestone should validate synthetic rendering/graph ground truth and obtain a reliable annotation/supervision mapping before graph learning. Improve the executor using separate development/training examples and freeze a new protocol before future evaluation; do not tune version 1 on this pilot's failures. UI work remains deferred.

## Re-run

From the repository root: `python scripts/bootstrap.py`, then the CLI sequence in `README.md`, and `.venv/Scripts/python.exe scripts/report_pilot.py`. Existing predictions are intentionally protected: preserve prior model output files in a new archive directory before another inference run. Do not reset resource ledgers to bypass the agreed limits. Full machine-readable results are in `research/pilot_results.json`; local raw predictions and artifact hashes remain under `experiments/` and `data/manifests/`.
