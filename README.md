# SenseGraph

The full FP32 visual-quality campaign uses separate original/corrected target runs, validation-only checkpoint selection, a frozen independent synthetic audit, and bounded OCR/legacy graph-transfer diagnostics. See the live [full perception report](research/full_perception_training_report.md) and [current status](PROJECT_STATUS.md) for actual executed results. Five hours is the cumulative GPU ceiling; early stopping may finish sooner. Earlier pilot failures below remain historical. Detection/OCR metrics do not establish completed numerical QA or real-world accessibility.

Current result: **0.8748 synthetic holdout AP50; 114 tests pass**. Real partial-label recall is only 77/616 boxes and detected-region OCR word F1 is 0.3078, so reliable real-chart QA remains blocked. See the [trained visual model card](research/trained_visual_model_card.md) for release hashes, inference contracts and limitations. OCR is optional and experimental.

```powershell
.\.venv\Scripts\python.exe -m sensegraph.corrected_visual_training verify
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-local
.\.venv\Scripts\python.exe scripts/report_visual_quality.py
# Once release checkpoints exist, image-only CPU inference with optional trained OCR:
.\.venv\Scripts\python.exe scripts/infer_full_visual.py --image data/synthetic/v3/images/synthetic-01400.png --output experiments/visual_training/release/inference_example.json --with-ocr
```

Protocols: [full visual curriculum](research/full_visual_training_protocol.md), [collision correction](research/corrected_visual_training_protocol.md), [OCR](research/ocr_visual_protocol.md), [frozen graph transfer](research/frozen_graph_transfer_protocol.md), [partial real-image audit](research/real_perception_audit_protocol.md). Training/evaluation namespaces refuse silent repeated attempts. Preserve them; use a separately versioned, preregistered study for new experiments. The independent audit may not drive further tuning. No frontend, deployment or full QA training is included.

Milestone 4's [bounded detector experiment](research/milestone4_report.md) executed after the user acquired the official checkpoint. Smoke checks passed and the frozen 240-step fit completed, but the validation detection gate **failed**: macro recall@.50 0.170 and AP@.50 0.113; point/tick recall zero. Work stopped at this perception bottleneck. The 84-test suite passes; 5,111 historical files remain unchanged. See the [frozen protocol](research/milestone4_protocol.md) and [current status](PROJECT_STATUS.md). No imperfect-proposal GAT improvement is claimed.

```powershell
# CPU-only verification and exports for the completed attempt; do not refit it.
$python = '.\.venv\Scripts\python.exe'
& $python -m sensegraph.perception_experiment verify
& $python scripts/export_perception_errors.py
& $python scripts/report_perception.py
```

Milestone 3 is complete: [controlled relation-learning report](research/milestone3_report.md), [frozen protocol](research/milestone3_protocol.md), and [current status](PROJECT_STATUS.md). The three-seed synthetic oracle-proposal comparison gives macro F1 0.7629 for GATv2, 0.7409 for the graph-free MLP, and 0.4472 for fixed rules. This does not establish real-image extraction or chart-QA improvements.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m sensegraph.relation_experiment check-history
& $python -m pytest -q --basetemp=.cache/pytest-m3 --junitxml=experiments/milestone3/tests.xml
& $python scripts/report_relations.py
```

The completed experiment is sealed: fitting and calibration in this namespace are rejected. The executed sequence was `freeze`, `verify-v2`, `expand`, `fit-v3`, the documented pre-test guard amendment and final `freeze`, then `seal`, `test`. Each command uses `python -m sensegraph.relation_experiment`; configs are `configs/relations.json` and `configs/structural_v3.json`. A new study requires a separate versioned experiment namespace and frozen protocol. Checkpoints, thresholds, raw predictions, tables, learning curves and resource accounting are preserved under `experiments/milestone3`. `sensegraph.relation_training.load_checkpoint` and `infer` provide cached-graph inference with targets separate from forward inputs. See the report for timing boundaries and reproducibility limits.

## Preserved Milestone 2

Milestone 2 structural-data tools and results: [milestone2_report.md](research/milestone2_report.md), [schema and training boundary](research/structural_data_schema.md), and [real annotation mapping](research/real_annotation_mapping.md). The baseline report below is historical and remains unchanged.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python scripts/install_structural.py
# Generates exactly 100 charts; refuses to overwrite an existing output version.
& $python -m sensegraph.structural_cli generate --config configs/structural.json
& $python -m sensegraph.structural_cli acquire-real
& $python -m sensegraph.structural_cli map-real
& $python -m sensegraph.structural_cli validate
& $python -m pytest -q --basetemp=.cache/pytest-m2 --junitxml=experiments/milestone2/tests.xml
& $python scripts/report_structural.py
```

For this completed checkout, skip `generate` and `acquire-real` when inputs are already present. Use a new versioned output path in a copied configuration to regenerate. The generator supports only linear-value axes and categorical x axes in this milestone; real temporal/calibration ambiguities are preserved rather than guessed. Validation traverses all three splits on CPU, pads images without scaling, tests PyG batching, and checks immutable baseline hashes/scores. It uses one CPU thread to avoid overhead from small-region reductions. Overlays include separate mark, context, and lookup-evidence layers for nine representative charts. No frontend or GNN training is run.

The original `requirements.lock` remains the baseline lock; `requirements-structural.lock` pins added dependencies. The installed optional package extra is `.[structural]`. New code uses `src/sensegraph/{synthetic,structural_validation,graph_data,leakage,real_mapping,structural_cli}.py`; the previously reserved directories remain unimplemented placeholders. Do not use the historical `report_pilot.py` to refresh Milestone 2 status; `report_structural.py` verifies/reports this milestone without rewriting baseline reports.

Structure-aware and evidence-grounded chart-understanding research. The earlier structural-data milestone and Windows UniChart / DePlot pilot remain preserved below. Milestone 3 tests relationship prediction with synthetic oracle proposals; calibrated confidence, real-image evidence prediction and an accessible application remain unimplemented.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for executed checks and actual results, and [research/literature_review.md](research/literature_review.md) for primary-source context and limitations.

## Windows setup

Run from the repository root in PowerShell. An existing Python is needed only to bootstrap the isolated Python 3.11 runtime. The default system interpreter is unchanged.

```powershell
python scripts/bootstrap.py
$python = '.\.venv\Scripts\python.exe'
& $python -m pip check
& $python -m sensegraph.cli environment
& $python -m pytest -q --basetemp=.cache/pytest
```

Bootstrap uses the official Python 3.11.9 Windows installer and official CUDA 12.8 PyTorch wheels. It stages dependency wheels, verifies published SHA256 values when available, installs a project-local environment, and writes `requirements.lock`. Re-running reuses verified artifacts and resumes partial transfers. The ledger includes a conservative 100 MB allowance for metadata, dependency resolution, and preliminary source inspection. Do not run acquisition/bootstrap commands concurrently: the serial ledger is intentionally single-writer.

The default transfer cap is 10,000,000,000 bytes. Dependencies, Python, checkpoints, and datasets all count. Large files are rejected when their declared size exceeds the remaining budget. Failed-transfer bytes remain charged. There are no paid API calls.

## Reproduce the pilot

```powershell
& $python -m sensegraph.cli acquire --dataset chartqa
& $python -m sensegraph.cli normalize
& $python -m sensegraph.cli inspect
& $python -m sensegraph.cli acquire --dataset vistext
& $python -m sensegraph.cli normalize --dataset vistext
& $python -m sensegraph.cli inspect
& $python -m sensegraph.cli acquire --models --model unichart
& $python -m sensegraph.cli acquire --models --model deplot
& $python -m sensegraph.cli infer --model unichart --smoke
& $python -m sensegraph.cli infer --model unichart
& $python -m sensegraph.cli infer --model deplot --smoke
& $python -m sensegraph.cli infer --model deplot
& $python -m sensegraph.cli score --model unichart
& $python -m sensegraph.cli score --model deplot
& $python -m sensegraph.cli oracle
```

All commands accept `--config configs/pilot.json` and `--root .`. The `sensegraph` console command is equivalent to `python -m sensegraph.cli`.

Selection uses stable source IDs and SHA256(seed:ID), seed 42, with 50 human and 50 augmented validation questions. Official test data is not acquired. Raw inputs stay unchanged; normalization preserves unavailable annotations as `null`. A frozen selection cannot be silently overwritten with different IDs.

Models load only local cached checkpoint files. CUDA batch size is one; models run sequentially in inference mode, FP16, deterministic beam decoding. The GPU ledger reserves time before model loading and records actual elapsed session time, conservatively counting loading and CPU overhead. Each pilot has 25 minutes and each smoke check five minutes within a shared one-hour budget. Interrupted sessions retain their reservation. Outputs are flushed after each question; existing prediction files are never silently overwritten.

## Outputs and interpretation

- `data/manifests/`: frozen pilot, source revisions, licenses, artifact sizes and hashes.
- `data/normalized/`: versioned records; questions/answers and reference tables are evaluation data only.
- `experiments/environment.json`: actual runtime and CUDA verification.
- `experiments/data_inspection.json`: missing labels, malformed tables, duplicates and overlap findings.
- `experiments/{model}/`: real predictions, raw model outputs, timing/memory, run provenance, and metrics.
- `experiments/oracle/`: separately labeled reference-table diagnostic, not image-only model performance.
- `experiments/download_ledger.json` and `gpu_budget.json`: persisted resource accounting.

Exact match (case sensitive, trimmed) and ChartQA relaxed accuracy include missing/failed predictions in denominators. Relaxed scoring follows the official Pix2Struct ChartQA implementation, including percentage conversion and its zero-target string fallback. Paired metrics use fixed question-only operation candidates, not a subset selected for successful extraction. Executor coverage is reported separately. Baseline confidence and image evidence are unavailable (`null`).

The DePlot executor supports conservative lookup, extrema, comparison, absolute difference, and counting forms. It rejects color-dependent, compound, ambiguous, and tied cases. Unsupported questions are not proof that the underlying chart is unreadable. Numeric percentage cells use percentage points for arithmetic; ChartQA scoring independently follows benchmark percentage conversion. Only exactly aligned table labels/order qualify for extraction MAE.

DePlot's prompt renderer needs a TrueType font even with checkpoint files cached. The adapter directly invokes Pix2Struct image preprocessing with locally installed Windows Arial; its path/hash is recorded, and no font is downloaded or redistributed. Set `deplot_font_path` in configuration on another host. Successful smoke checks were also executed with socket connections disabled.

VisText acquisition uses HTTPS range reads to fetch the training JSON member and selected training images/scenegraphs, without downloading benchmark test payloads. Failed member transfers can be retried by repeating acquisition. Raw scenegraph coordinates and linearized tables are preserved; they are not validated pixel boxes or parsed numerical tables. See [research/data_inspection.md](research/data_inspection.md) and [research/interfaces.md](research/interfaces.md).

UniChart authors recommend Transformers 4.28.1 and warn that versions can affect accuracy. This pilot starts with the agreed 4.57.3 and is a compatibility baseline, not an exact reproduction. DePlot plus deterministic execution differs from DePlot's original downstream LLM reasoning. A 100-question validation pilot does not establish research-target improvements or generalization.

## Layout

`src/sensegraph` contains implemented acquisition, data, reasoning, model adapters, evaluation, runner, and CLI. The requested `src/models`, `src/training`, `src/evaluation`, `synthetic_generator`, and `demo` directories reserve later phases; they contain no claimed implementations. `configs`, `tests`, and `research` contain the current configuration, checks, and review. Runtime/data/experiment directories are ignored by Git.
