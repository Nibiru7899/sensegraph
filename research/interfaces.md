# Pilot interfaces

## CLI

`sensegraph COMMAND --config configs/pilot.json --root .` emits JSON and returns a nonzero status on execution failure. Commands: `environment`, `acquire`, `normalize`, `inspect`, `infer`, `score`, `oracle`. Acquisition/normalization accepts `--dataset chartqa|vistext`; model acquisition adds `--models --model unichart|deplot`. Inference accepts `--model` and optional `--smoke`.

Commands preserve completed predictions and reject accidental overwrites. A completed run can contain failed/unsupported records; completion means all selected records were attempted, not that every prediction is correct. The run summary and per-record statuses must both be inspected.

## Configuration

The checked-in JSON pins source/model revisions, licenses, seed, sample counts, decoding parameters, and resource limits. Paths resolve under `--root`; an optional `deplot_font_path` selects a locally installed font on hosts without Windows Arial. Fonts are not redistributed. Defaults use one model at a time on CUDA, FP16, four beams, 128 new tokens for UniChart and 512 for DePlot.

## Normalized record, schema version 1

| Field | Meaning |
| --- | --- |
| `id`, `source`, `source_revision` | Stable source identity, dataset, immutable revision |
| `split`, `partition` | Original split and ChartQA human/augmented partition; partition is null for VisText |
| `image_path`, `image_name`, `image_sha256`, `width`, `height` | Relative image location, identity, and verified dimensions |
| `question`, `answers` | ChartQA question and reference answer list; null for VisText captions |
| `table` | Raw rectangular ChartQA CSV cells; null when unavailable/unparsed |
| `elements`, `relationships` | Null in this milestone; no inferred ground truth |
| `coordinate_convention` | Explicit limitations on any structural coordinates |
| `table_text`, `scenegraph`, `linearized_scenegraph`, `captions` | VisText source representations, preserved without conversion into raster boxes |

Reference answers and tables remain in evaluation records. The model adapter accepts **only an image path and question**, returning generated text. The deterministic executor receives the question and **predicted** table; the separately labeled oracle receives reference tables.

## Prediction and run artifacts

JSONL predictions contain `id`, `answer`, `status`, `raw_output`, generation completion/limits, extraction cache status, latency, precision, and peak PyTorch memory. Errors retain a reason and null answer. DePlot adds predicted cells/table, operation, rule version, and supporting table-cell indices. These cell indices are **not image evidence**. Image boxes, graph nodes, and confidence remain null.

Run JSON records expected/processed counts, completion, checkpoint identity, complete configuration, manifest/record/rule/implementation hashes, session elapsed time, precision fallback, and font provenance where applicable. Budget JSON reserves session time before loading; interrupted sessions retain their reservation until explicitly investigated.

Scoring includes failed or missing predictions in denominators. Exact match is case sensitive after trimming outer whitespace. Relaxed correctness follows [the official Pix2Struct ChartQA implementation](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py): numeric percentages are converted to fractions, nonzero numerical targets permit 5% relative error, and other targets use lowercase string equality. Zero therefore uses the string fallback. Scorer-source SHA256 is recorded alongside metrics.
