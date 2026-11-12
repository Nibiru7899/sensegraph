# Real annotation mapping investigation

Reviewed 2026-11-03. Original raw/normalized datasets, acquisition manifests, baseline predictions, and baseline reports remain immutable. New sources/conversions are under `data/real_annotations/`; results and overlays are under `experiments/milestone2/`.

## ChartQA: missing annotations resolved for the 20 training identities

The GitHub subset acquired in Milestone 1 genuinely lacks element annotations. The [official repository](https://github.com/vis-nlp/ChartQA) links the [author's full release](https://huggingface.co/datasets/ahmed-masry/ChartQA), which contains `ChartQA Dataset.zip`. The inspected release revision is `af8b6f5c08c95085271561c2a3f9d15f2b5a9031`; archive size is 875,370,872 bytes. Dataset metadata declares GPL-3.0; underlying third-party chart rights/provenance remain applicable.

Selective HTTPS ZIP reads acquire only annotation JSON and paired PNG members for the existing 20 training IDs. No full archive or official test payload is fetched. A separate source manifest records revision, archive ETag, member CRC, byte counts, SHA256, and split. Matching full-release/original image hashes identify intentional repeated source identities, not independent additional examples.

### Supported conversion

- Source boxes are dictionaries `{x,y,w,h}` on the paired raster. Conversion is `[x,y,x+w,y+h]`, with identity scale. No normalization/crop/resize is guessed. Paired images and overlays validate the mapping on the acquired sample.
- Plot-area/title/major-axis-label/legend-label boxes are copied only when supplied. Missing axis lines, tick marks, text geometry, or fields remain unavailable.
- Bar `models[].bboxes`, `x`, and `y` align one-to-one. Raw source values and categories are preserved, including percentage strings. Model grouping supports `belongs_to`; explicit unambiguous legend model names support `represents_series`.
- Line models have **N data values but N−1 segment boxes**. The converter preserves `line_segment` boxes and paired source data endpoints; it does not assign one point value to a segment or invent point pixel boxes.
- Zero-area source line boxes are retained as source metadata with no trainable positive-area box. No guessed stroke padding is introduced. Out-of-image/malformed boxes are flagged and withheld; ambiguous array alignment is unsupported rather than truncated with `zip`.
- `h_bar` source `x_axis` denotes the categorical variable even though its labels appear vertically; `y_axis` denotes the numerical variable even though its labels appear horizontally. These semantic keys are retained. They are not blindly treated as physical Cartesian directions.

The format is `real-partial-1`, distinct from synthetic v2. It has no invented QA evidence, numerical calibration, point boxes, or complete graph edge supervision. The original authors warn of noisy/heuristic annotations. Passing bounds checks plus representative visual inspection does not certify all original labels; these partial conversions are not automatically mixed into neural training.

## VisText: explicit local mapping, raster mapping remains unavailable

Inspected the 20 acquired hierarchical scenegraphs, their images/linearized tables, and the pinned author's [`code/dataset_generation.ipynb`](https://github.com/mitvis/vistext/blob/24df7d32a39652c375b821f00f77845fb62d28ee/code/dataset_generation.ipynb) and [`code/utils_gen.py`](https://github.com/mitvis/vistext/blob/24df7d32a39652c375b821f00f77845fb62d28ee/code/utils_gen.py). Downloaded source was inspected as text, never executed.

The root contains nested group translations and local chart geometry. Example 1's plot frame is 200×200 scene units while its saved image is 616×588 pixels. Negative title/axis offsets occur outside the frame. Image dimensions alone do **not** determine export scale, padding, crop, or global translation. The supplied graph and inspected generation code do not provide a verified local-to-raster affine mapping.

`vistext_local_nodes` therefore accumulates **only explicit group x/y translations**. It preserves point anchors and explicit rectangle dimensions in chart-local coordinates. Text anchors are not font bounds; line/area anchors are not inferred paths; missing dimensions remain missing. Output `vistext-local-1` sets every raster `bbox` and the global `raster_transform` to null. Its plots are clearly labeled **local-coordinate previews**, with no real-image background implying false alignment.

### Dates and categorical conventions

The apparent prior-day/year offset is present in the source scenegraph mark `description` strings. The author utility `parse_marks_dt` copies values from these descriptions; it does not perform a recoverable timestamp-to-image conversion. Example 1 chart labels begin in 2006 while a source description says `Dec 31, 2005`. A time-zone/serialization explanation remains plausible but **unverified**. No one-day correction or UTC assumption is applied. Preserve display labels, descriptions, and ordering; do not train temporal calibration from these pairs yet.

The utility's scale-description heuristic infers numeric versus categorical behavior from displayed tick strings and spacing, and explicitly leaves temporal scales unresolved. Numeric-looking categories/years are therefore not automatically reinterpreted as continuous or temporal scales by our converter. Original strings/descriptions and raw hierarchy are preserved.

## Cross-format and readiness boundaries

Synthetic v2 supplies exact axes, point/rectangle geometry, typed nodes, semantic/numerical relations, QA operands, and image evidence. ChartQA partial mappings supply source pixels and some grouping relationships; VisText supplies nested local structure and descriptions. They do not have interchangeable label semantics or annotation coverage. Missing labels must remain masked, not treated as negative class examples.

The 20 ChartQA full-release images intentionally overlap their original GitHub counterparts. Existing Statista mappings also show that ChartQA/VisText may share upstream material; no whole-corpus source-disjointness claim is made. Pixel-hash checks cannot exclude rerendered/shared numerical sources. Milestone 3 can use validated synthetic supervision for controlled oracle-proposal graph experiments; real image-only or temporal/grounding claims require additional verified metadata and perception work.
