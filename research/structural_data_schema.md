# Milestone 2 structural data contract

## Renderer-derived records (schema version 2)

Records live in `data/synthetic/v2/annotations/` and split JSONL files. `manifest.json` records the full configuration, its SHA256, library versions, image hashes, and annotation hashes. No baseline schema-v1 file is changed. The superseded synthetic `v1` draft is retained but excluded from training: its deterministic cycling coupled chart type and image resolution. Do not concatenate versions of the same source identities.

| Field | Contract |
| --- | --- |
| `id`, `source`, `split`, `chart_type` | Stable synthetic identity; original synthetic train/val/test membership; bar, grouped_bar, or line |
| `spec` | Seed, font/style/layout template, resolution, DPI, exact categorical labels, series names, integer values, template/data fingerprints |
| `image_path`, `image_sha256`, `width`, `height` | Relative root path, identity, exact saved PNG size |
| `font` | Resolved font family and local font-file SHA256 |
| `axes` | Categorical x labels/indices; linear y limits/ticks; plot extent; 3x3 data-to-final-image affine matrix |
| `nodes` | Stable ID, type, continuous pixel `bbox`; typed attributes such as text, series/category indices, numerical value, data/pixel anchor |
| `edges` | Directed source/target/relation triples; all endpoints must exist |
| `numerical_relationships` | Complete ordered pairwise comparisons between data marks, including equal values |
| `questions` | Operation, natural-language question, explicit operand IDs, numeric/boolean answer, supporting node IDs and exactly corresponding pixel boxes |

Coordinates are **continuous `[x0, y0, x1, y1]`**, origin at the final PNG's upper-left. Matplotlib display coordinates are flipped vertically using the actual canvas height. There is no tight crop, image resize, or guessed OCR geometry. Tick display values equal their exact calibration values. Negative values are allowed. Bars have an explicit 0.8-point outline, so a zero-height bar is visibly rendered as a horizontal stroke and has a nonzero stroke-support box.

Text boxes come from the final Agg renderer's font/layout metadata. Data mark and segment boxes include their explicit stroke/marker extents. Axis-aligned segment boxes enclose diagonal segments; they are not segmentation masks. Series boxes are unions of renderer-derived member boxes, not extra painted objects. Backgrounds, decorative gridlines, and legend swatches are not separately labeled. Annotation coverage refers to this declared ontology, not every painted pixel.

Node types: axis, tick, tick_label, title, axis_label, legend_item, series, bar, point, line_segment. Relation vocabulary/order is exported as `sensegraph.synthetic.RELATIONS`: belongs_to, maps_to_axis, maps_to_tick, represents_series, labels, connected_to, next_in_series, less_than, equal_to, greater_than. Several relations can hold for one ordered pair; labels are **multi-hot**, not a single softmax class. An all-zero edge label means no annotated relation in this closed synthetic ontology.

QA operations: lookup, maximum, minimum, absolute difference, greater-than comparison, counting Series 1 observations, and signed difference of Series 1/2 sums. Extrema compare all data marks, handle ties without arbitrarily selecting one winner, and return a numerical value. Comparison answers are boolean; other answers are numerical. Evidence includes operands, relevant category/legend labels, and axis calibration context. This is a declared sufficient-evidence policy, not human-validated minimal evidence.

## Split policy

The checked-in configuration creates 70/15/15 charts. Each split has all three chart types. A template family is a **resolved font/palette/background/layout/rotation/grid bundle**, not just an arbitrary ID. Train uses two families; validation and test each use a distinct held-out bundle. Template fingerprints exclude IDs, preventing renaming from disguising the same family. Chart rendering primitives are intentionally shared across splits; this is not a claim of unseen chart-type generalization.

RNG seeds depend on root seed, split, and sample index. Entire numerical/category/series configurations are fingerprinted and checked across splits. The current 100 samples have disjoint exact configurations; accidental duplicates fail validation. New renderings of an existing source configuration must stay with that source's split. Visual exact hashes and 64-bit grayscale difference hashes identify potential duplicates; the latter is only a candidate heuristic. No official benchmark test results select or tune templates. Baseline validation records are read only for integrity/scoring/overlap checks.

## PyTorch / PyG interface

`StructuralDataset(manifest, root)` returns image pixels, a `ChartGraph`, question text, and metadata. The dataset uses **renderer ground-truth region proposals for supervised/oracle experiments**. This is not a working region detector or an image-only inference claim.

`collate_charts` returns:

- `inputs.images`: padded RGB float32 `[B,3,Hmax,Wmax]`, with `image_mask` and original `[H,W]` sizes; no geometric scaling.
- `inputs.graph`: PyG Batch containing only 11 observable node features (normalized box coordinates, center, size, mean crop RGB) and all ordered non-self edge candidates. Candidate generation does not inspect ground-truth relations.
- `inputs.questions`: question text only. Stable IDs/splits/proposal provenance remain metadata.
- `targets.node_type_y`, `value_y`, `value_mask`, `edge_y`: node-type supervision, scalar numerical labels with explicit missingness masks, and candidate-aligned relation labels.
- `targets.question_op_y`, `answer_y`: operation classification and numeric/boolean answers.
- `targets.evidence_index`: `[2,K]` question-to-node positive links. A custom PyG increment offsets question and node spaces independently. Within each question, unlinked nodes are negative evidence labels under the declared policy.

PyG targets and inputs are separate objects. Do not use target relations as message-passing adjacency, target node types as observed types, target values as features, or target answers/evidence during inference. `observable_graph(image, proposed_boxes)` accepts detector-supplied proposals without annotations. Missing proposals, OCR/text features, cached learned embeddings, and calibration estimation remain perception work for later milestones. The v2 synthetic dataset deliberately does not accept the partial real formats silently.

All-pairs candidates cost O(N²). The 100-chart set has small graphs; scaling images and graph sizes requires memory profiling and candidate-pruning validation before larger training. No GATv2 model or optimization loop is implemented in this milestone.
