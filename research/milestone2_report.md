# Milestone 2 — validated structural training data

Generated 2026-11-03T18:25:30+05:30. **Complete for the bounded 100-chart milestone.** Ready for controlled oracle-proposal GATv2 versus graph-free experiments, with the limitations below. No neural model was trained and no frontend was built.

## Executed dataset validation

- Charts: **100** in `data/synthetic/v2` — 34 bar, 33 grouped bar, 33 line; split **70 training / 15 validation / 15 test**. Resolution and DPI are sampled independently of chart type. The superseded `v1` draft is preserved, excluded from training, and must not be concatenated with v2 because it repeats source identities.
- Nodes: **4040**, 26–66 per chart (mean 40.4).
- QA pairs: **666**, with validated operands, numerical/boolean answers, supporting node references, and final-image evidence boxes for every pair.
- Invalid samples: **0**. All image identities/dimensions, affine forward/inverse mappings, source data values, rendered mark/segment extents, ticks, typed node IDs, edge references/semantics, comparisons, answers, and evidence references/coordinates passed automated checks.
- Complete test suite: **67 passed**; JUnit: `experiments/milestone2/tests.xml`. This includes all 44 original baseline tests. A third-party PyG distributed-module deprecation warning was observed; distributed training is not used.
- All 100 charts loaded via PyTorch DataLoader/PyG collation: `{'train': {'loaded_charts': 70, 'batches': 18}, 'val': {'loaded_charts': 15, 'batches': 4}, 'test': {'loaded_charts': 15, 'batches': 4}}`. CPU only; the validation CLI constrains threads to avoid small-operation oversubscription.

| Relation | Positive directed labels |
| --- | ---: |
| labels | 1163 |
| belongs_to | 2183 |
| maps_to_axis | 1822 |
| maps_to_tick | 911 |
| next_in_series | 714 |
| represents_series | 197 |
| greater_than | 4635 |
| less_than | 4635 |
| connected_to | 618 |
| equal_to | 82 |

Full node/type, operation, coverage, split-loading, and candidate statistics are in `research/milestone2_statistics.json`.

## Visual and tensor verification

Reviewed mark overlays for all three chart types in every split: `synthetic-00000, synthetic-00001, synthetic-00002, synthetic-00072, synthetic-00070, synthetic-00071, synthetic-00087, synthetic-00085, synthetic-00086`. Rectangles follow bar boundaries, including negative values and visibly stroked zero-height bars. Point boxes follow markers; segment boxes enclose the corresponding line segments. Reviewed full-resolution context and lookup-evidence overlays: tick text, axis labels, title, legends, operand regions, and calibration context align with the saved canvases. Contact sheet: `experiments/milestone2/synthetic_overlay_contact_sheet.png`; individual mark/context/evidence layers are under `experiments/milestone2/overlays/`.

The bounded end-to-end example traversed synthetic PNG → annotation JSON → graph → dataset → three-chart batch. Images: `[3, 3, 480, 640]`; observable node features: `[106, 11]`; candidate edges: `[2, 3712]`; multi-hot edge targets: `[3712, 10]`; evidence links: `[2, 445]`. Finite-feature and graph-boundary checks passed. Variable-sized images are padded without scaling, and bipartite evidence links are independently offset by question/node counts.

Inputs contain pixels, region geometry/color features, text questions, and complete directed edge candidates. True types, numerical values, semantic edges, answers, and evidence stay in supervision. **Region proposals themselves are renderer ground truth in this dataset.** A real detector/OCR pipeline must supply proposals at inference; this is not proof of image-only chart understanding. Feature caching and learned visual backbones are not implemented.

## Real-source mapping results

- ChartQA full release: **20/20** existing training identities converted, **636** positive-area source pixel boxes; **20/20** paired images match the immutable baseline source images byte-for-byte. Source revision, member CRC/hashes, and split provenance are recorded separately. No official test payloads or whole ZIP were acquired.
- Source issues: `{'zero-area source box retained without invented padding': 43}`. Zero-area segment boxes remain unavailable as positive-area boxes; no padding or point annotations were invented. No out-of-image boxes were silently clipped.
- Real bar/text and line-segment overlays were visually inspected on the paired images. The separate converter retains line segments (N−1 boxes) rather than mislabeling them as N points. Partial real grouping edges are source-supported; QA evidence and calibration are not fabricated.
- VisText: **20/20** acquired graphs converted only into explicit local translations/anchors/rectangles. **Zero raster boxes asserted.** Plot frame and image dimensions do not supply a verified scale/padding/translation. Local previews are explicitly labeled and are not real-image overlays.
- The date discrepancy is already present in source mark descriptions; the author parser copies it. No unsupported one-day/time-zone correction was applied. See `research/real_annotation_mapping.md` for categorical, temporal, unsupported-transform, and ontology differences.

## Partitioning, leakage, and baseline preservation

Template families are resolved style/layout bundles with disjoint hashes between splits. Entire numerical arrays are also compared independently of series/category naming. The generated data has **0 cross-split identity violations, 0 exact image duplicates, and 0 near-duplicate candidates** at a predeclared 64-bit difference-hash threshold of 4. Hash candidates against acquired ChartQA/VisText images: **0**. These are sample-level checks; they do not prove whole-corpus semantic/source disjointness. The full ChartQA images intentionally repeat the same 20 original source identities and are not new independent examples.

Integrity check: **330 protected baseline files unchanged**, including pilot selection, original data/manifests, historical reports/lock, predictions, metrics, and baseline implementation/tests. Recomputed relaxed correctness remains UniChart **65/100** and DePlot+executor **12/100**. Only authorized download-budget bookkeeping is excluded from the immutable-file check. The original report-generation script was not used to rewrite historical baseline reports.

Current aggregate accounted downloads: **5.848 GB**, below the existing 10 GB cap. No additional GPU model sessions or training were run.

## Milestone 3 readiness and boundaries

The data/graph/batching path is ready for a **small, controlled synthetic oracle-proposal experiment**, comparing GATv2 and a graph-free model on identical observable inputs, splits, and budgets. Relation targets are multi-label; never substitute target adjacency/types/values into inference inputs. The 100-chart corpus is a correctness fixture, not enough for claims of improved generalization. Scaling toward 10,000–20,000 requires a new versioned dataset configuration and repeated integrity/leakage/annotation checks, not overwriting this one.

Real raster annotation coverage remains partial: ChartQA's original labels are noisy, zero-area line boxes are withheld, and VisText raster transforms/dates are unresolved. Real-world perception, learned features, calibration, evidence quality, multi-seed results, and BLV accessibility remain unvalidated. No graph accuracy, training improvement, robustness gain, or accessibility result is claimed. Stop after this milestone.

Commands and exact data contracts are in `README.md` and `research/structural_data_schema.md`. The structural dependency lock is separate from the preserved baseline lock.
