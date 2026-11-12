# Milestone 3 — controlled graph neural network comparison

Generated 2026-11-03T19:08:34+05:30 from executed, sealed local experiments. **Complete.** All results below concern **oracle-proposal synthetic structural relationship prediction**. They are not real-image extraction, chart QA, calibrated uncertainty, or accessibility results.

## Main result: untouched v3 test split

| Predictor | Macro F1, mean ± seed SD | Micro F1, mean ± seed SD | Macro AP, mean | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Rule (CPU) | 0.4472 | 0.6602 | 0.3342 | 0 |
| MLP (3 seeds) | 0.7409 ± 0.0054 | 0.8746 ± 0.0056 | 0.7251 | 195,530 |
| GAT (3 seeds) | 0.7629 ± 0.0116 | 0.8797 ± 0.0113 | 0.7705 | 204,170 |

The main three-layer GATv2 has a modest positive seed-mean advantage over the graph-free MLP under this protocol. This supports further investigation of contextual structure with these oracle proposals and engineered geometry/color inputs. It does not establish better real-world chart understanding.

- macro_f1: GAT−MLP **+2.202 percentage points**, conditional chart-bootstrap 95% interval **[+1.890, +2.500]**.
- micro_f1: GAT−MLP **+0.513 percentage points**, conditional chart-bootstrap 95% interval **[+0.276, +0.759]**.

Intervals resample 300 paired chart clusters, 2,000 draws, seed 20261103, and average paired differences over three fitted seeds. They condition on the fitted models, generator and one held-out style bundle: they **do not** include full training-population or unseen-domain uncertainty. Three seeds are not a large seed study. Individual seeds matter:

| Seed | MLP macro | GAT macro | Macro difference | MLP micro | GAT micro |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7415 | 0.7499 | +0.0084 | 0.8736 | 0.8720 |
| 43 | 0.7353 | 0.7720 | +0.0367 | 0.8696 | 0.8927 |
| 44 | 0.7459 | 0.7669 | +0.0210 | 0.8806 | 0.8745 |

## Protocol and controls

The experiment protocol/configuration and implementation hashes were frozen before fitting. A documented amendment **before any test evaluation** added orchestration guards requiring a complete inventory and forbidding post-seal fitting/calibration, and clarified identity-derived, label-independent node permutation seeds. Model, training, metric, hyperparameter, seed, threshold-grid and bootstrap implementations/configurations did not change. Initial and amended snapshots are preserved under `experiments/milestone3/source_snapshots` and `protocol_history`; amendment notes are saved separately.

Inputs: the same 11 geometry/crop-RGB features and eight directed pair features for both models. GAT adds six-neighbor, geometry-derived within-chart connectivity, never target semantic adjacency. Oracle node types, numerical values, true relations, answers, evidence, IDs and split metadata are absent from forward inputs. A fixed random permutation removes canonical annotation ordering; targets remain candidate-aligned. Target/metadata mutation and masked-feature invariance tests pass. Candidate pairs remain directed, and graph boundaries are checked.

Both main models use AdamW lr 0.001, weight decay 0.0001, batch 8, maximum 600 steps, gradient clip 1.0, dropout 0.15, and capped train-only BCE positive weights. GAT uses three 128-channel, four-head residual/LayerNorm layers. Main parameter counts differ by 4.42%. The MLP has no message passing. Validation macro AP selects checkpoints; per-class thresholds are calibrated on validation only and saved/hash-sealed before test access. No test-based refits or threshold adjustments occurred. Every planned successful experiment is reported; no overall edge accuracy is used as the headline.

## Per-class test performance and support

F1 averages three seeds; supports are the same ground truth for all predictors. All ten classes are retained. See `per_class.csv` for every run's precision/recall/F1/AP and TP/FP/FN/TN, and `error_examples.csv` plus raw scored NPZ files for actual errors.

| Relation | Positive pairs | Negative pairs | MLP mean F1 | GAT mean F1 | Rule F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| belongs_to | 7,662 | 642,778 | 0.6506 | 0.7128 | 0.2643 |
| maps_to_axis | 6,682 | 643,758 | 0.9967 | 0.9969 | 0.0000 |
| maps_to_tick | 3,341 | 647,099 | 0.9487 | 0.9101 | 0.5766 |
| represents_series | 599 | 649,841 | 0.5985 | 0.6076 | 0.0000 |
| labels | 3,766 | 646,674 | 0.9402 | 0.9861 | 0.7114 |
| connected_to | 2,310 | 648,130 | 0.5974 | 0.6477 | 0.1747 |
| next_in_series | 2,742 | 647,698 | 0.6827 | 0.7781 | 0.6204 |
| less_than | 21,863 | 628,577 | 0.9519 | 0.9549 | 0.9090 |
| equal_to | 276 | 650,164 | 0.0837 | 0.0818 | 0.3063 |
| greater_than | 21,863 | 628,577 | 0.9587 | 0.9533 | 0.9090 |

`equal_to` remains unreliable: GAT mean F1 **0.0818**, MLP **0.0837**. The graph model does not uniformly improve every relation. Geometry/color cannot always identify semantic roles, and equality is rare; high negative accuracy would conceal its false positives/misses. Capped positive weighting and validation thresholds mitigate imbalance but do not solve it. The fixed rule baseline abstains on `represents_series` and uses only approximate geometry/color heuristics; it is useful but not an exhaustive hand-engineered chart parser.

## Preplanned ablations

All ablations use seed 42, the same 600-step maximum and checkpoint/threshold rules. They are single-seed descriptions, not replicated significance claims.

| GAT variant | Macro F1 | Micro F1 | Macro AP |
| --- | ---: | ---: | ---: |
| main | 0.7499 | 0.8720 | 0.7718 |
| no_geometry | 0.2641 | 0.3608 | 0.1936 |
| no_color | 0.7305 | 0.8586 | 0.7271 |
| complete_connectivity | 0.7915 | 0.9173 | 0.7877 |

Removing explicit geometry hurts substantially. Its connectivity is also changed to complete so a spatial graph cannot leak coordinates; this coupled change prevents attributing the entire difference to feature masking alone. RGB crop means may still proxy region appearance/size. Removing color changes neither geometry nor spatial connectivity. Complete connectivity outperforms spatial-six-neighbor GAT for this seed, consistent with useful longer-range context, but that interpretation requires replicated follow-up. The winning ablation was not retroactively promoted to the main model after test evaluation.

## Pipeline verification versus larger development experiment

The preserved v2 corpus is **100 charts, 70/15/15** and was used first for one-batch and full small-data wiring checks. Both neural models completed 150 one-batch steps and 300 small-data steps. Their small-data test metrics are in the all-run table and are not real-world evidence.

- gat: initial fixed-batch BCE 0.968511; selected-checkpoint fixed-batch BCE 0.023173; last optimizer-step BCE 0.053961.
- mlp: initial fixed-batch BCE 0.967808; selected-checkpoint fixed-batch BCE 0.047289; last optimizer-step BCE 0.047695.

All neural gradient norms/losses were finite; checkpoint round trips and repeated evaluation were exactly equal in the recorded checks. Seeds/deterministic modes and CUDA workspace settings were used. Universal bitwise deterministic training was not established across devices or by full same-seed retraining; unsupported-kernel warnings are allowed and documented. A printed-loss scalar warning in logs did not affect gradients or results.

Only after v2 verification passed was **v3 generated: 2,000 charts, 1,400/300/300**. All image identities, annotation hashes, boxes, transforms, numerical values/relations, QA answers/evidence, and split invariants passed. No exact images or cross-split template/data identities collided; no v2 numerical arrays were reused. There were **45 near-hash candidates, 6 cross-split**. All candidate data hashes differ; all six cross-split pairs were visually reviewed as distinct values/layouts, and flags remain recorded. No automatic near-hash deduplication or false whole-corpus independence claim is made. Held-out font/palette/layout bundles differ from training. No v1/v2 drafts were merged into v3, and no further expansion occurred. Annotation test-label inspection was for integrity, never relation-model performance tuning.

## All successful experiments, resources and latency

| Run | Test macro F1 | Test micro F1 | Test macro AP | Selected step | Fit/calibration seconds | Peak train allocated MiB | Cached-graph median ms/device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2_gat_main_s42 | 0.7259 | 0.8425 | 0.7321 | 300 | 7.54 | 202.3 | 15.368 (cuda) |
| v2_mlp_main_s42 | 0.6316 | 0.7518 | 0.6791 | 300 | 3.64 | 237.7 | 2.902 (cuda) |
| v2_rule | 0.4901 | 0.6728 | 0.3813 | 0 | 0.06 | 0.0 | 1.602 (cpu) |
| v3_gat_complete_connectivity_s42 | 0.7915 | 0.9173 | 0.7877 | 600 | 33.04 | 571.5 | 9.309 (cuda) |
| v3_gat_main_s42 | 0.7499 | 0.8720 | 0.7718 | 600 | 21.80 | 262.9 | 7.032 (cuda) |
| v3_gat_main_s43 | 0.7720 | 0.8927 | 0.7626 | 600 | 23.71 | 261.1 | 9.495 (cuda) |
| v3_gat_main_s44 | 0.7669 | 0.8745 | 0.7770 | 600 | 21.74 | 245.3 | 6.955 (cuda) |
| v3_gat_no_color_s42 | 0.7305 | 0.8586 | 0.7271 | 500 | 20.97 | 262.9 | 7.239 (cuda) |
| v3_gat_no_geometry_s42 | 0.2641 | 0.3608 | 0.1936 | 500 | 33.09 | 571.5 | 7.475 (cuda) |
| v3_mlp_main_s42 | 0.7415 | 0.8736 | 0.7222 | 600 | 14.33 | 323.7 | 1.372 (cuda) |
| v3_mlp_main_s43 | 0.7353 | 0.8696 | 0.7114 | 600 | 13.45 | 320.8 | 1.640 (cuda) |
| v3_mlp_main_s44 | 0.7459 | 0.8806 | 0.7416 | 600 | 14.98 | 299.1 | 3.377 (cuda) |
| v3_rule | 0.4472 | 0.6602 | 0.3342 | 0 | 1.29 | 0.0 | 1.838 (cpu) |

Rule time is CPU validation/calibration, not optimization. Neural time includes GPU loading, fitting and validation scoring; feature-cache preparation is outside these timers and was not timed separately. Latency starts from cached observable tensors, includes batching/device transfer/scoring, and excludes image loading/feature extraction, output serialization, and metric computation. The evaluator also moves labels for scoring, but no targets enter forward. CPU rule and CUDA neural latency are not directly comparable speed claims. Allocated/reserved memory and checkpoint byte sizes are exported per run in `comparison.csv`; these are PyTorch measurements, not total device memory.

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU; PyTorch 2.8.0+cu128, CUDA 12.8; float32 local training. New charged model-session wall time: **253.903 s (4.23 min)**. With the preserved historical ledger, lifetime charged time is **21.54/60 min**. All reservations have actual completion times. `gpu_resource.csv`/`gpu_budget.json` include full session accounting, also charging test scoring/serialization overhead conservatively. No new downloads, paid APIs, external models, or frontend were used.

Full suite: **78 passed**. **637 historical/v2 files unchanged**, including prior baselines, reports, v2 data, old implementation and tests. No old report-generation script was run. The new report does not overwrite previous milestone reports.

## Artifacts and reproduction

- Frozen configs: `configs/relations.json`, `configs/structural_v3.json`; protocol: `research/milestone3_protocol.md`.
- Model/training/metric/CLI modules: `src/sensegraph/relation_*.py`.
- Run logs/checkpoints/validation thresholds/raw test scores/errors: `experiments/milestone3/runs/`.
- CSVs: `comparison.csv`, `per_class.csv`, `training_log.csv`, `error_examples.csv`, `ablations.csv`, `gpu_resource.csv` under `experiments/milestone3/`.
- Standalone figures: `experiments/milestone3/figures/learning_curves.png`, `ablations.png`.
- Visual review: `experiments/milestone3/visual_review.json`; sampled bar/grouped-bar/line overlays align with rendered marks and ticks. This sample review does not replace full automated annotation checks.
- Summary: `research/milestone3_results.json`; complete per-class measurements remain in each `test.json`.

The executed command sequence was `python -m sensegraph.relation_experiment freeze`, `verify-v2`, `expand`, `fit-v3`, final `freeze` after the documented guard amendment, `seal`, and `test`, using the local `.venv/Scripts/python.exe`. Commands and stdout logs are retained under `experiments/milestone3`. Re-running fitting in this sealed namespace is intentionally rejected; create a new protocol namespace for a new scientific study rather than resetting resource ledgers or modifying historical results. `load_checkpoint` plus `infer` expose cached-graph checkpoint inference; test files are reused on report regeneration, not reselected.

## Conclusion and stop boundary

There is a measurable but modest **synthetic oracle-proposal** graph benefit for the predefined main comparison, alongside significant relation-specific weaknesses and seed variation. The 2,000 charts come from a limited generator and one held-out test-style bundle; they do not demonstrate robustness to real-image proposals or learned perception errors. Real datasets' partial annotations, VisText transforms/dates, full QA, evidence reliability, uncertainty calibration and BLV usability remain unresolved/out of scope.

Stop here. The next milestone should investigate frozen learned visual features and imperfect region proposals with new controlled protocols and untouched evaluation data. No claim of improved UniChart/DePlot QA accuracy or accessibility follows from this relation experiment.
