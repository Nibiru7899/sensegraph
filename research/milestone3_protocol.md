# Milestone 3 frozen experimental protocol

Protocol established before neural fitting and before any relation-model test evaluation. Configuration: `configs/relations.json`; expansion recipe: `configs/structural_v3.json`. The `freeze` command records configuration, implementation, and v2 split hashes. Changes to prediction/training/metric implementations require a new protocol version before test evaluation. No official benchmark test results are consulted.

## Scope and data

Only oracle-proposal structural relation prediction is measured. Proposals include renderer-defined series/axis aggregates; they are not detections. No real-image extraction, chart QA, accessibility, or biological/clinical outcome is tested. Historical files and the entire validated v2 dataset are hash-protected. No v1/v2 examples are merged into v3.

First execute v2 one-batch overfit, gradient checks, and 300-step matched small-dataset fits (seed 42). Treat their metrics as synthetic pipeline verification only. Once this passes, generate exactly 2,000 independent-seed charts in v3: 1,400/300/300, with disjoint resolved template bundles and numerical configurations. Repeat image/annotation/graph/QA/evidence/partition/leakage checks. Near-hash candidates are reviewed as potential overlaps, not automatically asserted to be duplicates. Hard cross-split identity/annotation violations block fitting. No automatic expansion beyond 2,000 occurs.

## Observable inputs and forbidden information

Each node has the existing 11 features: normalized box xyxy (4), center (2), size (2), and mean crop RGB (3). Models receive no node IDs, oracle types, numeric values, semantic adjacency, QA answers, evidence, template IDs, split IDs, or source seeds as features. Cache input and label dictionaries are separated and hash-bound to original split files. Node order is randomly permuted once per chart with a fixed label-independent seed derived from version and chart identity; the same permutation is used for every model. Pair candidate order stays directed and targets stay aligned.

The edge head sees projected source/target embeddings plus eight pair features: signed center displacement (2), absolute displacement (2), signed size differences (2), bounding-box IoU (1), RGB Euclidean distance (1). No label-dependent negative sampling is used. Every ordered distinct node pair is classified for all ten relations; self pairs are excluded. Labels are ten multi-hot indicators, with all entries valid under the closed synthetic ontology. All-zero vectors are valid negative pairs. The rare `equal_to` class is retained.

The GAT message graph is independent of targets: each node receives messages from its six nearest region centers within its chart, with self loops added inside GATv2. No edges cross chart boundaries. Candidate classification directions are never symmetrized. The graph-free model receives exactly the same raw node/pair information but performs no message passing. No per-chart calibration/axis matrix is supplied.

## Models and optimization

- Rule baseline: fixed geometry/color heuristics for containment, alignment, proximity, endpoint overlap, similar-color ordering, and approximate height comparisons. It cannot reliably infer all semantic classes and abstains for `represents_series`. It never reads node type/value/semantic labels. It has zero trainable parameters; only decision thresholds are calibrated on validation.
- MLP: 11→128→128 node projection, pair head with hidden widths 448 and 128, ten logits. ReLU/dropout 0.15; no graph propagation.
- GATv2: 11→128 projection; three four-head layers, 32 channels/head concatenated to 128, dropout 0.15, residual addition and LayerNorm; pair head widths 256 and 128. Parameter counts are recorded before fitting and should be within 25% for the main neural pair.
- Both: AdamW lr=0.001, weight decay=0.0001, gradient norm clip=1; float32, batch eight charts, all labels/pairs. Train-only positive weights are `clip(negative/positive,1,50)`; absent-positive classes receive weight 1 and remain reported. The cap prevents extreme rare-class gradients; no absent labels are fabricated.
- Identical step limits: v2 300, v3 600. Validation every 100 steps and at the final step. Select the earliest checkpoint attaining the highest validation macro AP; no test-based selection or hyperparameter search. Train/validation curves retain all successful evaluations and losses. One-batch verification uses 150 steps, dropout 0, lr=0.003; its scores are separate from full fits.
- Main v3 seeds: 42,43,44 for both neural models. Ablations use seed 42 and the same step schedule: no geometry, no color, complete connectivity. No alternative architecture is introduced.

No-geometry zeros all eight geometry input features and their derived pair features; it uses complete within-chart connectivity so neighbor topology cannot expose geometry. This necessarily changes both geometry information and the spatial connectivity prior; interpretation must acknowledge that coupling. No-color zeros RGB and RGB distance, keeping spatial connectivity. Complete-connectivity keeps all features and changes only the message graph. All ablation architectures retain the same feature dimensions/parameter counts as GATv2.

## Selection, calibration, test isolation, statistics

Primary metrics: macro and micro relation F1. Report every class's precision, recall, F1, AP, TP/FP/FN/TN, and positive/negative support. Macro F1 includes all ten classes with zero F1 for undefined/no-positive cases; macro AP averages classes with positive support and reports that denominator. AP groups equal scores, avoiding tie-order inflation. No overall edge accuracy is used as the headline.

After selecting each checkpoint, calibrate ten thresholds on validation only using the fixed grid in configuration. Maximize each class's F1; choose the highest threshold on ties. For no validation positives use 0.5. Save calibrated thresholds and model/config/split hashes before opening test tensors. After all planned fits/ablations are fixed (or explicitly time-blocked), seal the experiment inventory. Evaluate every successful selected model and the rule baseline on the untouched relation test splits once; test results do not cause retraining or threshold changes.

Paired uncertainty: bootstrap chart clusters, not individual correlated edges, 2,000 draws with fixed seed. Compare GAT−MLP macro/micro F1 on paired test chart identities; report each seed and the bootstrap interval of the seed-mean difference. Intervals condition on these fitted models and generated styles; three seeds and one held-out style bundle do not establish real-world generalization. Ablations have one seed and are descriptive.

## Verification and budgets

Unit tests cover directed labels/features, target mutation invariance, masked-feature invariance, graph-boundary connectivity, AP ties, threshold behavior, rare/zero supports, parameter counts, CPU gradients, and checkpoint round trips. Executed GPU one-batch checks must show finite loss/gradients and decreasing evaluation loss. Checkpoint reloads and repeated inference compare tensors, recording max difference and exact agreement. Seed Python/NumPy/Torch, set CUDA workspace configuration, disable TF32/cuDNN benchmarking, and request deterministic algorithms with warnings for unsupported kernels; record limitations instead of claiming universal bitwise deterministic training.

The legacy ledger consumed 1,038.673 of 3,600 seconds. A separate M3 ledger preserves that historical file, reserves bounded session time before GPU work, charges actual wall time including GPU loading/CPU overhead, and keeps interrupted reservations charged. Lifetime total must stay below 3,600 seconds. Stop before the cap; partial runs/failed checks are recorded and cannot silently enter matched comparisons. Downloads remain within the existing 10 GB aggregate cap; no new models/libraries are required.

Record commands, source/data hashes, versions/hardware, steps, seeds, parameter count, loss/gradient curves, runtime, allocated/reserved peak memory, checkpoint bytes, inference latency and device, failures, and all successful results. Existing baseline and Milestone 2 reports are not regenerated. Learning curves and final reports are exported as standalone figures/files.
