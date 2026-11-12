# Full visual perception training and autonomous quality audit

Snapshot: 2026-11-04 05:53:45 +0530. This report supersedes the prior live snapshot; the original v2 report is preserved at `experiments/visual_training/v2/report_snapshot.md`. **No guarantee of perfection, absence of all leakage, or zero overfitting is possible.** Actual findings and limitations follow.

**Readiness: not ready for reliable real-chart QA.** Synthetic detection is learned, but only 77/616 supplied valid real-image boxes were recovered on 20 training-source images. OCR detected-region word F1 is 0.3078. These are observed quality blockers, not successful completion of chart understanding. No learned QA/calibration/new multimodal graph training or frontend has been added.

## Completed and ongoing training

| Visual run | Status | Complete epochs | Selected stage/epoch | Validation AP50 | AP75 | Peak allocated / reserved GiB |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| v2 | early_stopped | 18 | perception 8 | 0.8984910958984964 | 0.5288890511195182 | 0.539 / 0.635 |
| v3 | early_stopped | 17 | perception 8 | 0.9061730448662972 | 0.5323019021095803 | 0.526 / 0.635 |

Both visual runs use all1,400 training and300 validation charts, fresh required ImageNet1K_V1 ResNet18 plus128-channel FPN and the exact six perception heads. FP32, segmentation disabled, five warmup/up to30 C4/C5 epochs, frozen BN, AdamW3e-4/1e-5, weight decay1e-4, accumulation8, clip1, fixed LR schedule and validation-only selection. Early stopping uses5 epochs without meaningful AP50 improvement after minimum5 fine-tune epochs. No GAT/reasoner/frontend training and no old protected-test model evaluation. OCR is a separate independently supervised stage described below. See frozen protocols for exact settings.

v2 genuinely completed five warmup+13 fine-tuning epochs and stopped at3,150 optimizer updates. Its selected checkpoint is fine-tuning8, not the final overfit model. Training loss continued falling while validation AP fell; early stopping mitigates further fitting but **does not establish zero residual overfitting**. A normal planning-window pause resumed from step2,879/cursor632 and reached the exact18-epoch update count without replaying committed batches. Source snapshots, original/continuation logs and separate actual reservations are preserved.

v3 is a single-factor implementation correction, not an architecture search: max-union Gaussian halos over colliding instances. Original target halos were annotation-order dependent on8/1,700 development charts. Corrected fixed-order targets differ only on2 train and1 validation charts. All positive centers, boxes, counts and regression/valid masks remain identical; all1,700 records pass reversed-order invariance. Two runs with one seed cannot establish a robust statistical improvement. Select highest complete validation AP50; exact cross-run ties prefer corrected v3. No holdout-based selection.

## Independent audit-only evaluation

Selection locked before inference: **corrected_v3**. Complete three-model audit: **True**.

| Model | Fresh holdout AP50 | AP75 | Recall50 |
| --- | ---: | ---: | ---: |
| corrected_v3 | 0.8748 | 0.4838 | 0.8409 |
| original_v2 | 0.8533 | 0.4605 | 0.8261 |
| mobilenet | 0.1159 | 0.0175 | 0.1769 |

The selected model is NOT changed if another model scores higher here. These are new synthetic fonts/styles/configurations, not real-world data. The old detector retains its original320 input, min score .01 and cap100; new models use640, min .10 and cap128. All use identical physical GT/classes and .25 operating threshold. Training exposure differs substantially; this comparison does not isolate architecture. Keypoint5px metrics use new heatmap centers versus old box centers and are reported separately in each metric JSON.

Selected checkpoint's full1,400-chart training AP50 is0.9607, measured only as an overfitting diagnostic after selection. Compare with selected validation and audit AP, not as independent quality evidence. No updates or selection changes follow it.

Precision disclosure: training/epoch validation disabled TF32 and enabled deterministic algorithms with warn-only fallback. The one-time independent evaluator left PyTorch2.8 defaults: float32 tensors/no autocast, cuDNN TF32 allowed, matmul TF32 disabled, deterministic-algorithm enforcement off. All three audit models share that setting. CPU import-equivalent flags/source hash are recorded in `precision_configuration_audit.json`; individual GPU kernels were not traced. These are **not a strict-TF32-disabled rerun**. Preserve these first results; do not rerun/reselect on the holdout. Post-selection validation-only checks quantify arithmetic repeatability separately.

The new audit dataset has300 charts,100 each bar/grouped/line,12 type/style cells of25 charts. New Arial, Times New Roman and Calibri style bundles; deterministic seed61720261104. Config/script/generator hashes frozen before inference. New numerical arrays are disjoint from old versions,300 new actual image hashes checked, no exact image or template identity overlap. Six near-hash pairs were visually inspected as distinct configurations and retained; no accuracy-conditioned sample removal. The new files never enter training readers. Holdout selection/model hash is locked before first inference; no result may trigger tuning, retraining, a threshold change or changing the winner.

## Data-leakage and provenance audit

Current audit: **0 actual image-hash mismatches,0 split identity violations,0 exact image duplicates,0 reused v2 numerical arrays**. Original45 near-hash candidates remain flags; they are not asserted independent by a weak hash. Static inspection confirms only train/val readers and image-only encoder calls. Test manifests were read only for provenance/configuration fingerprints, not model inference/tuning. Raw pixels and image-derived resize transforms enter detection; renderer boxes/types/values are used for supervision/scoring only. No oracle proposals in inference. CPU image-only inference/reload and trained-source checks passed.

Validation is reused for development/selection and is **not an untouched final estimate**. Historical ChartQA/VisText shared-source overlap and partial real labels remain documented; they are not mixed into this synthetic training. Pretrained membership cannot be fully audited locally. No universal independence or 'no leakage anywhere' claim is made beyond the explicit checks. All historical protocols/ledgers/data remain hash-protected; v3 additionally protects the complete v2 artifacts.

Known synthetic shortcut risk: rendered titles explicitly say `Synthetic bar/grouped_bar/line`, and category/series/axis text uses a small repeated vocabulary. These are visible pixels rather than hidden source fields, but they can reveal plot family and simplify recognition. Split-identity checks do **not** establish shortcut-free learning. Do not alter frozen data now; a future separately preregistered generator study should vary natural titles/text and test sensitivity before broader claims. No empirical title-ablation result is claimed here.

The v2 checkpoint audit confirms50 early-backbone tensors unchanged from ImageNet,40 BN statistic tensors unchanged and87 unused graph/QA/OCR/segmentation tensors unchanged. CPU checkpoint reload and decoded predictions were exact. New inference defaults to CPU; all agent-run GPU evaluation/training is ledgered. Segmentation requires real stroke masks and remains untrained.

## Error analysis and limitations

Inspect `release/training_comparison.png`, `release/campaign_metrics.csv`, per-epoch metrics/raw predictions/native overlays, collision audit/examples, and `v2/best_error_analysis.json`. Tiny native area<1024 and collision-target recall are diagnostic subgroups, not alternative denominators hiding false positives. Confidence histograms are detection-score distributions, not calibrated probabilities. AP75 and thin-axis/segment failures remain central: AP50 alone does not establish precise boxes suitable for numerical calibration. Collision ambiguity is retained in evaluation under the user-approved5% exception; invisible/coincident instances cannot all be recovered from one center slot.

Chart-type AP averages defined classes; inherited macro recall averages all seven including absent types. Compare supported per-class recall across chart types, not raw macro ceilings. Synthetic held-out styles do not establish real ChartQA perception, full QA, robustness, uncertainty calibration, or accessibility. Do not retrain on the new audit dataset after seeing its scores.

Detection AP here is the repository's all-points precision-envelope integral with one-to-one typed matching, averaged over supported classes at IoU .50/.75, **not COCO101-point AP or COCO .50:.95 mAP**. Recall/precision use fixed score .25; AP uses all returned predictions after fixed decoder thresholds/caps. Native5-pixel keypoint metrics are separate from box IoU. Keep these definitions when reproducing comparisons.

## Additional bounded perception and transfer checks

OCR: early_stopped; 13 complete epochs, selected epoch8 by validation CER0.1703, exact-word0.4786. All327,488 OCR parameters train independently; no vision/graph/QA updates. Cache contains21,655 training/4,682 validation crops, no infeasible targets or exact cross-split pixel duplicates.109 of110 validation strings occur in training: this is closed synthetic vocabulary recognition, **not unseen-word OCR**. Geometry-only90-degree rotation does not guarantee arbitrary slanted-text recognition. CUDA explicitly warned that CTC backward lacks a deterministic implementation under warn-only policy. Seeds/checkpoints are recorded; bit-for-bit training replay is not guaranteed. Selected validation title exact accuracy and geometry-rotated-label exact accuracy are both0, so OCR is an experimental branch, not reliable general text extraction.

Selected OCR training-crop CER0.0014 versus validation CER0.1703; training recognition is a memorization/overfitting diagnostic, not generalization evidence. Holdout vocabulary overlap: {'unique_holdout_strings': 108, 'seen_unique_strings': 108, 'unseen_crops': 0, 'scope': 'Scoring only; vocabulary is never used for decoding/correction'}; no lexicon is used for decoding. An unseen-transcription subset with zero samples is unavailable evidence, not zero-error generalization.

Independent OCR: oracle-GT-crop CER0.2311, exact0.3761; image-only detected-region exact-word precision0.3260, recall0.2915, F10.3078, end-to-end CER0.4430. All4642 source text/legend regions remain in the denominator. Missing regions are deletions, extra text is insertion; matched-only CER is a separate diagnostic. No GT strings/crops/types enter recognition. Numeric, rotated and role subsets plus raw cases are in `ocr_holdout/`. No test-driven retraining or lexicon corrections. `selected_with_ocr.pt` changes only OCR tensors of the selected vision model.

Frozen graph audit complete: **True**. Any failed charts remain FN in denominators; incomplete audits do not receive paired comparative claims.

Frozen legacy graph transfer (NO graph retraining):

| Model | Predicted proposals macro F1, mean3 seeds | Oracle proposals diagnostic macro F1, mean3 seeds |
| --- | ---: | ---: |
| gat | 0.5789 | 0.7587 |
| mlp | 0.5451 | 0.7261 |

Only frozen oracle-trained11-feature geometry/RGB models and original validation thresholds are transferred. Conditional matched-node scores and paired chart-bootstrap intervals are separate in `graph_transfer/results.json`. Missing virtual series endpoints remain end-to-end FN. This does not train the proposed multimodal graph stack or establish a matched-learning/real-world GAT advantage. Paired macro difference0.0338, chart-bootstrap95% interval[0.029948737796133294, 0.03788932737433655]; conditional on these three frozen seeds, detector and synthetic generator, not new training-seed/real-domain uncertainty. GAT numeric mutual-exclusion violations by seed: [2998, 2305, 5153] across495,014 directed candidate pairs per seed; these raw predictions are unchanged. Learned comparisons are not verified arithmetic.

Twenty existing ChartQA **training** images, supported partial-label recall only:

| Provided class | Recovered / supported boxes | Recall50 |
| --- | ---: | ---: |
| text | 68 / 262 | 0.2595 |
| bar | 8 / 52 | 0.1538 |
| line_segment | 1 / 277 | 0.0036 |
| legend_item | 0 / 25 | 0.0000 |

These twenty training examples are not an independent benchmark test. Unknown labels cannot define false positives, full precision/AP or complete recall; none is claimed. Horizontal bars/real typography are outside synthetic support. VisText raster alignment stays unavailable; no annotations were invented. No real-image adaptation or threshold/model tuning occurred.

Post-selection GPU verification (TRAIN/VAL only):

| Arithmetic | Full validation AP50 | One TRAIN image repeat exact | Reload exact |
| --- | ---: | --- | --- |
| audit_defaults | 0.906070 | True | True |
| training_fp32 | 0.906173 | True | True |

Original strict selected-epoch validation AP50 exactly reproduced: **True**. In-process flags, memory and raw predictions are saved. One-case same-host equality is bounded evidence; independent holdout scores/selection remain unchanged.

## Resource accounting and verification

Latest available cumulative actual/elapsed GPU-session estimate: **215.06554621334493 minutes**; active session: **False**. In-flight reservations are conservative, not reported as completed execution. Five-hour ceiling is cumulative18,000 seconds, including all historical M1–M4/v1/v2/v3 time; no reset. The resource-only continuation changed planning time alone under explicit user authorization. v3 guards its own allocation plus600 evaluation seconds below that ceiling. No new model downloads beyond counted official weights; download ledger remains5,972,695,891/10,000,000,000 bytes.

An overnight Windows idle-sleep/hibernate interrupted v3 at committed step2,090/cursor1,320 in fine-tuning7. The original wall-time ledger is preserved unchanged. Windows power/clock records independently evidence14,514.172 non-execution seconds, conservatively subtracted in a **separate** reconciliation; later sleep outside that session is not subtracted. Resume uses the exact optimizer/RNG/cursor, unchanged scientific sources/configuration, Windows working-state time and a temporary idle-sleep hold. See `research/resource_reconciliation.md`, original power-event XML and both ledgers. Charged working-state sessions include CPU/loading overhead and upper-bound actual GPU execution; they are not kernel-only profiling.

**114 tests passed**, one existing PyG deprecation warning. Tests include complete existing baselines, image/target transforms, tiny targets, collisions, finite loss/gradients, directed graph boundaries, decoder equivalence, exact CPU optimizer/RNG/cursor resume, corrected halo invariance, validation-only tie selection, repeated-holdout refusal and five-hour guard. Historical and new source hashes must pass final verification; CPU audits are not model quality evidence. Full hardware-failure fault injection and exhaustive pretraining/source provenance are unexecuted.

## Reproduction and next boundary

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m sensegraph.corrected_visual_training verify
& $python -m sensegraph.corrected_visual_training run  # only resume an incomplete corrected run
& $python scripts/report_visual_quality.py
# Once locked selection/evaluation exists, CPU image-only inference:
& $python scripts/infer_full_visual.py --image data/synthetic/v3/images/synthetic-01400.png --output experiments/visual_training/release/inference.json
```

Independent evaluation command: `python scripts/evaluate_visual_holdout.py`; it is one-time and refuses existing attempts. Its config, model-selection record and outputs are retained; do not erase them to rerun. No new deployment or frontend. Deliver the selected visual checkpoint and truthful model card; graph-on-imperfect-proposal training, numerical calibration/QA and user-facing accessibility remain later validated research stages, not completed by detector metrics.
