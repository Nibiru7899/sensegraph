# SenseGraph-Net visual training — executed diagnostics and blocked curriculum

**Status: actual pretrained visual optimizer training completed on a tiny training subset; full 1,400/300 curriculum has NOT started.** This is a passed memorization/profile gate, not evidence of validation improvement or convergence.

## Implementation and provenance

The prescribed ResNet18/FPN/fusion and six perception heads are used unchanged. Official ImageNet1K_V1 weights: full SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`,46,830,571 bytes; official source/ledger in `weights.json`. New heads retain the specified initialization. No random-backbone fallback, renderer proposals at inference, segmentation supervision, GAT retraining, QA training or test access.

Implemented `visual_training.py`: exact640 letterboxing and inverse xyxy transforms, valid image masks, center/keypoint Gaussian targets, class-specific extents/offsets, typed image-only peak decoder, top128 cap and NMS, modified focal/SmoothL1/L1 losses. Five box classes and two keypoint classes map back to the historical physical ontology. Segmentation is excluded. `visual_experiment.py` provides counted acquisition, target preparation/freeze, training, validation, per-epoch checkpoint/optimizer/scaler/RNG histories and image-only overlays. The curriculum includes five warmup plus up to30 C4/C5 fine-tuning epochs, accumulation8, clipping, staged LR warmup/cosine, validation and early stopping; its full-run gates deliberately remain closed.

## Data integrity and target gates

All **1,400 training and300 validation charts** were scanned. Invalid supported boxes:0. Unsupported series-union annotations: train2798, val590; they are counted, not invented as physical detections. Positive-area small objects are retained. Native-pixel target overlays reviewed on a bar, grouped bar and line chart: predicted target centers align with actual objects/ticks (`targets_0.png`–`targets_2.png`). Full distributions in `data_statistics.json`.

Same-class stride4 center collisions: train points173/6424 (2.69%), segments160/5254 (3.05%); validation points3.06%, segments3.90%. These exceed the existing1% gate. Some are genuinely coincident physical marks; one center slot cannot distinguish every instance. Regression targets at conflicting cells are masked, center presence retained, collisions reported. No gate was relaxed and no samples removed. This is a representation/data-policy bottleneck distinct from optimizer fitting.

## Actual optimization and failures

The original AMP smoke stopped with a nonfinite gradient, before a saved optimizer update; retained in `smoke_stdout.log` and the resource ledger. A separately declared FP32 investigation preserved the original source/config/attempt. It performed100 steps on one chart then300 cycling four training charts. Every recorded loss/gradient was finite; initial20 mean loss4.9123, final200.1453. Actual checkpoint `experiments\visual_training\v1\fp32_diagnostic\checkpoint.pt` SHA256 `84fb2e39b861b01f615abb24e7e2dfe218c50cc5c61f177e5e172ddd19689cb0`. Reload inference was exactly equal: **True**. BN statistics stayed frozen through the model's explicit training policy.

The first FP32 run completed400 fitting steps, saved the checkpoint and ran20 extra fine-tuning timing updates, but final JSON export failed on a NumPy boolean. Its original script, trace and attempt are retained. Recovery loaded that checkpoint without repeating fitting; metrics were recomputed and20 fine-tuning timing updates repeated. Those40 timing updates are discarded and never presented as a trained full-split checkpoint. Recovery GPU time is charged. The isolated script's bool conversion was corrected; no frozen main source/config was changed after prepare.

## Four-training-chart results — NOT validation performance

Macro recall50 **0.9396**, macro AP50 **0.9195**, macro AP75 **0.8477**. Tiny fitting gate passed: **True**. Actual overlays reviewed for bar/grouped-bar/line show learned boxes aligned with individual marks, text and axes, with some duplicates/imperfect extents. This supports supervised wiring and capacity to memorize, not generalization.

| Class | GT | Precision | Recall | AP50 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bar | 21 | 1.0000 | 1.0000 | 1.0 | 0 | 0 |
| point | 24 | 1.0000 | 1.0000 | 0.9999999999999999 | 0 | 0 |
| line_segment | 21 | 1.0000 | 0.9524 | 0.9523809523809523 | 0 | 1 |
| text | 51 | 1.0000 | 1.0000 | 1.0 | 0 | 0 |
| legend_item | 8 | 1.0000 | 1.0000 | 1.0 | 0 | 0 |
| axis | 8 | 0.6250 | 0.6250 | 0.484375 | 3 | 3 |
| tick | 39 | 1.0000 | 1.0000 | 1.0 | 0 | 0 |

Explicit native-pixel keypoints: point24/24 matched within5px, mean error0.08579598503032078; tick39/39, mean error0.08153091294643207. Precision/FP/FN also retained in `fp32_diagnostic/result.json`. Box/AP evaluation uses learned extents; center recovery is a different metric and is reported separately. No oracle box is supplied to the decoder.

The old detector's .170 recall/.1134 AP50 and zero point/tick recalls were measured on30 validation charts after128-chart development. The new values above are four-chart **training** measurements: **they are not a fair performance comparison or evidence of improvement**. All300 validation charts and the first30 matched subset remain to be evaluated after legitimate curriculum training. Existing evaluation code preserves the old targets; resolution, heads and training exposure differ. The full-run validation command path records AP50/AP75, losses, class recall, localization, keypoint FP/FN and timing.

## Resource accounting and concrete budget proposal

Warmup batch median0.1063s; fine-tuning batch median0.1140s (20 measured updates). Evaluation time extrapolated from the four diagnostic images. Maximum5+30 epoch estimate **168.2 minutes**, including25% overhead. Early stopping may reduce it; this is not a promise of convergence. Actual recovery/profile peak allocated/reserved **497.1/618.0MiB**; original400-step run peak was not recovered after the export error and is not fabricated.

New total charged sessions52.827s; lifetime **22.809/60 minutes**, leaving37.19 minutes. Download ledger **5,972,695,891/10,000,000,000 bytes**. No reset. `gpu_budget.json` preserves AMP failure, FP32 fitting and metric recovery reservations/actuals.

**Approval proposal:** raise cumulative GPU limit to **240 minutes** (additional180 minutes above the old cap), and authorize a new frozen FP32 protocol allowing at most5% per-class colliding target objects with the existing conflicting-regression masks, retaining every GT object in evaluation. The proposed relaxation is explicit and unapproved; it does not resolve unidentifiable coincident objects. Alternative: redesign collision representation or generate independently validated collision-controlled training data under another protocol. Do not silently overwrite v1 or treat a budget increase alone as target-policy approval. Details: `budget_proposal.json`.

## Verification, commands and next step

**95 tests passed** including historical tests; one existing PyG deprecation warning. **5120 historical files unchanged.** Scripts and frozen hashes remain in this namespace. `diagnostic_learning_curve.png` marks the single-chart/four-chart transition; it is not an epoch validation curve. Full curriculum checkpoint, full validation metrics and convergence curves are **unavailable** until gates and budget authorization are resolved.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m sensegraph.visual_experiment acquire
& $python -m sensegraph.visual_experiment prepare
# v1 AMP failed; do not overwrite or rerun its attempt.
& $python scripts/diagnose_visual_fp32.py
# Recovery was required solely for the recorded export error; refuses overwrite.
& $python scripts/recover_visual_diagnostic.py
& $python -m sensegraph.visual_experiment verify
& $python scripts/report_visual_training.py
```

Current checkpoints/attempts cannot be silently overwritten. `train` has a 1% collision gate, visual-review gate, smoke gate and projected-budget gate; the isolated FP32 diagnostic does not forge the v1 smoke record to bypass those guards. After explicit approval, prepare a separate frozen FP32 curriculum namespace, retain/respect all prior time, correct remaining continuation/checkpoint-resume ergonomics, and run the prescribed full inventories. No graph/QA/frontend work is authorized in this stage.
