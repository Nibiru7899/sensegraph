# Milestone 4 — bounded learned perception

**Status: stopped at learned-detection bottleneck. Milestone 4 is not claimed complete.**

## Implemented scope

New frozen protocol/config, official pretrained MobileNetV3 Faster R-CNN acquisition through the existing budget, local checkpoint-only detector factory, frozen ROI image features, small RPN/classification/regression head training, image-only proposals, spatial/complete proposal graphs, empty/duplicate/false-region handling, one-to-one typed IoU matching, AP/recall/localization/duplicate metrics, and end-to-end/conditional relation recovery evaluator. A local optional Tesseract adapter reports unavailable without inserting renderer text. Rotations beyond explicit right angles and OCR recognition accuracy are unsupported/unmeasured here.

The renderer's `series` union is not a painted detection target. Text and enclosing segment rectangles retain declared semantics, not point or mask substitutions. Unsupported/invalid annotations are recorded. Every GT relation, including those involving unobserved series nodes, remains in end-to-end relation-recovery denominators. Existing node types, values, source text and semantic adjacency never enter image-only forward methods.

## Actual validation detector measurements

240 development steps; 30 validation charts. Macro recall@IoU .50: **0.1700**; macro AP@.50: **0.1134**. Macro AP@.75: **0.0131**. These are validation development results, not untouched-test evidence. The predeclared gate passed: **False**. No post-gate hyperparameter search or replacement with oracle scores.

| Class | GT | Precision | Recall | AP50 | FP | FN | Duplicates | Normalized center error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bar | 182 | 0.4805 | 0.4066 | 0.29069357084409975 | 80 | 108 | 6 | 0.05715445447335574 |
| point | 136 | 0.0000 | 0.0000 | 0.0 | 32 | 136 | 0 | unavailable |
| line_segment | 113 | 0.3926 | 0.4690 | 0.3722951933494467 | 82 | 60 | 15 | 0.08799881150700012 |
| text | 405 | 0.1039 | 0.1630 | 0.03384561685694463 | 569 | 339 | 9 | 0.09999901293353601 |
| legend_item | 56 | 0.0333 | 0.0179 | 0.0012755102040816326 | 29 | 55 | 0 | 0.12067306041717529 |
| axis | 60 | 0.1127 | 0.1333 | 0.09553870826078562 | 63 | 52 | 1 | 0.023332332842983305 |
| tick | 315 | 0.0000 | 0.0000 | 0.0 | 13 | 315 | 0 | unavailable |

AP integrates the interpolated precision envelope over achieved recall (not COCO's 101-point mAP). Confidence >= .25 for precision/recall; AP uses all returned detections >= .01 with a 100-proposal cap. Normalized center errors, .75-IoU results and raw predictions are retained in `result.json`/`validation.pt`.

Final checkpoint reload exact: True. Median measured pipeline latency: 0.0437 seconds/image (decoding + detector + frozen ROI embeddings + graph construction + two **untrained** relation wiring forwards). This is not trained end-to-end relationship accuracy. Peak allocated/reserved: 221.7/322.0 MiB. Frozen ROI embeddings are 1024-D; geometry/color is 11-D. Only geometry/color is passed to the historical architecture for shape verification; new visual-feature relationship models have not been fitted. Per-image shape checks remain in the result.

OCR probe: `unavailable`. Recognition accuracy is unmeasured. Detector checkpoint, training log, before/after smoke overlays and six validation overlays are preserved. See `visual_review.json` if present for actual reviewed samples.

Acquisition is resolved: the user executed the unchanged budgeted official acquisition command. Source, full SHA256 and byte count are recorded in `weights.json`; trained checkpoint provenance is in `checkpoint_provenance.json`. Earlier acquisition blockers/reviews are retained as historical records, not current blockers. Smoke loss/gradient and exact reload/repeated-inference checks passed (`smoke.json`). Actual accepted-proposal ROI embeddings were finite and 1024-D (`smoke_interface_checks.json`). The auxiliary exact-boundary check found a 0.000031-pixel float32 overshoot on a subthreshold box; diagnostic records are retained. All accepted smoke boxes were strictly in bounds. No prediction, threshold or frozen model code was altered.

CPU error export: `scripts/export_perception_errors.py` writes `detection_metrics.csv` (both IoU thresholds, including localization error), `detection_errors.json` (missed node IDs, unmatched/duplicate detections, unsupported/invalid annotations), and checkpoint provenance. All validation examples remain in denominators. Two small additional diagnostic model sessions, including the initial failed strict-boundary assertion, are charged in the M4 GPU ledger alongside smoke and fitting.


## Verification and resources

- Complete suite: **84 passed**; one existing PyG deprecation warning. New fixtures exercise duplicate and type matching, exact IoU boundary, missing/extra relationship endpoints, empty graphs, graph boundaries, invalid/unsupported boxes and unavailable OCR.
- **5111 historical files unchanged**: old source/tests/configs/reports, v2/v3 data, sealed experiments and original GPU ledgers. The current download ledger is deliberately excluded from immutable files; its exact pre-run snapshot is frozen and all new transfers are counted.
- Downloads: **5,925,865,320/10,000,000,000 bytes**. New GPU-session charge: **23.157 seconds**. Lifetime **21.929/60 minutes**. No paid APIs or official benchmark test use.
- First 128 v3 training and 30 validation chart IDs and source hashes frozen in `experiments/milestone4/freeze.json`. These are development subsets, not independent generalization evaluation. Protected test sets stay closed.

## Remaining work and stop boundary

This bounded gate does not answer whether GAT still outperforms MLP with learned proposals. Matched learned-visual/projection models, three-seed predicted-proposal training, complete-connectivity comparison, controlled proposal degradation, OCR accuracy, new held-out synthetic evaluation and real-domain pilot remain **unexecuted**. Do not report M3 oracle results as M4 end-to-end performance. If detection fails, fix perception under a newly frozen protocol before continuing. A passed gate requires a separate frozen continuation, not automatic test access.

ChartQA's 20 partial training annotations remain development-only, not sufficient real-world evaluation. See `real_annotation_mapping.md`: missing/zero-area/partial labels need masking, series/line semantics differ, and VisText raster transforms remain unverified. Independently reviewed held-out real annotations and rights review are future work. No real-data annotation or calibrated evidence is invented.

## Commands and provenance

Use `.\.venv\Scripts\python.exe -m sensegraph.perception_experiment` with `freeze`, `acquire`, `smoke`, `fit`, `verify` (in that order; review `smoke_after.png` and record `visual_approval.json` before `fit`). `--root` is supported. Freeze verifies exact sources and historic artifacts; changes require a new protocol. Smoke/fit refuse to overwrite attempted runs. Interrupted reservations remain charged. Inspect partial artifacts before any continuation; never remove attempts or reset ledgers to retry fitting.

Run `.\.venv\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-m4 --junitxml=experiments/milestone4/tests.xml`, then `.\.venv\Scripts\python.exe scripts/report_perception.py`. Report regeneration is CPU-only and does not rerun models. Acquisition retry is resumable and records actual received bytes.

Architecture/API: [Torchvision 0.23 documentation](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html); code [BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE). COCO pretrained weights have underlying training-data terms; no blanket image-rights grant or redistribution is claimed. Official checkpoint URL/hash is in `weights.json` when acquired.

No frontend, full chart QA, accessibility claims or large-scale runs.
