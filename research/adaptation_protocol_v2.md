# Controlled adaptation continuation v2

User authorized resolving the data blocker and completing the matched pilot within
the existing lifetime GPU allowance on 2026-11-05. No limit increase.

Preserve the entire v1 study and all its frozen code/data/diagnostics. This is an
explicit dataset/process amendment, not silent replacement of failed samples.
Keep all 1,700 predetermined indices, families, fonts, resolutions, titles, layouts,
series counts and data-value ranges from seed20261107. Only the line value process
changes: independent uniform innovations become correlated series around a common
trend (75% of line charts), or opposing trends with crossings (25%). See the exact
formula in adaptation_data.py. No rejection, retries, collision-based per-sample
editing, annotation deletion or metric-based generator tuning. A failed gate still
blocks fitting. Retain 5% per-class/per-split ceiling, union heatmaps, ambiguous
regression masking, full-instance evaluation and disabled segmentation.

Non-line charts intentionally repeat the v1 diagnostic cohort; this is not a new
independent synthetic holdout. Old synthetic and real-validation A diagnostics
were already observed. The real final60 remain closed, and their results must not
change generation, fitting, selection, preprocessing or decoding. No independent
generalization claim follows from reused validation.

All other choices inherit domain_generalization_protocol_v1.md: A frozen release,
old synthetic continuation control, matched new realistic synthetic continuation;
five paired alternating epochs /875 updates per arm, shared stopping >=3 paired
epochs with both stale twice, seed20261109, FP32/TF32 off, C4/C5+FPN+active heads,
frozen early stages/BN/unused branches, microbatch1/accumulation8, AdamW head1e-4,
backbone1e-5, decay1e-4, clipping1, warmup then cosine floor1%. Preserve committed
optimizer/RNG/cursors. Validate each epoch on synthetic300 and fixed real30. Select
macro supported real recall50 with synthetic AP50 >= A-.05, earliest ties; overall
A/control/realistic tie order. Fixed thresholds .10/.25, NMS.50, cap128.

Budget inherits the latest closed v1 ledger (215.8844 minutes total). At most300s
profiling,3300s fitting,1200s final/OCR, lifetime18000s. Profile both arms and atomic
checkpoint writes plus complete validation before admission. Conservative p95
update projection plus validation and I/O overhead must fit the full ten epochs
and evaluation reserve. Recheck remaining capacity at fit start. If capacity fails,
report that blocker; never call partial fitting complete.

Real inventory/coverage exclusions and OCR assets remain byte-identical to v1.
Real-only/mixed fitting remains ineligible; no new downloads, graph/QA/calibration
training, architecture/decoder changes, OCR fine-tuning or frontend.
Compare CTC and pinned TrOCR with unchanged preprocessing on validation crops.
Lock selected detector and recognizer before one final60 inference per arm.
Report all arms, eligibility, failures, proposal counts, per-class recall/localization,
OCR diagnostics, conditional single-seed paired chart bootstrap2000 seed20261110,
latency, memory and complete resource chain. No final-driven reselection.

CLI: `python -m sensegraph.adaptation COMMAND --config configs/adaptation_v2.json`.
Commands: init, prepare --limit100, prepare, validate, inspect, freeze, profile, fit,
evaluate, report, verify. Archive current status/report with digests before updates.
