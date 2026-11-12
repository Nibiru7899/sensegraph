# Controlled real-domain pilot protocol v1

This protocol is frozen before new model inference. The selected release (SHA256
`97b5cc72cf5f138d0d0709bcd01e40792cbcecc05b9b39a5f4146cabbb72cf8e`)
is A. B-control and B-realistic would start from exactly A with fresh AdamW states.
Only perception is eligible for adaptation. No graph, QA, calibration, or UI work.

## Data and admission

New rendering namespace `data/synthetic/domain_v1b`: 1,400 train / 300 validation,
seed 20261107. Five chart families balanced explicitly with five training fonts
and three disjoint validation fonts. Complete titles, title grammar, style hashes,
and numerical arrays are split-disjoint. Exact rendering metadata supplies boxes
and affine transforms. Every visible supported text artist is recorded; overlapping
or clipped text is ambiguous for recognition. Markerless vertices are not points.
Line rectangles are envelopes, never stroke masks. Segmentation stays disabled.

The initial 100 at `domain_v1` are preserved. The corrected namespace fixes percent
formatting for large magnitudes, xyxy plot bounds, and marker path extents. This
amendment preceded full generation and all new inference. It does not replace a
failed final gate. Validate all 1,700 records; the existing class-specific 5%
collision ceiling applies separately to train and validation. Union heatmaps and
masked ambiguous regressions retain every reference instance for evaluation.
**Any failed data gate prohibits profile optimizer updates and both fitting arms.**
Inventory freezing is permitted with `training_gate_passed=false`; it is not fitting
admission. No sample replacement, threshold relaxation, or final-result-driven fix.

Acquire official ChartQA training identities only, pinned full release
`af8b6f5c08c95085271561c2a3f9d15f2b5a9031`. Group exact images/tables/titles and
dHash distance <=4 before splitting; select one representative/group with seed
20261108. Train/development60, validation30, final60, each balanced h_bar/v_bar/line.
Prior20 remain development-only. Exclude known prior identities, identical tables,
titles and images. Near-prior groups are allowed only in development. Unresolved
publisher/pretraining overlap remains a limitation. Missing tick transcriptions do
not invalidate valid pixel boxes. Ambiguous line-model alignment is unsupported;
zero-area boxes unavailable; no padding, point boxes, or VisText raster transforms
are invented. Agent review validates correspondence, not exhaustive coverage.
Pre-inference candidate overlays exposed displaced axis-description boxes and
untranscribed numeric-tick boxes in source metadata (e.g. `two_col_1836`). Retain
them raw but exclude all axis descriptions and all untranscribed ticks from
supported-label scoring. Explicit supported node IDs/exclusions are frozen before
model inference; this conservative scope cannot establish whole-image recall.
Multi-model bar marks are also withheld: reviewed stacked/horizontal examples
contain displaced or cumulative-union rectangles (e.g. `multi_col_20889` and
`multi_col_80515`). This blanket pre-inference rule conservatively withholds even
some valid grouped bars; it does not remove their images or raw annotations.

## Matched fitting, conditional on all gates

Seed20261109; 640 input, decoder threshold .10, operating score .25, NMS .50, cap128.
Freeze early backbone, BN statistics/affine, segmentation, OCR and downstream
branches. Train C4/C5, FPN/fusion and active dense heads in FP32; TF32 disabled.
Batch1, accumulate8, AdamW, heads/FPN1e-4, backbone1e-5, decay1e-4, clip1.
First-epoch linear warmup; cosine to1% base LR by epoch5. Alternate complete
control/realistic epochs, max5 each /875 optimizer steps per arm. Shared early stop
after >=3 paired epochs when both have two epochs without real-validation gain
>=max(.001 absolute,.5% relative). A still-improving epoch5 is a bounded pilot.

Validate every epoch on new synthetic val and real val. Select macro supported-label
real recall at IoU.50 over supported classes, requiring synthetic AP50 >= A-.05.
Earliest epoch ties; overall ties A then control. Report proposal counts to expose
score inflation. Missing reference objects remain in all denominators. Partial real
background is unknown: no real/mixed fitting and no global real AP/precision/F1.

Inherit the latest closed working-state ledger. Profile <=300sec, fitting <=3300sec,
evaluation/OCR <=1200sec, cumulative <=18000sec. Admit only when ten projected
complete epochs plus evaluation reserve fit both remaining lifetime and stage
allowances. Preserve reservations/failures, RNG/optimizer/update cursors. Existing
Windows working-state timer excludes sleep; temporary system hold is restored.

## OCR and evaluation

Pinned Microsoft TrOCR-small-printed revision
`04e994ab854b0089d4929f48c2b4dbe2ce78a340`; safetensors245839136 bytes,
SHA256`49350a39968df83e5a1adc90fc0ede02ff247671aed70b842af350fd4a7103f3`.
Explicit TrOCRProcessor and VisionEncoderDecoderModel in Transformers4.57.3,
local_files_only, no dependency change or remote code. Official Microsoft project
README's MIT declaration is recorded separately from absent HF model-card license
metadata. Acquire no weight if permission verification fails. No substitutes.
New transfers <=1,000,000,000 bytes and lifetime <=10,000,000,000 bytes.

Same boxes, geometry-only clockwise90 rotation when height>1.5width; each
recognizer keeps its native preprocessing. Greedy decoding, beam1, no sampling,
max128 new tokens for TrOCR. Count truncations/failures, CER (NFC only), token WER,
exact-region accuracy, and supported provided-text recovery. GT-crop diagnostics
and predicted-crop results are separate. Unmatched partial-label predictions unknown.
Historical "word F1" was exact-region F1, not token-level word F1.

Select recognizer on trusted real-validation GT-crop CER; lock detector and OCR
identities before final inference. Final150-inventory subset60 is evaluated once
for all available complete arms, paired chart-type bootstrap2000, seed20261110.
No reselection after final. If fitting is blocked, final remains sealed; A-only
validation/OCR diagnostics may proceed with their own immutable attempt lock and
bounded resource session. They do not complete the matched study or establish
adaptation benefit. No bootstrap comparison is fabricated without two trained arms.

## Reproducible entry point

`python -m sensegraph.domain_experiment COMMAND --config configs/domain_generalization_v1.json`

Commands: `init`, `diagnose`, `prepare --limit 100`, `prepare`, `validate`, `acquire`, `inspect`,
`freeze`, `profile`, `fit`, `diagnose-validation`, `evaluate`, `report`, `verify`.
Freeze requires completed source acquisition and agent real review. Failed gates
exit nonzero for fit/profile; evaluate requires a complete paired campaign. Final
and diagnostic attempts are immutable. Historical root status is resolved through
the explicit archived-byte mapping, never by weakening snapshot hashes.
