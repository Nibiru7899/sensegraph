# SenseGraph project status

Controlled real-domain generalization pilot: **completed**. Model quality remains insufficient for reliable real-chart QA.

## Implemented and executed

- Resolved the synthetic collision blocker in an explicit version2 generator: correlated trends plus 25% crossing-line charts. All 1,700 predetermined indices/series counts/styles retained; no rejection sampling or raised 5% threshold. Old failed inventory/code/results preserved.
- Train1,400 / validation 300 charts; training line collisions1.45%, validation1.26%. Full rendering/transform/hash/identity checks pass. All 1,360 non-line images intentionally match v1;340 line images changed; all 1,700 node counts preserved. Validation reuse is disclosed.
- Completed five alternating epochs and 875 optimizer updates per arm: original-data continuation control and realistic-data adaptation. Same A initialization, optimizer/exposure, FP32, resolution, decoder and freeze policy. Frozen tensors and BN state remained unchanged.
- Selected control epoch 1 and realistic epoch 3 using validation only, with the synthetic AP50 retention guard. TrOCR selected by real-validation GT-crop CER. Detector/OCR identities locked before one final 60-image inference per arm. No final-result-driven retraining.
- **130 tests passed** after training. CPU offline checkpoint reload and prediction equality pass; 59 active parameter tensors changed; frozen tensors/BN buffers equal A. Best.pt and selected epoch 3 have equal tensor states (container SHA values differ).

## Measured final results

Supported-label box recall at IoU.50 on 60 fresh official-training images,20 per horizontal bar/vertical bar/line family. These are partial-label source-inventory metrics, not official ChartQA test accuracy.

| Arm | Macro supported recall50 | Mean proposals/chart |
|---|---:|---:|
| A | 6.6836% | 14.67 |
| B-control | 7.3044% | 13.20 |
| B-realistic | 34.9033% | 56.43 |

Realistic minus control: **27.60 percentage points**, conditional 95% paired chart-bootstrap interval **[21.03, 34.09]**; 2,000 resamples, one training seed. Control minus A interval includes zero. Higher proposal volume is a material tradeoff; global precision remains unavailable.

Realistic per-class recovery: bars 235/340; text 195/767; legend labels 11/49; valid line envelopes 69/305. TrOCR final GT-crop CER 0.930926; detected-crop supported exact-region recovery 26/816 (3.1863%). No inference exceptions; two GT-crop token-limit truncations recorded.

## Limitations and next work

- Recall improved, but many marks/text regions remain missed or poorly localized. Long bars can yield duplicates; text boxes can truncate words. OCR remains poor even with GT crops. Reliable real-chart QA, learned numerical calibration and evidence-grounded reasoning are not established.
- Real labels remain partial/noisy. Untranscribed ticks, unverified axis descriptions, multi-model bar marks and invalid/zero-area boxes stay excluded/unknown. Unmatched predictions unknown. No global real AP/precision/F1 or real point/tick accuracy claims.
- Validation reuse,274 synthetic near-hash candidates, partial annotation coverage and unresolved publisher/pretraining overlap limit generalization claims. No claim of zero leakage or no overfitting. Model selection preserved earlier checkpoints when later validation declined; one seed does not establish robustness.
- Next work requires a separately approved protocol: reliable text localization/transcription and trustworthy real annotations before graph/QA/calibration training. No frontend, new GATv2 or QA training performed here.

## Resources and preservation

Charged GPU-work total **260.1963/300 minutes**; remaining **39.8037 minutes**. Includes the evidenced conservative 168-second reboot upper bound. Current charged ledger chain is closed; the raw interrupted reservation is preserved with a separate reconciled replacement, not reset or silently discarded. Committed cursor 640/step 80 was resumed without replay. Windows event evidence is under recovery/.

No new v2 transfers. Lifetime downloaded6,268,235,048 bytes; cumulative pilot transfers295,539,157 bytes remain under the1GB/10GB limits. All historical checkpoints/data/experiments preserved. Original status archive SHA256 e2bde3b3943813073128fa2f55b9c73fc6dd1733872ad86dbcecf4c96dd5b9e3; previous blocked status/report also archived with digests under v2/archive/.

[Detailed report](research/real_domain_generalization.md) · [V2 protocol](research/adaptation_protocol_v2.md) · [Baseline reproduction](research/baseline_report.md) · [Architecture](research/neural_model_architecture.md)

Config: configs/adaptation_v2.json. Artifacts: experiments/domain_generalization/v2/. CLI: python -m sensegraph.adaptation report / verify. The initial100-question QA pilot and oracle-proposal GAT/MLP findings describe different tasks and remain unchanged.
