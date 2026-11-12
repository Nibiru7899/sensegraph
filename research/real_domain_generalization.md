# Controlled real-domain generalization: version2 continuation

**Outcome: matched pilot completed.**

## Preservation and data amendment

The user authorized resolving the blocker and completing the pilot within the existing lifetime allowance. The entire prior study, code, data, failed gate, diagnostics, ledgers and reports remain preserved. Prior report/status copies with digests are under `experiments/domain_generalization/v2/archive/`; original v1 source guards remain valid.
The version2 generator keeps all 1,700 predetermined chart indices, font/family cells, titles, layouts, resolutions and series counts. Only line values change: correlated common trends in 75% of line charts, opposing crossing trends in 25%. No sample retries, filtering, annotation deletion, threshold increase or model-result-driven generator tuning. Every instance remains in evaluation. Segmentation remains disabled.
Non-line charts intentionally repeat the prior validation diagnostic cohort. This is a versioned repair, not a new independent synthetic holdout. Real validation 30 was already observed; final 60 was closed until locked inference. No zero-overfitting or zero-leakage claim is justified.

## Data and verification gates

All 1,700 records: gate passed=True; failures=[]

| Split | Class | Instances | Colliding instances | Fraction |
|---|---|---:|---:|---:|
| train | 6 | 2800 | 0 | 0.0000% |
| train | 7 | 17571 | 0 | 0.0000% |
| train | 4 | 23117 | 0 | 0.0000% |
| train | 5 | 2666 | 0 | 0.0000% |
| train | 1 | 14616 | 0 | 0.0000% |
| train | 3 | 4901 | 71 | 1.4487% |
| train | 2 | 4249 | 51 | 1.2003% |
| val | 6 | 600 | 0 | 0.0000% |
| val | 7 | 3790 | 0 | 0.0000% |
| val | 4 | 4985 | 0 | 0.0000% |
| val | 5 | 573 | 0 | 0.0000% |
| val | 1 | 3168 | 0 | 0.0000% |
| val | 3 | 1033 | 13 | 1.2585% |
| val | 2 | 880 | 12 | 1.3636% |

Classes:1 bar,2 point,3 line envelope,4 text,5 legend label,6 axis,7 tick. The per-class/per-split gate stays 5%. Union heatmaps/masked ambiguous regressions and all-instance scoring are unchanged.
Automated renderer bounds, transforms, source hashes and train/validation identities were checked. Anomaly overlays and near-hash candidates are retained in data_review.json; visual similarity is not proof of shared data. Font sets/title grammar/numerical arrays are separated. Old benchmark tests remain untouched.
Real inventory and conservative supported-label exclusions are byte-identical to v1. Untranscribed ticks, unverified axis descriptions, multi-model bar marks, invalid/zero-area boxes and unaligned line models remain unscored/unknown. No real AP, global precision/F1 or real point/tick accuracy is claimed. Agent review is not exhaustive human relabeling; source annotations remain noisy.

## Matched fitting

Fitting status: `paired_early_stop`. Completed arm epochs: {'control': 5, 'realistic': 5}.
Both arms initialize the same pinned A checkpoint with fresh optimizers. Control uses original synthetic training data; realistic uses the repaired inventory. Architecture640, FP32/TF32 off, C4/C5+FPN+active heads, frozen BN/unused branches, batch1/accumulation8, AdamW1e-4/1e-5, decay1e-4, clip1, warmup/cosine and fixed decoder are matched. No graph/QA/calibration training or OCR fine-tuning occurred.
Training/validation curves and optimizer/RNG/cursor checkpoints are retained. Selection uses real-validation macro supported recall50 and synthetic AP50 >= A-.05, earliest ties; overall A/control/realistic tie order. A five-epoch endpoint with improving validation is a bounded pilot, not demonstrated convergence.

| Arm | Epoch | Updates cumulative | Train loss | Synthetic val AP50 | Real val recall50 |
|---|---:|---:|---:|---:|---:|
| A | 0 | 0 | — | 0.251640 | 0.053122 |
| control | 1 | 175 | 0.16107 | 0.246011 | 0.062212 |
| control | 2 | 350 | 0.14529 | 0.243542 | 0.056476 |
| control | 3 | 525 | 0.12258 | 0.233780 | 0.051073 |
| control | 4 | 700 | 0.10586 | 0.231606 | 0.048721 |
| control | 5 | 875 | 0.09934 | 0.231979 | 0.050501 |
| realistic | 1 | 175 | 3.35706 | 0.537506 | 0.168591 |
| realistic | 2 | 350 | 1.54141 | 0.666201 | 0.196404 |
| realistic | 3 | 525 | 1.30326 | 0.708726 | 0.211120 |
| realistic | 4 | 700 | 1.20618 | 0.719932 | 0.205979 |
| realistic | 5 | 875 | 1.17509 | 0.722180 | 0.206043 |

Profile admission: `{'admitted': True, 'projected_required_seconds': 3840.02687499964, 'available_seconds': 4995.9934522992935, 'evaluation_reserve_seconds': 1200}`. Projected full arm epoch: 264.00s, including checkpoint I/O and conservative timing margin. Profile invariants: {'frozen_parameters_equal': True, 'BN_buffers_equal': True, 'checkpoint_reload_equal': True}. Profile updates were discarded; scientific arms reload A.

## Locked final evaluation

Validation-selected detector: `realistic` epoch 3; OCR: `trocr`. Selection digests were written before final inference. Ineligible detector arms (reported but not selectable): [].
Final inventory:60 fresh official-training images,20 per orientation/type; this is a held-out source inventory, not official ChartQA benchmark-test performance. No final-result-driven retraining/reselection.

| Arm | Macro supported recall50 | Mean proposals | Unknown unmatched predictions | Failures |
|---|---:|---:|---:|---:|
| A | 0.066836 | 14.67 | 767 | 0 |
| control | 0.073044 | 13.20 | 671 | 0 |
| realistic | 0.349033 | 56.43 | 2876 | 0 |

### A: per-class supported recall

| Class | Recovered / support | Recall50 | Matched mean IoU |
|---|---:|---:|---:|
| text | 43/767 | 0.056063 | 0.5952768048574758 |
| bar | 54/340 | 0.158824 | 0.7145101715017248 |
| legend_item | 0/49 | 0.000000 | None |
| line_segment | 16/305 | 0.052459 | 0.6835393942892551 |

Failure categories: {'localization': 81, 'detection_miss': 1262, 'recovered': 113, 'class_confusion': 5}. Thin envelopes: {'support': 37, 'recovered': 0, 'recall50': 0.0}. Full pass 2.625s, including preprocessing/overlays/scoring; not isolated model-kernel latency.

### control: per-class supported recall

| Class | Recovered / support | Recall50 | Matched mean IoU |
|---|---:|---:|---:|
| text | 42/767 | 0.054759 | 0.5859602235612416 |
| bar | 64/340 | 0.188235 | 0.6865933192893863 |
| legend_item | 0/49 | 0.000000 | None |
| line_segment | 15/305 | 0.049180 | 0.6933927694956462 |

Failure categories: {'localization': 74, 'detection_miss': 1261, 'recovered': 121, 'class_confusion': 5}. Thin envelopes: {'support': 37, 'recovered': 0, 'recall50': 0.0}. Full pass 2.625s, including preprocessing/overlays/scoring; not isolated model-kernel latency.

### realistic: per-class supported recall

| Class | Recovered / support | Recall50 | Matched mean IoU |
|---|---:|---:|---:|
| text | 195/767 | 0.254237 | 0.6040747850369185 |
| bar | 235/340 | 0.691176 | 0.6748023466860994 |
| legend_item | 11/49 | 0.224490 | 0.586226837201552 |
| line_segment | 69/305 | 0.226230 | 0.6638566393783127 |

Failure categories: {'localization': 450, 'detection_miss': 475, 'recovered': 510, 'class_confusion': 26}. Thin envelopes: {'support': 37, 'recovered': 1, 'recall50': 0.02702702702702703}. Full pass 2.891s, including preprocessing/overlays/scoring; not isolated model-kernel latency.

### Paired chart-bootstrap comparisons

Differences in supported macro recall50; 2,000 stratified chart resamples, seed20261110. Uncertainty is conditional on this one training seed, label coverage and source inventory. No across-seed robustness claim.

| Comparison | Difference | Conditional95% interval |
|---|---:|---|
| control_minus_A | 0.006207 | [-0.004024990319275626, 0.01769517452363729] |
| realistic_minus_A | 0.282197 | [0.21754577115458174, 0.34613129927667374] |
| realistic_minus_control | 0.275990 | [0.21030984976985823, 0.3409374245135697] |

## OCR: GT crops versus detected crops

| Validation recognizer | GT-crop CER | Token WER | Exact-region accuracy |
|---|---:|---:|---:|
| ctc | 1.044617 | 1.006024 | 0.000000 |
| trocr | 0.806572 | 0.861446 | 0.200855 |

Final selected OCR oracle: `{'regions': 816, 'characters': 5704, 'character_errors': 5310, 'CER': 0.9309256661991585, 'exact_region_accuracy': 0.16176470588235295, 'reference_words': 1096, 'correct_words': 157, 'substitutions': 853, 'deletions': 86, 'insertions': 185, 'word_recovery_accuracy': 0.14324817518248176, 'WER': 1.0255474452554745, 'normalization': 'NFC only; case, punctuation, sign and decimal differences retained'}`; truncations=2; failures=[].

Final selected OCR detected: `{'regions': 816, 'characters': 5704, 'character_errors': 5370, 'CER': 0.9414446002805049, 'exact_region_accuracy': 0.031862745098039214, 'reference_words': 1096, 'correct_words': 28, 'substitutions': 223, 'deletions': 845, 'insertions': 2, 'word_recovery_accuracy': 0.025547445255474453, 'WER': 0.9762773722627737, 'normalization': 'NFC only; case, punctuation, sign and decimal differences retained', 'supported_exact_region_recall': 0.031862745098039214, 'provided_regions': 816, 'matched_regions': 206, 'unknown_unmatched_predictions': 1662, 'precision': None, 'F1': None, 'scope': 'Supported real-text recovery only; missed provided text is deletion, unmatched real predictions unknown'}`; truncations=0; failures=[].

Native preprocessing, same geometry-only rotation rule, pinned TrOCR revision and greedy beam1/max128 tokens remain unchanged. CER/WER retain case/sign/decimal differences; CER can exceed1 through insertions. Missed provided text remains in denominators; unmatched predictions unknown. Historical word F1 was exact-region F1, not token-level F1. Numerical/negative/decimal/long/vertical/unseen-string subsets are in raw oracle JSON; zero support gives no performance claim.

## Resource accounting and reproducibility

Closed-ledger cumulative GPU work: 260.1963/300min; remaining 39.8037min; v2 work 44.3119min. No new downloads. Budget chains preserve all prior attempts and working-state timers/idle-sleep holds.
Real/mixed fitting remains ineligible without complete background supervision and a separate allocation. No trained QA or reliable real-chart QA claim follows even if detection recall improves.
```powershell
.\.venv\Scripts\python.exe -m sensegraph.adaptation validate
.\.venv\Scripts\python.exe -m sensegraph.adaptation inspect # before freezing only
.\.venv\Scripts\python.exe -m sensegraph.adaptation freeze
.\.venv\Scripts\python.exe -m sensegraph.adaptation profile
.\.venv\Scripts\python.exe -m sensegraph.adaptation fit
.\.venv\Scripts\python.exe -m sensegraph.adaptation evaluate # immutable final attempt
.\.venv\Scripts\python.exe -m sensegraph.adaptation report
.\.venv\Scripts\python.exe -m sensegraph.adaptation verify
.\.venv\Scripts\python.exe -m pytest -q --basetemp experiments/adaptation_recheck_tests
```
Config: configs/adaptation_v2.json; protocol: research/adaptation_protocol_v2.md; all artifacts: experiments/domain_generalization/v2/. Existing baseline reproduction stays in research/baseline_report.md. The original v1 report is preserved under the v2 archive. Final inference is not rerun by report/verify.

## Interpretation and executed handoff checks

The measured data-adaptation effect is positive on this supported-label inventory: realistic minus control 27.60 percentage points, conditional 95% interval [21.03, 34.09]. The control-minus-A interval contains zero. These differences support the value of this versioned synthetic-data adaptation for the fixed model/protocol; they do not isolate a generally superior architecture or establish performance across training seeds. Proposal counts rose markedly, and partial labels prevent a full precision assessment.

130 tests passed after training (experiments/adaptation_posttraining_tests_stdout.log). Offline CPU inference from a development image gives exact save/reload equality. Frozen parameters and BN buffers equal A; 59 active parameter tensors changed. The selected epoch 3 file and best.pt contain equal model tensors, though serialized-container hashes differ. See offline_checkpoint_check.json. Every completed epoch checked frozen tensors/buffers; inference failures remain in denominators.

A system reboot interrupted control epoch 1. The original unresolved 3300-second reservation remains unmodified. Recovery charged a conservative 168-second upper bound from command-log creation to the next OS boot, including verification/shutdown overhead. Relevant Windows reboot events and hashes, original chain, replacement ledger and saved cursor are preserved under recovery/. The resumed job started at cursor 640/update 80, without replay. The current charged chain is closed. Total charged work is 260.1963min, including that upper bound; this is not a claim that168 seconds were exactly measured GPU execution.

Qualitative review finds partial-word text crops and duplicate/incomplete long-bar boxes. Final OCR recovery 26/816 remains inadequate, and GT-crop CER is high. Reliable chart QA is not ready. Numerical values, graph proposals and final answers were not supplied as inference inputs. No GATv2/QA/calibration training or frontend was added. The final endpoint is a bounded single-seed study, not an absence-of-overfitting guarantee.

Retention was checked on the declared repaired synthetic validation inventory. Historical holdout results were preserved, not reopened for tuning; this does not prove absence of forgetting on every old domain. The final 60 images were reviewed for annotation compatibility before fitting, but were held out from model fitting/selection.

![Measured learning curves](../experiments/domain_generalization/v2/learning_curves.png)

Regenerate the figure with `python scripts/plot_adaptation.py --config configs/adaptation_v2.json`. Report/verify do not repeat final inference. The archived v1 report preserves the original failure and diagnostics.
