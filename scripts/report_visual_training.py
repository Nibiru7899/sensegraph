"""Report executed visual-training diagnostics without claiming curriculum completion."""
from pathlib import Path
import sys,csv
import xml.etree.ElementTree as ET
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph.visual_experiment import verify,location


def main():
    d=location(ROOT); r=read_json(d/'fp32_diagnostic/result.json'); stats=read_json(d/'data_statistics.json'); ledger=read_json(d/'gpu_budget.json')
    integrity=verify(ROOT); tests=ET.parse(d/'tests.xml')
    assert not tests.findall('.//failure') and not tests.findall('.//error')
    passed=len(tests.findall('.//testcase')); total=ledger['historical_seconds']+ledger['charged_seconds']; remaining=3600-total
    metrics=r['ap50']['per_class']; table='\n'.join(f"| {name} | {v['support']} | {v['precision']:.4f} | {v['recall']:.4f} | {v['ap'] if v['ap'] is not None else 'unavailable'} | {v['fp']} | {v['fn']} |" for name,v in metrics.items())
    with (d/'diagnostic_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        rows=[{'iou':iou,'class':name,**v} for iou in ['ap50','ap75'] for name,v in r[iou]['per_class'].items()]
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    log=read_json(d/'fp32_diagnostic/training_log.json'); figure=Figure(figsize=(9,4)); FigureCanvasAgg(figure); ax=figure.subplots()
    ax.plot([v['step'] for v in log],[v['loss'] for v in log]); ax.axvline(100,color='gray',linestyle='--'); ax.set(xlabel='Optimizer step',ylabel='Supervised detection loss',title='Training-only diagnostic: 100 fixed-chart, then 300 four-chart steps')
    figure.tight_layout(); figure.savefig(d/'diagnostic_learning_curve.png')
    proposal={'status':'approval_required_before_full_training','existing_cumulative_limit_minutes':60,'current_cumulative_minutes':total/60,
              'estimated_maximum_campaign_minutes':r['campaign_estimated_seconds']/60,'estimate_includes_overhead_fraction':.25,
              'proposed_cumulative_limit_minutes':240,'additional_authorization_above_existing_cap_minutes':180,
              'remaining_under_current_limit_minutes':remaining/60,'requested_protocol_amendments':{
                  'precision':'FP32; initial AMP attempt had nonfinite gradient',
                  'collision_gate':'New version allows <=5% per-class same-cell collision objects; keep conflicting regression targets masked and ALL physical GT instances in detection denominators. This requires explicit approval; current1% gate remains intact.',
                  'configuration':'New v2 namespace, source snapshot and frozen config; no reset of old ledgers'},
              'not_a_guarantee':'Estimate uses four training images for evaluation throughput and20 fine-tuning profile batches, not the complete300 validation charts; convergence/latency may differ.'}
    write_json(d/'budget_proposal.json',proposal)
    report=f'''# SenseGraph-Net visual training — executed diagnostics and blocked curriculum

**Status: actual pretrained visual optimizer training completed on a tiny training subset; full 1,400/300 curriculum has NOT started.** This is a passed memorization/profile gate, not evidence of validation improvement or convergence.

## Implementation and provenance

The prescribed ResNet18/FPN/fusion and six perception heads are used unchanged. Official ImageNet1K_V1 weights: full SHA256 `{read_json(d/'weights.json')['sha256']}`,46,830,571 bytes; official source/ledger in `weights.json`. New heads retain the specified initialization. No random-backbone fallback, renderer proposals at inference, segmentation supervision, GAT retraining, QA training or test access.

Implemented `visual_training.py`: exact640 letterboxing and inverse xyxy transforms, valid image masks, center/keypoint Gaussian targets, class-specific extents/offsets, typed image-only peak decoder, top128 cap and NMS, modified focal/SmoothL1/L1 losses. Five box classes and two keypoint classes map back to the historical physical ontology. Segmentation is excluded. `visual_experiment.py` provides counted acquisition, target preparation/freeze, training, validation, per-epoch checkpoint/optimizer/scaler/RNG histories and image-only overlays. The curriculum includes five warmup plus up to30 C4/C5 fine-tuning epochs, accumulation8, clipping, staged LR warmup/cosine, validation and early stopping; its full-run gates deliberately remain closed.

## Data integrity and target gates

All **1,400 training and300 validation charts** were scanned. Invalid supported boxes:0. Unsupported series-union annotations: train{stats['train']['unsupported']}, val{stats['val']['unsupported']}; they are counted, not invented as physical detections. Positive-area small objects are retained. Native-pixel target overlays reviewed on a bar, grouped bar and line chart: predicted target centers align with actual objects/ticks (`targets_0.png`–`targets_2.png`). Full distributions in `data_statistics.json`.

Same-class stride4 center collisions: train points{stats['train']['collision_objects']['2']}/{stats['train']['counts']['2']} ({stats['train']['collision_fraction']['2']:.2%}), segments{stats['train']['collision_objects']['3']}/{stats['train']['counts']['3']} ({stats['train']['collision_fraction']['3']:.2%}); validation points{stats['val']['collision_fraction']['2']:.2%}, segments{stats['val']['collision_fraction']['3']:.2%}. These exceed the existing1% gate. Some are genuinely coincident physical marks; one center slot cannot distinguish every instance. Regression targets at conflicting cells are masked, center presence retained, collisions reported. No gate was relaxed and no samples removed. This is a representation/data-policy bottleneck distinct from optimizer fitting.

## Actual optimization and failures

The original AMP smoke stopped with a nonfinite gradient, before a saved optimizer update; retained in `smoke_stdout.log` and the resource ledger. A separately declared FP32 investigation preserved the original source/config/attempt. It performed100 steps on one chart then300 cycling four training charts. Every recorded loss/gradient was finite; initial20 mean loss{r['loss_first20']:.4f}, final20{r['loss_last20']:.4f}. Actual checkpoint `{(d/'fp32_diagnostic/checkpoint.pt').relative_to(ROOT)}` SHA256 `{sha256(d/'fp32_diagnostic/checkpoint.pt')}`. Reload inference was exactly equal: **{r['checkpoint_reload_exact']}**. BN statistics stayed frozen through the model's explicit training policy.

The first FP32 run completed400 fitting steps, saved the checkpoint and ran20 extra fine-tuning timing updates, but final JSON export failed on a NumPy boolean. Its original script, trace and attempt are retained. Recovery loaded that checkpoint without repeating fitting; metrics were recomputed and20 fine-tuning timing updates repeated. Those40 timing updates are discarded and never presented as a trained full-split checkpoint. Recovery GPU time is charged. The isolated script's bool conversion was corrected; no frozen main source/config was changed after prepare.

## Four-training-chart results — NOT validation performance

Macro recall50 **{r['ap50']['macro_recall']:.4f}**, macro AP50 **{r['ap50']['macro_ap']:.4f}**, macro AP75 **{r['ap75']['macro_ap']:.4f}**. Tiny fitting gate passed: **{r['passed']}**. Actual overlays reviewed for bar/grouped-bar/line show learned boxes aligned with individual marks, text and axes, with some duplicates/imperfect extents. This supports supervised wiring and capacity to memorize, not generalization.

| Class | GT | Precision | Recall | AP50 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Explicit native-pixel keypoints: point{r['keypoints_5px']['2']['matched']}/{r['keypoints_5px']['2']['gt']} matched within5px, mean error{r['keypoints_5px']['2']['mean_localization_pixels']}; tick{r['keypoints_5px']['7']['matched']}/{r['keypoints_5px']['7']['gt']}, mean error{r['keypoints_5px']['7']['mean_localization_pixels']}. Precision/FP/FN also retained in `fp32_diagnostic/result.json`. Box/AP evaluation uses learned extents; center recovery is a different metric and is reported separately. No oracle box is supplied to the decoder.

The old detector's .170 recall/.1134 AP50 and zero point/tick recalls were measured on30 validation charts after128-chart development. The new values above are four-chart **training** measurements: **they are not a fair performance comparison or evidence of improvement**. All300 validation charts and the first30 matched subset remain to be evaluated after legitimate curriculum training. Existing evaluation code preserves the old targets; resolution, heads and training exposure differ. The full-run validation command path records AP50/AP75, losses, class recall, localization, keypoint FP/FN and timing.

## Resource accounting and concrete budget proposal

Warmup batch median{r['warmup_batch_seconds']:.4f}s; fine-tuning batch median{r['finetune_batch_seconds']:.4f}s (20 measured updates). Evaluation time extrapolated from the four diagnostic images. Maximum5+30 epoch estimate **{r['campaign_estimated_seconds']/60:.1f} minutes**, including25% overhead. Early stopping may reduce it; this is not a promise of convergence. Actual recovery/profile peak allocated/reserved **{r['peak_allocated_bytes_recovery_profile']/2**20:.1f}/{r['peak_reserved_bytes_recovery_profile']/2**20:.1f}MiB**; original400-step run peak was not recovered after the export error and is not fabricated.

New total charged sessions{ledger['charged_seconds']:.3f}s; lifetime **{total/60:.3f}/60 minutes**, leaving{remaining/60:.2f} minutes. Download ledger **{read_json(ROOT/'experiments/download_ledger.json')['transferred_bytes']:,}/10,000,000,000 bytes**. No reset. `gpu_budget.json` preserves AMP failure, FP32 fitting and metric recovery reservations/actuals.

**Approval proposal:** raise cumulative GPU limit to **240 minutes** (additional180 minutes above the old cap), and authorize a new frozen FP32 protocol allowing at most5% per-class colliding target objects with the existing conflicting-regression masks, retaining every GT object in evaluation. The proposed relaxation is explicit and unapproved; it does not resolve unidentifiable coincident objects. Alternative: redesign collision representation or generate independently validated collision-controlled training data under another protocol. Do not silently overwrite v1 or treat a budget increase alone as target-policy approval. Details: `budget_proposal.json`.

## Verification, commands and next step

**{passed} tests passed** including historical tests; one existing PyG deprecation warning. **{integrity['protected_files']} historical files unchanged.** Scripts and frozen hashes remain in this namespace. `diagnostic_learning_curve.png` marks the single-chart/four-chart transition; it is not an epoch validation curve. Full curriculum checkpoint, full validation metrics and convergence curves are **unavailable** until gates and budget authorization are resolved.

```powershell
$python = '.\\.venv\\Scripts\\python.exe'
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
'''
    (ROOT/'research/perception_training_report.md').write_text(report,encoding='utf-8')
    (ROOT/'PROJECT_STATUS.md').write_text(f'''# SenseGraph project status

Visual curriculum **incomplete; awaiting target-policy and GPU-budget approval**. Historical milestones1–3 and failed M4 detector remain preserved.

- Required official ResNet18 weights acquired/hash-verified; all1,400 train and300 validation target records scanned; zero invalid supported boxes, small targets retained.
- v1 AMP smoke failed nonfinite gradient. Separate FP32 diagnostic genuinely trained100 single-chart+300 four-chart updates; tiny-set macro recall{r['ap50']['macro_recall']:.4f}; exact checkpoint restore. This is training-only wiring evidence, not validation improvement.
- Full campaign has not started: 2.7–3.0% train/3.1–3.9% validation point/segment center collisions exceed the declared1% gate; measured campaign projection{r['campaign_estimated_seconds']/60:.1f}min exceeds{remaining/60:.2f}min remaining.
- {passed} tests passed; {integrity['protected_files']} historical files unchanged. Lifetime GPU time{total/60:.3f}/60min; download total{read_json(ROOT/'experiments/download_ledger.json')['transferred_bytes']:,}/10GB.
- Proposal: explicit new FP32 protocol with≤5% collision fraction, all GT evaluation denominators preserved, and cumulative GPU cap240min. No limits relaxed yet.
- [Training report](research/perception_training_report.md), [architecture](research/neural_model_architecture.md). Artifacts: `experiments/visual_training/v1/`. No GAT/QA/frontend training.
''',encoding='utf-8')
    print({'status':'approval_required','tests':passed,'lifetime_gpu_minutes':total/60,'campaign_estimated_minutes':r['campaign_estimated_seconds']/60})


if __name__=='__main__': main()
