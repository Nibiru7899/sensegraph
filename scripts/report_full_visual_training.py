"""CPU-only report from real full-split checkpoints and validation records."""
from pathlib import Path
import sys,time,csv
import xml.etree.ElementTree as ET
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json
from sensegraph.full_visual_training import verify


def main():
    d=ROOT/'experiments/visual_training/v2'; integrity=verify(ROOT)
    history=read_json(d/'history.json') if (d/'history.json').exists() else []
    initial=read_json(d/'initial_metrics.json') if (d/'initial_metrics.json').exists() else None
    ledger=read_json(d/'gpu_budget.json'); status=read_json(d/'run_status.json')['status'] if (d/'run_status.json').exists() else 'running'
    elapsed=sum(s['actual_seconds'] if s['actual_seconds'] is not None else time.time()-s['start_unix'] for s in ledger['sessions'])
    tests=ET.parse(d/'tests.xml'); assert not tests.findall('.//failure') and not tests.findall('.//error'); count=len(tests.findall('.//testcase'))
    best=max(history,key=lambda h:h['validation']['ap50']['macro_ap']) if history else None
    if history:
        fig=Figure(figsize=(12,4)); FigureCanvasAgg(fig); axes=fig.subplots(1,2)
        axes[0].plot(range(1,len(history)+1),[r['train_loss'] for r in history],label='train')
        axes[0].plot(range(1,len(history)+1),[r['validation']['loss'] for r in history],label='validation'); axes[0].legend(); axes[0].set(xlabel='Completed epoch',ylabel='Detection loss')
        axes[1].plot(range(1,len(history)+1),[r['validation']['ap50']['macro_ap'] for r in history],label='macro AP50')
        axes[1].plot(range(1,len(history)+1),[r['validation']['ap50']['macro_recall'] for r in history],label='macro recall50'); axes[1].legend(); axes[1].set(xlabel='Completed epoch',ylabel='300-chart validation score',ylim=(0,1))
        fig.tight_layout(); fig.savefig(d/'learning_curves.png')
        rows=[{'stage':r['stage'],'epoch':r['epoch'],'train_loss':r['train_loss'],'val_loss':r['validation']['loss'],'ap50':r['validation']['ap50']['macro_ap'],'ap75':r['validation']['ap75']['macro_ap'],'recall50':r['validation']['ap50']['macro_recall'],'optimizer_steps':r['optimizer_steps'],'peak_allocated':r['peak_allocated_bytes'],'peak_reserved':r['peak_reserved_bytes']} for r in history]
        with (d/'epoch_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result='Initial300-chart validation is still running.'
    if best:
        v=best['validation']; classes=v['ap50']['per_class']; old=read_json(ROOT/'experiments/milestone4/result.json')['metrics']['0.5']
        class_rows=[{'checkpoint':f"{best['stage']}_{best['epoch']}",'metric':metric,'class':name,**values} for metric in ['ap50','ap75'] for name,values in v[metric]['per_class'].items()]
        with (d/'best_per_class_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(class_rows[0])); writer.writeheader(); writer.writerows(class_rows)
        table='\n'.join(f"| {k} | {x['positives'] if 'positives' in x else x['support']} | {x['precision']:.4f} | {x['recall']:.4f} | {x['ap'] if x['ap'] is not None else 'unavailable'} | {v['ap75']['per_class'][k]['ap']} | {x['fp']} | {x['fn']} | {x['normalized_center_error']} |" for k,x in classes.items())
        result=f'''Best completed validation checkpoint: **{best['stage']} epoch{best['epoch']}**, from{len(history)} completed epochs. All{v['images']} validation charts included. Macro AP50 **{v['ap50']['macro_ap']:.4f}**, AP75 **{v['ap75']['macro_ap']:.4f}**, recall50 **{v['ap50']['macro_recall']:.4f}**. These are validation-selected development scores, not protected-test results.

| Physical class | GT | Precision | Recall | AP50 | AP75 | FP | FN | Normalized center error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Explicit5-pixel keypoint results: `{v['keypoints_5px']}`. Enclosing-box AP and keypoint recovery have different geometrical tolerances and are not interchangeable. Chart-type results for bar/grouped-bar/line are retained in every epoch metric JSON. Validation style is held out from training, but is only one generator style bundle, not real-world generalization.

Aligned first30 comparison, same physical target inventory: new AP50 **{v['aligned_first30']['macro_ap']:.4f}**, recall50 **{v['aligned_first30']['macro_recall']:.4f}**; old detector AP50 **{old['macro_ap']:.4f}**, recall50 **{old['macro_recall']:.4f}**. Old detector had128-chart/240-step development at320 pixels; new model has1,400 training charts and640 input. A score difference does not isolate architecture from exposure, resolution or optimization. No old detector retraining or test usage.

Per-class AP50/AP75 including duplicate counts/localization are exported to `best_per_class_metrics.csv`. Class-conditioned chart-type AP averages only classes with GT support; the inherited chart-type macro recall averages all seven classes and assigns zero to absent classes. Its ceiling therefore differs by chart type; compare supported per-class recall rather than interpreting those macro recalls as directly comparable. Full300-chart and first30-chart metrics have all seven physical classes.
'''
        error_path=d/'best_error_analysis.json'
        if error_path.exists():
            analysis=read_json(error_path)
            if analysis['checkpoint']==f"{best['stage']}_{best['epoch']}":
                result+=f"\nPost-selection subgroup recovery: `{analysis['summary']}`. Tiny means native box area<1024; these are GT-recovery recalls, not subgroup AP/precision. All FP remain in the main metrics. `best_error_analysis.json` lists actual missed source nodes and unmatched detections; `best_confidence_distribution.png` displays returned scores, not calibrated probabilities.\n"
    elif initial:
        result=f"Initial all300-chart validation: AP50{initial['ap50']['macro_ap']:.4f}, recall50{initial['ap50']['macro_recall']:.4f}. No epoch completed yet."
    text=f'''# Full visual perception training — FP32 v2

**Actual run status: {status}. Completed epochs: {len(history)}.** Snapshot generated {time.strftime('%Y-%m-%d %H:%M:%S %z')}. Neither four-chart memorization nor a partial epoch is substituted for full-split evidence. Sources/protocol frozen before initial validation and optimization; see `full_visual_training_protocol.md`.

## Authorization, collision analysis and data

User explicitly authorized longer training (around1–1.5 hours, old60-minute limit revoked) and separately approved the5% class-specific collision exception. A90-minute new campaign planning window is recorded; historical ledgers are immutable and inherited usage remains counted. No automatic240-minute limit increase. New namespace `experiments/visual_training/v2`; v1 remains untouched.

Full1,400 train /300 validation v3 charts; no protected test access. Point collisions: train49 quantized groups/37 coincident-center groups, val14/7. Segment collisions: train62/18, val17/5. All collision examples are classified in `collision_analysis.json` and representative overlays retained. No invalid supported source boxes found by target audit. Coincidence refers to centers, not necessarily identical full instances. Ambiguous size/offset supervision alone is masked; heatmap presence and ALL ground-truth evaluation instances retained. Existing Gaussian target generation/architecture unchanged. Series aggregates are unsupported detector types, not fabricated positives.

## Executed results

{result}

Initial all300-chart metrics reside in `initial_metrics.json` when completed. Every completed epoch evaluates all300 charts, stores AP50/AP75/per-class FP/FN/localization,5-pixel point/tick metrics, confidence quantiles, chart-type subsets, predicted boxes and native image overlays. `history.json`, `epoch_metrics.csv` and `learning_curves.png` contain actual records when available. Initial/final prediction samples must be visually reviewed; losses alone are not success. Tiny-object/collision error breakdown is an additional diagnostic, not a tuning signal.

## Training and resume semantics

Required official ImageNet1K_V1 backbone verified against local provenance; newly initialized FPN/heads, no four-chart checkpoint transfer. FP32 only; no AMP retry. Five full warmup epochs with ResNet frozen; up to30 C4/C5 fine-tuning epochs, all BN moments and fine-tuned BN affine frozen, segmentation disabled. AdamW3e-4/new layers,1e-5/backbone, decay1e-4, accumulation8, clip1, linear first-stage-epoch warmup then cosine decay. Earliest best validation AP50 selection; stopping after minimum5 fine-tune epochs and5 epochs without max(.5% relative,1e-4 absolute) improvement. Stop at convergence/deterioration criterion or the planning window, never label window exhaustion convergence.

`last.pt` atomically commits model, optimizer, exact next-chart cursor, deterministic epoch order and RNG after each optimizer step. It supports resumption without replaying committed optimization. An abrupt interruption between an update and disk commit leaves an unavoidable uncommitted-update window; it remains charged as overhead. A prior unresolved reservation blocks unattended ledger reuse until dead-process/accounting reconciliation; never delete/reset that ledger. Local self-generated checkpoint uses pickle to restore NumPy/Python RNG; never load an untrusted external checkpoint this way.

```powershell
$python = '.\\.venv\\Scripts\\python.exe'
& $python -m sensegraph.full_visual_training verify
& $python -m sensegraph.full_visual_training run  # fresh run or resume committed cursor
& $python scripts/report_full_visual_training.py
```

Completed runs return their recorded status; no new optimization starts. Best weights: `best.pt`; per-epoch weight snapshots named by stage/epoch. Prediction decoder is vectorized with a tested numerical-equivalence fixture against v1. It does not consume annotations; GT is used afterward by the evaluator.

## Verification and accounting

**{count} tests passed**; one existing PyG deprecation warning. **{integrity['protected_files']} historical files unchanged.** Tests cover prior targets/loss/decoder checks plus vectorized decoding equivalence, early-stop boundaries, atomic cursor checkpoint roundtrip, and exact resumed AdamW/dropout/RNG equivalence on a CPU toy network without replaying committed steps. Full GPU campaign fault injection remains unexecuted; the live run was not interrupted to create this test.

Historical GPU time{ledger['historical_seconds']/60:.3f}min. Current actual elapsed estimate{elapsed/60:.3f}min; lifetime estimate{(ledger['historical_seconds']+elapsed)/60:.3f}min. In-flight reserved charge **{ledger['charged_seconds']/60:.3f}min** is conservative and is not claimed actual completed execution. Final actual sessions replace unused reservation only on normal exit. Epoch peak allocated/reserved values are PyTorch memory, not full device memory. Downloads remain{read_json(ROOT/'experiments/download_ledger.json')['transferred_bytes']:,}/10,000,000,000 bytes; no new acquisition for v2. All serialization/validation/CPU overhead is conservatively charged session wall time.

No GATv2/reasoning/OCR/frontend training. No real-world chart QA, robustness or accessibility claim. If performance is poor, inspect errors and overfitting before proposing a new controlled representation/resolution experiment; do not retroactively change this frozen run.
'''
    (ROOT/'research/full_perception_training_report.md').write_text(text,encoding='utf-8')
    (ROOT/'PROJECT_STATUS.md').write_text(f'''# SenseGraph project status

Full visual FP32 v2: **{status}**, completed full epochs **{len(history)}**. User authorized longer training and5% collision exception; old ledgers/protocols retained.

- All1,400 train/300 validation charts; required pretrained ResNet18/FPN; no GAT/QA/frontend training or protected test access.
- {count} tests passed; {integrity['protected_files']} historical files unchanged.
- Lifetime elapsed estimate{(ledger['historical_seconds']+elapsed)/60:.3f}min; current reserved campaign accounting retained separately from actual elapsed.
- [Full training report](research/full_perception_training_report.md); run/checkpoints `experiments/visual_training/v2`.
- Historical [diagnostics](research/perception_training_report.md), [M3](research/milestone3_report.md), [M4 detector](research/milestone4_report.md) remain unchanged.
''',encoding='utf-8')
    print({'status':status,'completed_epochs':len(history),'best_ap50':best['validation']['ap50']['macro_ap'] if best else None})


if __name__=='__main__': main()
