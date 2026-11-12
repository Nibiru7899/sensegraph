"""Combined autonomous-quality report; no model evaluation or selection side effects."""
from pathlib import Path
import sys,time,csv
import xml.etree.ElementTree as ET
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json


def main():
    base=ROOT/'experiments/visual_training'; release=base/'release'; release.mkdir(parents=True,exist_ok=True)
    runs={}; rows=[]
    for version in ['v2','v3']:
        d=base/version; history=read_json(d/'history.json') if (d/'history.json').exists() else []
        state=read_json(d/'run_status.json')['status'] if (d/'run_status.json').exists() else 'running'
        best=max(history,key=lambda r:r['validation']['ap50']['macro_ap']) if history else None
        runs[version]={'status':state,'epochs':len(history),'history':history,'best':best}
        for r in history:
            rows.append({'run':version,'stage':r['stage'],'epoch':r['epoch'],'train_loss':r['train_loss'],'val_loss':r['validation']['loss'],'val_ap50':r['validation']['ap50']['macro_ap'],'val_ap75':r['validation']['ap75']['macro_ap'],'optimizer_steps':r['optimizer_steps'],'peak_allocated_bytes':r['peak_allocated_bytes'],'peak_reserved_bytes':r['peak_reserved_bytes']})
    if rows:
        with (release/'campaign_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        figure=Figure(figsize=(12,8)); FigureCanvasAgg(figure); axes=figure.subplots(2,2)
        for i,(name,run) in enumerate(runs.items()):
            h=run['history']; x=range(1,len(h)+1)
            axes[i,0].plot(x,[r['train_loss'] for r in h],label='train loss'); axes[i,0].plot(x,[r['validation']['loss'] for r in h],label='validation loss'); axes[i,0].set(title=name+' losses',xlabel='Complete epoch'); axes[i,0].legend()
            axes[i,1].plot(x,[r['validation']['ap50']['macro_ap'] for r in h],label='AP50'); axes[i,1].plot(x,[r['validation']['ap75']['macro_ap'] for r in h],label='AP75'); axes[i,1].set(title=name+' all300 validation',xlabel='Complete epoch',ylim=(0,1)); axes[i,1].legend()
        figure.tight_layout(); figure.savefig(release/'training_comparison.png')
    audit=read_json(base/'v2/integrity_audit.json'); corrected=read_json(base/'v3/target_fix_audit.json')
    test_path=release/'tests.xml' if (release/'tests.xml').exists() else base/'v3/tests.xml'
    tests=ET.parse(test_path); assert not tests.findall('.//failure') and not tests.findall('.//error'); count=len(tests.findall('.//testcase'))
    ledger_path=base/'v3/gpu_budget_awake.json' if (base/'v3/gpu_budget_awake.json').exists() else base/'v3/gpu_budget.json'
    ledger=read_json(ledger_path) if ledger_path.exists() else None
    total=None; in_flight=False
    if ledger:
        actual=sum(s['actual_seconds'] if s['actual_seconds'] is not None else time.time()-s['start_unix'] for s in ledger['sessions']); total=ledger['historical_seconds']+actual; in_flight=any(s['actual_seconds'] is None for s in ledger['sessions'])
    holdout=base/'independent_holdout'; results=read_json(holdout/'results.json') if (holdout/'results.json').exists() else None
    if (holdout/'gpu_budget.json').exists():
        l=read_json(holdout/'gpu_budget.json'); total=l['historical_seconds']+(l['actual_seconds'] if l['actual_seconds'] is not None else 0); in_flight=l['actual_seconds'] is None
    for stage in ['ocr_v1','ocr_holdout','graph_transfer','real_partial','postselection_verification']:
        if (base/stage/'gpu_budget.json').exists():
            l=read_json(base/stage/'gpu_budget.json'); total=l['historical_seconds']+(l['actual_seconds'] if l['actual_seconds'] is not None else l['charged_seconds']); in_flight=l['actual_seconds'] is None
    ocr=read_json(base/'ocr_v1/result.json') if (base/'ocr_v1/result.json').exists() else None
    ocr_test=read_json(base/'ocr_holdout/results.json') if (base/'ocr_holdout/results.json').exists() else None
    graph=read_json(base/'graph_transfer/results.json') if (base/'graph_transfer/results.json').exists() else None
    real=read_json(base/'real_partial/results.json') if (base/'real_partial/results.json').exists() else None
    verification=read_json(base/'postselection_verification/results.json') if (base/'postselection_verification/results.json').exists() else None
    extra='OCR, frozen graph transfer and partial real-image audits have not all completed; no unexecuted stage is claimed as a result.'
    if not ocr and (base/'ocr_v1/smoke.json').exists() and not read_json(base/'ocr_v1/smoke.json')['passed']:
        extra='OCR training-only memorization gate failed. Full OCR training and independent OCR audit were not run; inspect `ocr_v1/smoke.json` for actual metrics/failure evidence.'
    if ocr:
        extra=f"OCR: {ocr['status']}; {ocr['completed_epochs']} complete epochs, selected epoch{ocr['selected_epoch']} by validation CER{ocr['validation']['cer']:.4f}, exact-word{ocr['validation']['exact']:.4f}. All327,488 OCR parameters train independently; no vision/graph/QA updates. Cache contains21,655 training/4,682 validation crops, no infeasible targets or exact cross-split pixel duplicates.109 of110 validation strings occur in training: this is closed synthetic vocabulary recognition, **not unseen-word OCR**. Geometry-only90-degree rotation does not guarantee arbitrary slanted-text recognition."
        extra+=' CUDA explicitly warned that CTC backward lacks a deterministic implementation under warn-only policy. Seeds/checkpoints are recorded; bit-for-bit training replay is not guaranteed. Selected validation title exact accuracy and geometry-rotated-label exact accuracy are both0, so OCR is an experimental branch, not reliable general text extraction.'
        h=read_json(base/'ocr_v1/history.json'); fig=Figure(figsize=(10,4)); FigureCanvasAgg(fig); ax=fig.subplots(1,2); x=range(1,len(h)+1)
        ax[0].plot(x,[r['train_loss'] for r in h]); ax[0].set(title='OCR train CTC',xlabel='Epoch'); ax[1].plot(x,[r['validation']['cer'] for r in h],label='validation CER'); ax[1].plot(x,[1-r['validation']['exact'] for r in h],label='word error'); ax[1].set(xlabel='Epoch'); ax[1].legend(); fig.tight_layout(); fig.savefig(release/'ocr_curves.png')
    if ocr_test:
        cond=ocr_test['oracle_gt_crop_recognition_DIAGNOSTIC']; end=ocr_test['end_to_end']
        extra+=f"\n\nSelected OCR training-crop CER{ocr_test['training_crop_recognition_DIAGNOSTIC']['cer']:.4f} versus validation CER{ocr_test['selected_validation_crop_recognition']['cer']:.4f}; training recognition is a memorization/overfitting diagnostic, not generalization evidence."
        extra+=f" Holdout vocabulary overlap: {cond['vocabulary_overlap']}; no lexicon is used for decoding. An unseen-transcription subset with zero samples is unavailable evidence, not zero-error generalization."
        extra+=f"\n\nIndependent OCR: oracle-GT-crop CER{cond['cer']:.4f}, exact{cond['exact']:.4f}; image-only detected-region exact-word precision{end['word_precision']:.4f}, recall{end['word_recall']:.4f}, F1{end['word_f1']:.4f}, end-to-end CER{end['cer_including_missing_and_extra_regions']:.4f}. All{end['gt_words']} source text/legend regions remain in the denominator. Missing regions are deletions, extra text is insertion; matched-only CER is a separate diagnostic. No GT strings/crops/types enter recognition. Numeric, rotated and role subsets plus raw cases are in `ocr_holdout/`. No test-driven retraining or lexicon corrections. `selected_with_ocr.pt` changes only OCR tensors of the selected vision model."
    if graph:
        extra+=f"\n\nFrozen graph audit complete: **{graph['complete']}**. Any failed charts remain FN in denominators; incomplete audits do not receive paired comparative claims."
        extra+='\n\nFrozen legacy graph transfer (NO graph retraining):\n\n| Model | Predicted proposals macro F1, mean3 seeds | Oracle proposals diagnostic macro F1, mean3 seeds |\n| --- | ---: | ---: |\n'
        for arch in ['gat','mlp']:
            values=[sum(graph['runs'][f'{arch}_s{s}_{mode}']['end_to_end']['macro_f1'] for s in [42,43,44])/3 for mode in ['predicted','oracle_diagnostic']]
            extra+=f'| {arch} | {values[0]:.4f} | {values[1]:.4f} |\n'
        extra+='\nOnly frozen oracle-trained11-feature geometry/RGB models and original validation thresholds are transferred. Conditional matched-node scores and paired chart-bootstrap intervals are separate in `graph_transfer/results.json`. Missing virtual series endpoints remain end-to-end FN. This does not train the proposed multimodal graph stack or establish a matched-learning/real-world GAT advantage.'
        if graph['complete']:
            paired=graph['paired_chart_intervals']['predicted_end']['macro_f1']
            extra+=f" Paired macro difference{paired['gat_minus_mlp']:.4f}, chart-bootstrap95% interval{paired['ci95']}; conditional on these three frozen seeds, detector and synthetic generator, not new training-seed/real-domain uncertainty."
        conflicts=[r['numeric_mutual_exclusion_violations']['contradictory_pairs'] for name,r in graph['runs'].items() if name.startswith('gat') and name.endswith('_predicted')]
        extra+=f" GAT numeric mutual-exclusion violations by seed: {conflicts} across495,014 directed candidate pairs per seed; these raw predictions are unchanged. Learned comparisons are not verified arithmetic."
    if real:
        extra+='\n\nTwenty existing ChartQA **training** images, supported partial-label recall only:\n\n| Provided class | Recovered / supported boxes | Recall50 |\n| --- | ---: | ---: |\n'
        extra+='\n'.join(f"| {name} | {row['recovered']} / {row['source_supported_boxes']} | {row['recall50']:.4f} |" for name,row in real['by_source_supported_class'].items())
        extra+='\n\nThese twenty training examples are not an independent benchmark test. Unknown labels cannot define false positives, full precision/AP or complete recall; none is claimed. Horizontal bars/real typography are outside synthetic support. VisText raster alignment stays unavailable; no annotations were invented. No real-image adaptation or threshold/model tuning occurred.'
    if verification:
        extra+='\n\nPost-selection GPU verification (TRAIN/VAL only):\n\n| Arithmetic | Full validation AP50 | One TRAIN image repeat exact | Reload exact |\n| --- | ---: | --- | --- |\n'
        extra+='\n'.join(f"| {name} | {r['validation']['ap50']['macro_ap']:.6f} | {r['training_image_repeat_exact']} | {r['training_image_reload_exact']} |" for name,r in verification['checks'].items())
        extra+=f"\n\nOriginal strict selected-epoch validation AP50 exactly reproduced: **{verification['strict_validation_ap50_exactly_reproduced']}**. In-process flags, memory and raw predictions are saved. One-case same-host equality is bounded evidence; independent holdout scores/selection remain unchanged."
    readiness='Training/audits are ongoing; no real-world readiness is claimed.'
    if real and ocr_test:
        supported=sum(r['source_supported_boxes'] for r in real['by_source_supported_class'].values()); recovered=sum(r['recovered'] for r in real['by_source_supported_class'].values())
        readiness=f"**Readiness: not ready for reliable real-chart QA.** Synthetic detection is learned, but only {recovered}/{supported} supplied valid real-image boxes were recovered on 20 training-source images. OCR detected-region word F1 is {ocr_test['end_to_end']['word_f1']:.4f}. These are observed quality blockers, not successful completion of chart understanding. No learned QA/calibration/new multimodal graph training or frontend has been added."
    run_table='\n'.join(f"| {name} | {r['status']} | {r['epochs']} | {r['best']['stage']+' '+str(r['best']['epoch']) if r['best'] else 'unavailable'} | {r['best']['validation']['ap50']['macro_ap'] if r['best'] else 'unavailable'} | {r['best']['validation']['ap75']['macro_ap'] if r['best'] else 'unavailable'} | {max((x['peak_allocated_bytes'] for x in r['history']),default=0)/2**30:.3f} / {max((x['peak_reserved_bytes'] for x in r['history']),default=0)/2**30:.3f} |" for name,r in runs.items())
    independent='Independent holdout is frozen but has not been evaluated. No unseen-holdout result is claimed.'
    if results:
        independent=f"Selection locked before inference: **{results['selected_before_inference']}**. Complete three-model audit: **{results['complete']}**."
        independent+='\n\n| Model | Fresh holdout AP50 | AP75 | Recall50 |\n| --- | ---: | ---: | ---: |\n'
        independent+='\n'.join(f"| {name} | {m['ap50']['macro_ap']:.4f} | {m['ap75']['macro_ap']:.4f} | {m['ap50']['macro_recall']:.4f} |" for name,m in results['results'].items())
        independent+='\n\nThe selected model is NOT changed if another model scores higher here. These are new synthetic fonts/styles/configurations, not real-world data. The old detector retains its original320 input, min score .01 and cap100; new models use640, min .10 and cap128. All use identical physical GT/classes and .25 operating threshold. Training exposure differs substantially; this comparison does not isolate architecture. Keypoint5px metrics use new heatmap centers versus old box centers and are reported separately in each metric JSON.'
        if results.get('selected_training_DIAGNOSTIC'):
            independent+=f"\n\nSelected checkpoint's full1,400-chart training AP50 is{results['selected_training_DIAGNOSTIC']['ap50']['macro_ap']:.4f}, measured only as an overfitting diagnostic after selection. Compare with selected validation and audit AP, not as independent quality evidence. No updates or selection changes follow it."
        independent+='\n\nPrecision disclosure: training/epoch validation disabled TF32 and enabled deterministic algorithms with warn-only fallback. The one-time independent evaluator left PyTorch2.8 defaults: float32 tensors/no autocast, cuDNN TF32 allowed, matmul TF32 disabled, deterministic-algorithm enforcement off. All three audit models share that setting. CPU import-equivalent flags/source hash are recorded in `precision_configuration_audit.json`; individual GPU kernels were not traced. These are **not a strict-TF32-disabled rerun**. Preserve these first results; do not rerun/reselect on the holdout. Post-selection validation-only checks quantify arithmetic repeatability separately.'
    text=f'''# Full visual perception training and autonomous quality audit

Snapshot: {time.strftime('%Y-%m-%d %H:%M:%S %z')}. This report supersedes the prior live snapshot; the original v2 report is preserved at `experiments/visual_training/v2/report_snapshot.md`. **No guarantee of perfection, absence of all leakage, or zero overfitting is possible.** Actual findings and limitations follow.

{readiness}

## Completed and ongoing training

| Visual run | Status | Complete epochs | Selected stage/epoch | Validation AP50 | AP75 | Peak allocated / reserved GiB |
| --- | --- | ---: | --- | ---: | ---: | ---: |
{run_table}

Both visual runs use all1,400 training and300 validation charts, fresh required ImageNet1K_V1 ResNet18 plus128-channel FPN and the exact six perception heads. FP32, segmentation disabled, five warmup/up to30 C4/C5 epochs, frozen BN, AdamW3e-4/1e-5, weight decay1e-4, accumulation8, clip1, fixed LR schedule and validation-only selection. Early stopping uses5 epochs without meaningful AP50 improvement after minimum5 fine-tune epochs. No GAT/reasoner/frontend training and no old protected-test model evaluation. OCR is a separate independently supervised stage described below. See frozen protocols for exact settings.

v2 genuinely completed five warmup+13 fine-tuning epochs and stopped at3,150 optimizer updates. Its selected checkpoint is fine-tuning8, not the final overfit model. Training loss continued falling while validation AP fell; early stopping mitigates further fitting but **does not establish zero residual overfitting**. A normal planning-window pause resumed from step2,879/cursor632 and reached the exact18-epoch update count without replaying committed batches. Source snapshots, original/continuation logs and separate actual reservations are preserved.

v3 is a single-factor implementation correction, not an architecture search: max-union Gaussian halos over colliding instances. Original target halos were annotation-order dependent on8/1,700 development charts. Corrected fixed-order targets differ only on{corrected['changes_by_split'].get('train',0)} train and{corrected['changes_by_split'].get('val',0)} validation charts. All positive centers, boxes, counts and regression/valid masks remain identical; all1,700 records pass reversed-order invariance. Two runs with one seed cannot establish a robust statistical improvement. Select highest complete validation AP50; exact cross-run ties prefer corrected v3. No holdout-based selection.

## Independent audit-only evaluation

{independent}

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

{extra}

## Resource accounting and verification

Latest available cumulative actual/elapsed GPU-session estimate: **{total/60 if total is not None else 'unavailable'} minutes**; active session: **{in_flight}**. In-flight reservations are conservative, not reported as completed execution. Five-hour ceiling is cumulative18,000 seconds, including all historical M1–M4/v1/v2/v3 time; no reset. The resource-only continuation changed planning time alone under explicit user authorization. v3 guards its own allocation plus600 evaluation seconds below that ceiling. No new model downloads beyond counted official weights; download ledger remains{read_json(ROOT/'experiments/download_ledger.json')['transferred_bytes']:,}/10,000,000,000 bytes.

An overnight Windows idle-sleep/hibernate interrupted v3 at committed step2,090/cursor1,320 in fine-tuning7. The original wall-time ledger is preserved unchanged. Windows power/clock records independently evidence14,514.172 non-execution seconds, conservatively subtracted in a **separate** reconciliation; later sleep outside that session is not subtracted. Resume uses the exact optimizer/RNG/cursor, unchanged scientific sources/configuration, Windows working-state time and a temporary idle-sleep hold. See `research/resource_reconciliation.md`, original power-event XML and both ledgers. Charged working-state sessions include CPU/loading overhead and upper-bound actual GPU execution; they are not kernel-only profiling.

**{count} tests passed**, one existing PyG deprecation warning. Tests include complete existing baselines, image/target transforms, tiny targets, collisions, finite loss/gradients, directed graph boundaries, decoder equivalence, exact CPU optimizer/RNG/cursor resume, corrected halo invariance, validation-only tie selection, repeated-holdout refusal and five-hour guard. Historical and new source hashes must pass final verification; CPU audits are not model quality evidence. Full hardware-failure fault injection and exhaustive pretraining/source provenance are unexecuted.

## Reproduction and next boundary

```powershell
$python = '.\\.venv\\Scripts\\python.exe'
& $python -m sensegraph.corrected_visual_training verify
& $python -m sensegraph.corrected_visual_training run  # only resume an incomplete corrected run
& $python scripts/report_visual_quality.py
# Once locked selection/evaluation exists, CPU image-only inference:
& $python scripts/infer_full_visual.py --image data/synthetic/v3/images/synthetic-01400.png --output experiments/visual_training/release/inference.json
```

Independent evaluation command: `python scripts/evaluate_visual_holdout.py`; it is one-time and refuses existing attempts. Its config, model-selection record and outputs are retained; do not erase them to rerun. No new deployment or frontend. Deliver the selected visual checkpoint and truthful model card; graph-on-imperfect-proposal training, numerical calibration/QA and user-facing accessibility remain later validated research stages, not completed by detector metrics.
'''
    (ROOT/'research/full_perception_training_report.md').write_text(text,encoding='utf-8')
    status='quality audits finished' if all([results and results['complete'],ocr_test,graph and graph['complete'],real and real['complete'],verification and verification['complete']]) else 'visual audit finished; additional checks pending or incomplete' if results else 'corrected comparison in progress' if 'running' in runs['v3']['status'] else 'training done; independent audit pending'
    (ROOT/'PROJECT_STATUS.md').write_text(f'''# SenseGraph project status

Autonomous visual quality phase: **{status}**. Five-hour cumulative GPU ceiling; existing data/experiments preserved.

{readiness}

- Original full visual run:18 complete epochs, early-stopped; selected fine-tuning8 AP50 .898491 on300 validation charts.
- Corrected collision-target run: {runs['v3']['status']}, {runs['v3']['epochs']} complete epochs. Only Gaussian union correction; no architecture/threshold search.
- {count} tests pass. Image/split/configuration audit found no identity leakage; known near-hash candidates and validation reuse are explicitly documented.
- Independent300-chart new-font holdout frozen; selected model is locked before inference, with no result-driven reselection.
- [Full report](research/full_perception_training_report.md), [architecture](research/neural_model_architecture.md). All histories/checkpoints/audits under `experiments/visual_training/`.
- Preserved milestones: frozen100-question ChartQA pilot UniChart65/100, DePlot deterministic executor12/100; oracle-proposal GAT macro F1 .7629 versus MLP .7409. These historical metrics describe different tasks/data and are not comparable with detection AP.
- Baseline reproduction remains documented in `research/baseline_report.md`; structural mappings/partial supervision in `research/real_annotation_mapping.md`. No historical artifact is replaced by the new quality audit.
- OCR stage: {ocr['status'] if ocr else 'not completed'}; detected-region independent word F1: {ocr_test['end_to_end']['word_f1'] if ocr_test else 'unavailable'}.
- Frozen graph transfer: {'executed; no retraining' if graph else 'pending'}; real partial-label recall audit: {'executed on20 training images' if real else 'pending'}.
- Charged GPU-work sessions: {total/60:.2f} / 300 minutes; all sessions closed: {not in_flight}. Windows suspension reconciliation preserves the original ledger and event evidence. Both visual/OCR fits stopped on declared convergence rules; unused time is not filled with post-holdout training.
- No claim of perfect/no-overfit performance. Multimodal graph training, learned QA/calibration and frontend remain untrained/deferred.
''',encoding='utf-8')
    write_json(release/'quality_summary.json',{'status':status,'tests':count,'training_runs':{k:{'status':v['status'],'complete_epochs':v['epochs'],'best_ap50':v['best']['validation']['ap50']['macro_ap'] if v['best'] else None} for k,v in runs.items()},'cumulative_gpu_seconds_estimated':total,'gpu_session_in_flight':in_flight,'holdout_complete':results['complete'] if results else False,'ocr_complete':bool(ocr_test),'graph_transfer_complete':graph['complete'] if graph else False,'real_partial_complete':real['complete'] if real else False,'postselection_verification_complete':bool(verification and verification['complete']),'ready_for_reliable_real_chart_qa':False,'blockers':['real-domain perception transfer','OCR generalization','untrained new multimodal graph/QA and numerical calibration']})
    print({'status':status,'tests':count,'holdout_evaluated':bool(results)})


if __name__=='__main__': main()
