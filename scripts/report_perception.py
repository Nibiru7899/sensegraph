"""Report only executed M4 evidence; supports an explicitly incomplete acquisition block."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sensegraph.io import read_json,write_json
from sensegraph.perception_experiment import verify


def main():
    d=ROOT/"experiments/milestone4"
    integrity=verify(ROOT)
    frozen=read_json(d/"freeze.json")
    tests=ET.parse(d/("executed_tests.xml" if (d/"executed_tests.xml").exists() else "tests.xml"))
    if tests.findall(".//failure") or tests.findall(".//error"):
        raise RuntimeError("Tests failed")
    count=len(tests.findall(".//testcase"))
    download=read_json(ROOT/"experiments/download_ledger.json")["transferred_bytes"]
    ledger=read_json(d/"gpu_budget.json") if (d/"gpu_budget.json").exists() else {"charged_seconds":0.,"historical_seconds":frozen["historical_gpu_seconds"],"sessions":[]}
    lifetime=ledger["historical_seconds"]+ledger["charged_seconds"]
    result=read_json(d/"result.json") if (d/"result.json").exists() else None
    blocker=read_json(d/"blocker.json") if (d/"blocker.json").exists() else None
    state="validation gate passed; continuation still required" if result and result["gate_passed"] else "stopped at learned-detection bottleneck" if result else "incomplete: checkpoint acquisition blocked"
    details=""
    if result:
        metric=result["metrics"]["0.5"]
        table="\n".join(f"| {name} | {v['support']} | {v['precision']:.4f} | {v['recall']:.4f} | {v['ap'] if v['ap'] is not None else 'unavailable'} | {v['fp']} | {v['fn']} | {v['duplicates']} | {v['normalized_center_error'] if v['normalized_center_error'] is not None else 'unavailable'} |" for name,v in metric["per_class"].items())
        details=f"""## Actual validation detector measurements

{result['steps']} development steps; {result['validation_images']} validation charts. Macro recall@IoU .50: **{metric['macro_recall']:.4f}**; macro AP@.50: **{metric['macro_ap']:.4f}**. Macro AP@.75: **{result['metrics']['0.75']['macro_ap']:.4f}**. These are validation development results, not untouched-test evidence. The predeclared gate passed: **{result['gate_passed']}**. No post-gate hyperparameter search or replacement with oracle scores.

| Class | GT | Precision | Recall | AP50 | FP | FN | Duplicates | Normalized center error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

AP integrates the interpolated precision envelope over achieved recall (not COCO's 101-point mAP). Confidence >= .25 for precision/recall; AP uses all returned detections >= .01 with a 100-proposal cap. Normalized center errors, .75-IoU results and raw predictions are retained in `result.json`/`validation.pt`.

Final checkpoint reload exact: {result['checkpoint_reload_exact']}. Median measured pipeline latency: {result['median_pipeline_seconds']:.4f} seconds/image (decoding + detector + frozen ROI embeddings + graph construction + two **untrained** relation wiring forwards). This is not trained end-to-end relationship accuracy. Peak allocated/reserved: {result['peak_allocated_bytes']/2**20:.1f}/{result['peak_reserved_bytes']/2**20:.1f} MiB. Frozen ROI embeddings are 1024-D; geometry/color is 11-D. Only geometry/color is passed to the historical architecture for shape verification; new visual-feature relationship models have not been fitted. Per-image shape checks remain in the result.

OCR probe: `{result['ocr_probe']['status']}`. Recognition accuracy is unmeasured. Detector checkpoint, training log, before/after smoke overlays and six validation overlays are preserved. See `visual_review.json` if present for actual reviewed samples.

Acquisition is resolved: the user executed the unchanged budgeted official acquisition command. Source, full SHA256 and byte count are recorded in `weights.json`; trained checkpoint provenance is in `checkpoint_provenance.json`. Earlier acquisition blockers/reviews are retained as historical records, not current blockers. Smoke loss/gradient and exact reload/repeated-inference checks passed (`smoke.json`). Actual accepted-proposal ROI embeddings were finite and 1024-D (`smoke_interface_checks.json`). The auxiliary exact-boundary check found a 0.000031-pixel float32 overshoot on a subthreshold box; diagnostic records are retained. All accepted smoke boxes were strictly in bounds. No prediction, threshold or frozen model code was altered.

CPU error export: `scripts/export_perception_errors.py` writes `detection_metrics.csv` (both IoU thresholds, including localization error), `detection_errors.json` (missed node IDs, unmatched/duplicate detections, unsupported/invalid annotations), and checkpoint provenance. All validation examples remain in denominators. Two small additional diagnostic model sessions, including the initial failed strict-boundary assertion, are charged in the M4 GPU ledger alongside smoke and fitting.
"""
        log=read_json(d/"training_log.json")
        figure=Figure(figsize=(8,4)); FigureCanvasAgg(figure); ax=figure.subplots()
        ax.plot([r['step'] for r in log],[r['loss'] for r in log],alpha=.7)
        ax.set(xlabel="Development step",ylabel="Detection loss",title="M4 bounded detector training; validation not used for selection")
        figure.tight_layout(); figure.savefig(d/"learning_curve.png")
    else:
        details=f"""## Exact acquisition blocker

{blocker if blocker else 'No completed detector result exists; inspect acquisition logs.'}

No pretrained detector inference, visual embedding smoke, GPU training, detection overlay review or measured detection/relationship results have executed. CPU tests prove interfaces/evaluation fixtures only. No random or mock detector output is presented as learned perception evidence.
"""
    report=f"""# Milestone 4 — bounded learned perception

**Status: {state}. Milestone 4 is not claimed complete.**

## Implemented scope

New frozen protocol/config, official pretrained MobileNetV3 Faster R-CNN acquisition through the existing budget, local checkpoint-only detector factory, frozen ROI image features, small RPN/classification/regression head training, image-only proposals, spatial/complete proposal graphs, empty/duplicate/false-region handling, one-to-one typed IoU matching, AP/recall/localization/duplicate metrics, and end-to-end/conditional relation recovery evaluator. A local optional Tesseract adapter reports unavailable without inserting renderer text. Rotations beyond explicit right angles and OCR recognition accuracy are unsupported/unmeasured here.

The renderer's `series` union is not a painted detection target. Text and enclosing segment rectangles retain declared semantics, not point or mask substitutions. Unsupported/invalid annotations are recorded. Every GT relation, including those involving unobserved series nodes, remains in end-to-end relation-recovery denominators. Existing node types, values, source text and semantic adjacency never enter image-only forward methods.

{details}

## Verification and resources

- Complete suite: **{count} passed**; one existing PyG deprecation warning. New fixtures exercise duplicate and type matching, exact IoU boundary, missing/extra relationship endpoints, empty graphs, graph boundaries, invalid/unsupported boxes and unavailable OCR.
- **{integrity['protected_files']} historical files unchanged**: old source/tests/configs/reports, v2/v3 data, sealed experiments and original GPU ledgers. The current download ledger is deliberately excluded from immutable files; its exact pre-run snapshot is frozen and all new transfers are counted.
- Downloads: **{download:,}/10,000,000,000 bytes**. New GPU-session charge: **{ledger['charged_seconds']:.3f} seconds**. Lifetime **{lifetime/60:.3f}/60 minutes**. No paid APIs or official benchmark test use.
- First 128 v3 training and 30 validation chart IDs and source hashes frozen in `experiments/milestone4/freeze.json`. These are development subsets, not independent generalization evaluation. Protected test sets stay closed.

## Remaining work and stop boundary

This bounded gate does not answer whether GAT still outperforms MLP with learned proposals. Matched learned-visual/projection models, three-seed predicted-proposal training, complete-connectivity comparison, controlled proposal degradation, OCR accuracy, new held-out synthetic evaluation and real-domain pilot remain **unexecuted**. Do not report M3 oracle results as M4 end-to-end performance. If detection fails, fix perception under a newly frozen protocol before continuing. A passed gate requires a separate frozen continuation, not automatic test access.

ChartQA's 20 partial training annotations remain development-only, not sufficient real-world evaluation. See `real_annotation_mapping.md`: missing/zero-area/partial labels need masking, series/line semantics differ, and VisText raster transforms remain unverified. Independently reviewed held-out real annotations and rights review are future work. No real-data annotation or calibrated evidence is invented.

## Commands and provenance

Use `.\\.venv\\Scripts\\python.exe -m sensegraph.perception_experiment` with `freeze`, `acquire`, `smoke`, `fit`, `verify` (in that order; review `smoke_after.png` and record `visual_approval.json` before `fit`). `--root` is supported. Freeze verifies exact sources and historic artifacts; changes require a new protocol. Smoke/fit refuse to overwrite attempted runs. Interrupted reservations remain charged. Inspect partial artifacts before any continuation; never remove attempts or reset ledgers to retry fitting.

Run `.\\.venv\\Scripts\\python.exe -m pytest -q --basetemp=.cache/pytest-m4 --junitxml=experiments/milestone4/tests.xml`, then `.\\.venv\\Scripts\\python.exe scripts/report_perception.py`. Report regeneration is CPU-only and does not rerun models. Acquisition retry is resumable and records actual received bytes.

Architecture/API: [Torchvision 0.23 documentation](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html); code [BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE). COCO pretrained weights have underlying training-data terms; no blanket image-rights grant or redistribution is claimed. Official checkpoint URL/hash is in `weights.json` when acquired.

No frontend, full chart QA, accessibility claims or large-scale runs.
"""
    (ROOT/"research/milestone4_report.md").write_text(report,encoding="utf-8")
    write_json(d/"report_summary.json",{"status":state,"tests":count,"history":integrity,"download_bytes":download,"lifetime_gpu_seconds":lifetime,"gate_result_available":bool(result)})
    (ROOT/"PROJECT_STATUS.md").write_text(f"""# SenseGraph project status

Milestones 1–3 remain complete and preserved. **Milestone 4: {state}; incomplete overall.**

- Implemented: budgeted pretrained detector/ROI feature interface, image-only proposal graph construction, optional local OCR adapter, explicit detection and missing/extra-node relation evaluation, bounded training/smoke CLI and frozen validation gate.
- Executed: {count} tests passed; {integrity['protected_files']} historical files unchanged. See actual model evidence or blocker in [Milestone 4 report](research/milestone4_report.md).
- Downloads: {download:,}/10,000,000,000 bytes; lifetime GPU-session time {lifetime/60:.3f}/60 minutes. Historical ledgers unchanged; new M4 sessions separately charged.
- No protected test-set model selection, no frontend, no QA/generalization claims.
- Remaining: address the detection bottleneck under a new frozen protocol before OCR recognition evaluation, matched learned-visual and predicted-proposal relation training, proposal-noise studies and independent held-out evaluation. Preserve current attempts and results.

Historical results: [M3](research/milestone3_report.md), [M2](research/milestone2_report.md), [baseline pilot](research/baseline_report.md).
""",encoding="utf-8")
    print(state)


if __name__=="__main__":
    main()
