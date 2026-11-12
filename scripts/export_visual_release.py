"""CPU-only release audit and model card; no training or benchmark inference."""
from pathlib import Path
import sys,time,platform,subprocess
import torch,torchvision,torch_geometric
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph.neural_spec import SenseGraphNet
from sensegraph.visual_training import preprocess
from sensegraph.full_visual_training import decode_maps


def main():
    torch.set_num_threads(1); base=ROOT/'experiments/visual_training'; d=base/'release'
    selection=read_json(base/'independent_holdout/selection.json'); summary=read_json(d/'quality_summary.json')
    assert not summary['gpu_session_in_flight'] and summary['cumulative_gpu_seconds_estimated']<=18000
    winner=selection['selected']; source=ROOT/selection['candidates'][winner]['checkpoint']; checkpoint=d/'selected.pt'
    assert sha256(source)==sha256(checkpoint)==selection['candidates'][winner]['sha256']
    vision=torch.load(checkpoint,weights_only=True,map_location='cpu'); model=SenseGraphNet().eval(); model.load_state_dict(vision['model'],strict=True)
    stages={}
    for stage in ['warmup','perception','ocr','structural','reasoning','joint']:
        model.set_stage(stage); stages[stage]=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters()); scopes=['visual_perception']; ocr=None; ocr_quality='No trained OCR quality result available.'
    image,affine=preprocess(ROOT/'data/synthetic/v3/images/synthetic-01400.png',640)
    with torch.inference_mode(): before=model.encode_image(image[None])[1]; decoded_before=decode_maps(before,affine,vision['config'])
    if (d/'selected_with_ocr.pt').exists():
        combined=torch.load(d/'selected_with_ocr.pt',weights_only=True,map_location='cpu')
        assert all(torch.equal(v,combined['model'][k]) for k,v in vision['model'].items() if not k.startswith('ocr.'))
        model.load_state_dict(combined['model'],strict=True)
        with torch.inference_mode(): after=model.encode_image(image[None])[1]; decoded_after=decode_maps(after,affine,vision['config'])
        assert all(torch.equal(v,after[k]) for k,v in before.items()) and all(torch.equal(v,decoded_after[k]) for k,v in decoded_before.items())
        ocr=read_json(base/'ocr_v1/result.json'); scopes.append('ASCII_chart_crop_OCR')
        oq=read_json(base/'ocr_holdout/results.json'); ocr_quality=f"Training CER {oq['training_crop_recognition_DIAGNOSTIC']['cer']:.4f}, validation CER {ocr['validation']['cer']:.4f}, new-style oracle-crop CER {oq['oracle_gt_crop_recognition_DIAGNOSTIC']['cer']:.4f}; detected-region exact-word F1 {oq['end_to_end']['word_f1']:.4f}, CER {oq['end_to_end']['cer_including_missing_and_extra_regions']:.4f}. This is strong overfitting/domain-shift evidence. OCR is experimental and **not reliable for numerical QA**. CUDA CTC backward warned of nondeterminism under warn-only policy; fixed seeds do not guarantee bit-exact training."
    version={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'pyg':torch_geometric.__version__,'torch_cuda_build':torch.version.cuda,
             'gpu_query':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()}
    write_json(d/'release_audit.json',{'time_unix':time.time(),'versions':version,'parameters':total,'trainable_by_policy':stages,'executed_training_scopes':scopes,
        'graph_qa_segmentation_in_new_checkpoint':'untrained initialization; NOT usable trained predictions','selected_checkpoint_sha256':sha256(checkpoint),'selected_run':winner,
        'combined_non_ocr_tensors_and_cpu_decodes_exact':bool(ocr),'gpu_execution_in_this_audit':False,'resource_summary':summary})
    test=read_json(base/'independent_holdout/results.json')['results'][winner]
    text=f'''# Trained visual model card

Selected release: **{winner}**, {vision['stage']} epoch {vision['epoch']}. Vision checkpoint SHA256 `{sha256(checkpoint)}`. Selection used the complete 300-chart validation AP50 only, locked before any independent holdout inference. No reselection/retraining follows audit scores.

**Research checkpoint; not ready for reliable real-chart QA.** Real partial-source recall was 77/616 valid provided boxes across 20 ChartQA training images. OCR/domain transfer and numerical calibration remain blockers; a trained synthetic detector is not completed chart understanding.

ResNet18 ImageNet1K_V1 +128-channel FPN +stride4 fused CenterNet-style five box classes (bar, text, legend, line-segment envelope, axis) and separate point/tick heatmap/extent/offset heads. Full skeleton has **{total:,} parameters**; trainable policy counts are recorded in `release_audit.json`. Five warmup epochs freeze ResNet; later C4/C5 fine-tuning keeps all BN statistics and BN affine parameters frozen. Original and corrected target versions use the same architecture/hyperparameters/splits/seed. New line/axis segmentation, multimodal GAT, question/operand/evidence heads remain **untrained**. Legacy frozen GAT/MLP transfer is a separate diagnostic, not this network's trained graph branch.

Input: RGB image only, aspect-fit/white padding640x640, ImageNet normalization, FP32. Decoder uses predicted peaks, positive extents, sigmoid offsets, classwise NMS .5, peak threshold .10, cap128 before suppression, fixed report/inference operating score .25. Output boxes and centers are inverse-transformed to original-image top-left pixel coordinates; classes/scores are predicted. No GT boxes/types/values/answers enter deployed inference. Scores are not calibrated confidence. Segment envelopes are not stroke masks. No numerical calibration or final QA answer is produced.

Data:1,400 synthetic training/300 validation charts, bar/grouped/line with renderer labels and known linear categorical-axis support. Audit-only300 new-style charts are disjoint in checked numeric arrays/template identities/exact image hashes. Six visually distinct near-hash pairs retained. Validation is reused for development; pretrained membership and every shared graphic convention cannot be proven independent. Official historical benchmark tests stay closed. Horizontal bars, logarithmic/temporal axes, real typography and crowded/occluded marks are limitations.

Synthetic shortcuts remain: titles name plot family (`Synthetic bar/grouped_bar/line`) and labels use a small repeated vocabulary. Those pixels can help classification/recognition; identity-disjoint data are not proof of shortcut-free understanding. Frozen inventories are not modified after evaluation. Natural-title/text diversification and sensitivity tests require a new preregistered study.

Independent synthetic holdout AP50 **{test['ap50']['macro_ap']:.4f}**, AP75 **{test['ap75']['macro_ap']:.4f}**, macro recall50 **{test['ap50']['macro_recall']:.4f}**. AP uses the repository's all-points precision-envelope integral, not COCO101-point AP or .50:.95 mAP. See full per-class/type/keypoint metrics and raw predictions. This is not real-world QA/accessibility performance. Overfitting was observed in training histories; early validation-selected checkpoints and untouched new audit data limit claims, not a guarantee of zero overfitting.

Arithmetic limitation: training disabled TF32; the one-time three-model GPU audit retained PyTorch2.8 default cuDNN TF32 permission and did not enforce deterministic algorithms. Tensors remained float32 without autocast. Individual kernels were not traced; all three audit models share the configuration. Results are retained without rerunning/tuning on the holdout. Separate post-selection validation-only precision/repeatability checks are reported. CPU inference has no TF32 path.

OCR: {f"trained independently for {ocr['completed_epochs']} complete epochs, selected epoch {ocr['selected_epoch']} by validation CER {ocr['validation']['cer']:.4f}" if ocr else 'not completed'}. Optional 327,488-parameter CNN/BiGRU/CTC recognizes ASCII crops, geometry-only 90-degree rotation, greedy CTC, no dictionary/source-string correction. 109/110 validation strings are seen in training, so this is closed-vocabulary synthetic recognition. Detected-region end-to-end word/CER metrics count missing/extra regions; GT-crop recognition is explicitly separate. No claim of arbitrary Unicode/rotated text or verified numeric values.

{ocr_quality}

Use local release weights without downloads:

```powershell
.\\.venv\\Scripts\\python.exe scripts/infer_full_visual.py --image path/to/chart.png --output experiments/my_chart.json --with-ocr
```

CPU inference. Omit `--with-ocr` for vision only. Use `encode_image` and the declared decoder, not the skeleton's untrained graph/QA forward outputs. `selected_with_ocr.pt` packages trained vision+OCR; non-OCR tensors and CPU decoded predictions are verified identical. No frontend/deployment is included.

Reproducibility: frozen sources/configs, initial/pretrained hashes, per-epoch weights/metrics, raw predictions, source guards, checkpoints, optimizer/RNG/cursors, curves and resource ledgers retained. Charged cumulative GPU-work sessions **{summary['cumulative_gpu_seconds_estimated']/60:.2f} minutes**;{summary['tests']} passing tests. Working-state session time includes CPU/loading overhead; documented Windows suspension is excluded in a separately evidenced reconciliation with the raw ledger preserved. See `research/full_perception_training_report.md` for exact comparisons, limitations and continuation boundaries. No model/data download occurred during the new quality stages. Upstream model/data licenses and source terms remain in acquisition manifests; no new redistribution license is inferred.
'''
    (ROOT/'research/trained_visual_model_card.md').write_text(text,encoding='utf-8')
    files=[p for folder in [ROOT/'src/sensegraph',ROOT/'scripts',ROOT/'configs',ROOT/'tests',ROOT/'research'] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files.extend(ROOT/name for name in ['PROJECT_STATUS.md','README.md','pyproject.toml','requirements.lock','requirements-structural.lock','.gitignore'])
    files+=[p for p in d.glob('*') if p.is_file() and p.name!='source_and_release_hashes.json']
    for folder in ['v3','independent_holdout','ocr_v1','ocr_holdout','graph_transfer','real_partial','postselection_verification']:
        files.extend(p for p in (base/folder).rglob('*') if p.is_file())
    for folder in ['data/ocr_visual_v1','data/synthetic/perception_holdout_v1']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    write_json(d/'source_and_release_hashes.json',{'files':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))},'scope':'Local audit snapshot, not signed provenance. Older experiment guards independently protect raw data/artifacts.'})
    print({'release':winner,'parameters':total,'trained_scopes':scopes,'audit':'CPU only'})


if __name__=='__main__': main()
