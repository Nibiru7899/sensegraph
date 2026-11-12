# Trained visual model card

Selected release: **corrected_v3**, perception epoch 8. Vision checkpoint SHA256 `97b5cc72cf5f138d0d0709bcd01e40792cbcecc05b9b39a5f4146cabbb72cf8e`. Selection used the complete 300-chart validation AP50 only, locked before any independent holdout inference. No reselection/retraining follows audit scores.

**Research checkpoint; not ready for reliable real-chart QA.** Real partial-source recall was 77/616 valid provided boxes across 20 ChartQA training images. OCR/domain transfer and numerical calibration remain blockers; a trained synthetic detector is not completed chart understanding.

ResNet18 ImageNet1K_V1 +128-channel FPN +stride4 fused CenterNet-style five box classes (bar, text, legend, line-segment envelope, axis) and separate point/tick heatmap/extent/offset heads. Full skeleton has **13,875,510 parameters**; trainable policy counts are recorded in `release_audit.json`. Five warmup epochs freeze ResNet; later C4/C5 fine-tuning keeps all BN statistics and BN affine parameters frozen. Original and corrected target versions use the same architecture/hyperparameters/splits/seed. New line/axis segmentation, multimodal GAT, question/operand/evidence heads remain **untrained**. Legacy frozen GAT/MLP transfer is a separate diagnostic, not this network's trained graph branch.

Input: RGB image only, aspect-fit/white padding640x640, ImageNet normalization, FP32. Decoder uses predicted peaks, positive extents, sigmoid offsets, classwise NMS .5, peak threshold .10, cap128 before suppression, fixed report/inference operating score .25. Output boxes and centers are inverse-transformed to original-image top-left pixel coordinates; classes/scores are predicted. No GT boxes/types/values/answers enter deployed inference. Scores are not calibrated confidence. Segment envelopes are not stroke masks. No numerical calibration or final QA answer is produced.

Data:1,400 synthetic training/300 validation charts, bar/grouped/line with renderer labels and known linear categorical-axis support. Audit-only300 new-style charts are disjoint in checked numeric arrays/template identities/exact image hashes. Six visually distinct near-hash pairs retained. Validation is reused for development; pretrained membership and every shared graphic convention cannot be proven independent. Official historical benchmark tests stay closed. Horizontal bars, logarithmic/temporal axes, real typography and crowded/occluded marks are limitations.

Synthetic shortcuts remain: titles name plot family (`Synthetic bar/grouped_bar/line`) and labels use a small repeated vocabulary. Those pixels can help classification/recognition; identity-disjoint data are not proof of shortcut-free understanding. Frozen inventories are not modified after evaluation. Natural-title/text diversification and sensitivity tests require a new preregistered study.

Independent synthetic holdout AP50 **0.8748**, AP75 **0.4838**, macro recall50 **0.8409**. AP uses the repository's all-points precision-envelope integral, not COCO101-point AP or .50:.95 mAP. See full per-class/type/keypoint metrics and raw predictions. This is not real-world QA/accessibility performance. Overfitting was observed in training histories; early validation-selected checkpoints and untouched new audit data limit claims, not a guarantee of zero overfitting.

Arithmetic limitation: training disabled TF32; the one-time three-model GPU audit retained PyTorch2.8 default cuDNN TF32 permission and did not enforce deterministic algorithms. Tensors remained float32 without autocast. Individual kernels were not traced; all three audit models share the configuration. Results are retained without rerunning/tuning on the holdout. Separate post-selection validation-only precision/repeatability checks are reported. CPU inference has no TF32 path.

OCR: trained independently for 13 complete epochs, selected epoch 8 by validation CER 0.1703. Optional 327,488-parameter CNN/BiGRU/CTC recognizes ASCII crops, geometry-only 90-degree rotation, greedy CTC, no dictionary/source-string correction. 109/110 validation strings are seen in training, so this is closed-vocabulary synthetic recognition. Detected-region end-to-end word/CER metrics count missing/extra regions; GT-crop recognition is explicitly separate. No claim of arbitrary Unicode/rotated text or verified numeric values.

Training CER 0.0014, validation CER 0.1703, new-style oracle-crop CER 0.2311; detected-region exact-word F1 0.3078, CER 0.4430. This is strong overfitting/domain-shift evidence. OCR is experimental and **not reliable for numerical QA**. CUDA CTC backward warned of nondeterminism under warn-only policy; fixed seeds do not guarantee bit-exact training.

Use local release weights without downloads:

```powershell
.\.venv\Scripts\python.exe scripts/infer_full_visual.py --image path/to/chart.png --output experiments/my_chart.json --with-ocr
```

CPU inference. Omit `--with-ocr` for vision only. Use `encode_image` and the declared decoder, not the skeleton's untrained graph/QA forward outputs. `selected_with_ocr.pt` packages trained vision+OCR; non-OCR tensors and CPU decoded predictions are verified identical. No frontend/deployment is included.

Reproducibility: frozen sources/configs, initial/pretrained hashes, per-epoch weights/metrics, raw predictions, source guards, checkpoints, optimizer/RNG/cursors, curves and resource ledgers retained. Charged cumulative GPU-work sessions **215.07 minutes**;114 passing tests. Working-state session time includes CPU/loading overhead; documented Windows suspension is excluded in a separately evidenced reconciliation with the raw ledger preserved. See `research/full_perception_training_report.md` for exact comparisons, limitations and continuation boundaries. No model/data download occurred during the new quality stages. Upstream model/data licenses and source terms remain in acquisition manifests; no new redistribution license is inferred.
