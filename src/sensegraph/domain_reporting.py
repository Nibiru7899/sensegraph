"""Locked evaluation and evidence-limited reporting for the domain pilot."""
from pathlib import Path
import time,os
import torch
from .io import read_json,read_jsonl,write_json,sha256
from .domain_metrics import select_checkpoint,bootstrap_real
from .domain_ocr import Recognizer,oracle_ocr,detected_ocr
from .domain_training import (directory,verify,DomainSession,load_reference,
    evaluate_synthetic,evaluate_real,latest_ledger)
from .domain_data import DomainDataset
from .relation_training import seed_everything


def ocr_comparison(root,cfg,records,predictions,output,session):
    results={}; output.mkdir(parents=True,exist_ok=True)
    source=Path(root)/'data/models/domain_trocr/source_manifest.json'
    for kind in ['ctc','trocr']:
        if kind=='trocr' and (not source.exists() or not read_json(source).get('permission_resolved')):
            results[kind]={'available':False,'blocker':'Pinned source permission/assets unavailable'}; continue
        started=time.monotonic()
        try:
            if session.remaining<60: raise RuntimeError('Insufficient OCR evaluation reserve')
            recognizer=Recognizer(root,kind)
            oracle=oracle_ocr(root,records,recognizer,session)
            # GT strings and matching never enter recognizer inputs.
            detected=detected_ocr(root,records,predictions,recognizer,session)
            results[kind]={'available':True,'oracle':oracle,'detected':detected,'seconds':time.monotonic()-started,
                           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            del recognizer; torch.cuda.empty_cache()
        except Exception as error:
            results[kind]={'available':False,'blocker':f'{type(error).__name__}: {error}'}; torch.cuda.empty_cache()
        write_json(output/f'{kind}.json',results[kind])
        print({'OCR':kind,'available':results[kind]['available'],'remaining_seconds':session.remaining},flush=True)
    eligible=[k for k,v in results.items() if v.get('available') and v['oracle']['complete'] and v['oracle']['metrics']['CER'] is not None]
    selected=min(eligible,key=lambda k:(results[k]['oracle']['metrics']['CER'],k!='ctc')) if eligible else None
    write_json(output/'comparison.json',{'recognizers':{k:{'available':v.get('available'),'CER':v.get('oracle',{}).get('metrics',{}).get('CER'),'blocker':v.get('blocker')} for k,v in results.items()},
                                       'selected_on_validation':selected,'criterion':'Trusted provided real-validation GT-crop CER; CTC tie preference'})
    return results,selected


def diagnose_validation(root,cfg):
    verify(root,cfg); d=directory(root,cfg); out=d/'validation_diagnostic'; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): return read_json(out/'result.json')
    attempt=out/'attempt.json'
    if attempt.exists(): raise RuntimeError('Prior diagnostic attempted; preserve it, do not silently rerun')
    write_json(attempt,{'freeze_sha256':sha256(d/'freeze.json'),'reference_sha256':cfg['reference_sha256'],
                        'inventory_sha256':sha256(Path(root)/cfg['real_output']/'val.jsonl'),
                        'scope':'A-only validation diagnostic; final inventory remains closed; no adaptation claim'})
    seed_everything(cfg['seed']); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    records=read_jsonl(Path(root)/cfg['real_output']/'val.jsonl')
    with DomainSession(root,cfg,'evaluation') as session:
        torch.cuda.reset_peak_memory_stats(); model=load_reference(root,cfg).cuda().eval()
        val=DomainDataset(root,str(Path(cfg['synthetic_output'])/'val.jsonl'),'val')
        synthetic=evaluate_synthetic(model,val,cfg,out,'A_synthetic_val',session)
        real=evaluate_real(model,root,records,cfg,out,'A_real_val',session)
        saved=torch.load(Path(root)/cfg['reference'],weights_only=True,map_location='cpu')['model']
        unchanged=all(torch.equal(value.cpu(),saved[name]) for name,value in model.state_dict().items())
        # Repeat a pixel-only pass offline, including decoder, before releasing A.
        from .visual_training import preprocess
        from .full_visual_training import decode_maps
        image,affine=preprocess(Path(root)/records[0]['image_path'],640)
        with torch.inference_mode():
            _,a=model.encode_image(image[None].cuda()); one=decode_maps(a,affine,cfg)
            _,b=model.encode_image(image[None].cuda()); two=decode_maps(b,affine,cfg)
        deterministic=all(torch.equal(one[k],two[k]) for k in one)
        predicted=torch.load(out/'A_real_val_predictions.pt',weights_only=True)['predictions']
        del model,saved; torch.cuda.empty_cache()
        ocr,selected=ocr_comparison(root,cfg,records,predicted,out/'ocr',session)
        result={'scope':'Validation-only A/OCR diagnostic, not a completed A/B comparison','final_inventory_inference':False,
                'synthetic_ap50':synthetic['ap50']['macro_ap'],'real_supported_recall50':real['macro_supported_recall50'],
                'all_reference_tensors_unchanged':unchanged,'offline_repeat_equal':deterministic,
                'OCR_available':{k:v.get('available') for k,v in ocr.items()},'validation_OCR_choice':selected,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    write_json(out/'result.json',result); return result


def evaluate_final(root,cfg):
    verify(root,cfg); d=directory(root,cfg)
    status=read_json(d/'fit_status.json') if (d/'fit_status.json').exists() else {}
    if status.get('status') not in ['paired_early_stop','five_epoch_pilot_complete']:
        raise RuntimeError('Complete paired fitting unavailable; final inventory remains closed')
    out=d/'final'; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): return read_json(out/'result.json')
    if (out/'attempt.json').exists(): raise RuntimeError('Final inference already attempted; preserve failures, no automatic rerun')
    prof=read_json(d/'profile.json'); candidates=[{'arm':'A','real_recall':prof['reference_real_recall'],'epoch':0,'path':cfg['reference']}]
    for arm in ['control','realistic']:
        row=select_checkpoint(read_json(d/f'{arm}_history.json'),prof['reference_synthetic_ap50'],cfg['synthetic_guard_drop'])
        if row: candidates.append({**row,'path':str((d/arm/'best.pt').relative_to(root))})
    winner=max(candidates,key=lambda r:(r['real_recall'],-['A','control','realistic'].index(r['arm'])))
    seed_everything(cfg['seed']); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    validation=read_jsonl(Path(root)/cfg['real_output']/'val.jsonl')
    with DomainSession(root,cfg,'evaluation') as session:
        selected=load_reference(root,cfg).cuda().eval()
        selected.load_state_dict(torch.load(Path(root)/winner['path'],map_location='cpu',weights_only=True)['model'])
        evaluate_real(selected,root,validation,cfg,out,'selected_validation',session)
        predictions=torch.load(out/'selected_validation_predictions.pt',weights_only=True)['predictions']; del selected; torch.cuda.empty_cache()
        _,recognizer=ocr_comparison(root,cfg,validation,predictions,out/'validation_ocr',session)
        if recognizer is None: raise RuntimeError('No complete validation OCR baseline; final lock not created')
        lock={'detector':winner,'recognizer':recognizer,'weights':{r['arm']:sha256(Path(root)/r['path']) for r in candidates},
              'freeze_sha256':sha256(d/'freeze.json'),'final_manifest_sha256':sha256(Path(root)/cfg['real_output']/'final.jsonl')}
        write_json(out/'selection.json',lock); write_json(out/'attempt.json',{'selection_sha256':sha256(out/'selection.json'),'started_unix':time.time()})
        records=read_jsonl(Path(root)/cfg['real_output']/'final.jsonl'); results={}; cases={}
        for row in candidates:
            model=load_reference(root,cfg).cuda().eval(); model.load_state_dict(torch.load(Path(root)/row['path'],map_location='cpu',weights_only=True)['model'])
            results[row['arm']]=evaluate_real(model,root,records,cfg,out,row['arm'],session)
            cases[row['arm']]=read_json(out/f"{row['arm']}_cases.json"); del model; torch.cuda.empty_cache()
        chosen_predictions=torch.load(out/f"{winner['arm']}_predictions.pt",weights_only=True)['predictions']
        engine=Recognizer(root,recognizer); ocr={'oracle':oracle_ocr(root,records,engine,session),'detected':detected_ocr(root,records,chosen_predictions,engine,session)}
        write_json(out/'ocr.json',ocr); del engine; torch.cuda.empty_cache()
    comparisons={arm+'_minus_A':bootstrap_real(cases['A'],rows,seed=cfg['bootstrap_seed']) for arm,rows in cases.items() if arm!='A'}
    if 'control' in cases and 'realistic' in cases: comparisons['realistic_minus_control']=bootstrap_real(cases['control'],cases['realistic'],seed=cfg['bootstrap_seed'])
    result={'selection':lock,'detectors':results,'paired_bootstrap':comparisons,'OCR_complete':all(v['complete'] for v in ocr.values()),'single_training_seed':cfg['seed'],
            'complete':all(v['complete'] for v in results.values()) and all(v['complete'] for v in ocr.values()),
            'ineligible_arms':[arm for arm in ['control','realistic'] if arm not in results]}
    write_json(out/'result.json',result); return result


def report(root,cfg):
    root=Path(root); d=directory(root,cfg); initial=read_json(d/'initial.json'); validation=read_json(d/'synthetic_validation.json')
    manifest=read_json(root/cfg['synthetic_output']/'manifest.json'); realpath=root/cfg['real_output']/'manifest.json'
    real=read_json(realpath) if realpath.exists() else {'complete':False}; diagnostic=d/'validation_diagnostic/result.json'
    diagnostic=read_json(diagnostic) if diagnostic.exists() else None
    ledger=read_json(latest_ledger(root,cfg)); charged=ledger['historical_seconds']+ledger['charged_seconds']
    download=read_json(root/'experiments/download_ledger.json')['transferred_bytes']; finalpath=d/'final/result.json'
    complete=read_json(finalpath).get('complete',False) if finalpath.exists() else False
    result={'pilot_complete':complete,'synthetic_gate_passed':validation['valid'],'synthetic_gate_failures':validation['failures'],
            'synthetic_records':manifest['records'],'real_inventory_complete':real['complete'],'validation_diagnostic':diagnostic,
            'GPU_work_seconds_total':charged,'GPU_work_seconds_new':charged-initial['prior_gpu_seconds'],'GPU_work_seconds_remaining':18000-charged,
            'new_download_bytes':download-initial['download_initial'],'lifetime_download_bytes':download,'final_inventory_closed':not complete}
    write_json(d/'outcome.json',result)
    diagnosis=read_json(d/'diagnosis.json')['metrics']; review=read_json(d/'data_review.json') if (d/'data_review.json').exists() else {}
    reviewnotes=read_json(d/'real_review.json') if (d/'real_review.json').exists() else {}
    lines=['# Controlled real-domain generalization pilot','',
        '**Outcome: '+('matched pilot evaluated.' if complete else 'incomplete; fitting blocked by the declared synthetic collision gate.')+'**',
        'No adaptation benefit, convergence, reliable real-chart QA, or zero-leakage guarantee is established. Historical checkpoints and experiments are preserved.',
        '', '## Executed preparation and gate','',
        f"Generated {manifest['records']} charts: 1,400 train and 300 validation, 340 per family across five families. Train family/font cells contain 56 images; validation cells contain 20. Titles are unique across all 1,700 records. Fonts are split-disjoint.",
        'The initial 100-example preflight is retained at `data/synthetic/domain_v1`. Corrections made before full generation are recorded in `experiments/domain_generalization/v1/preflight_v1/review.json`; the corrected inventory is `data/synthetic/domain_v1b`.',
        'All records passed rendering-bound, inverse-transform, axis semantics, image size/hash, and numerical geometry checks. No QA labels or segmentation masks were invented. The final collision gate failed:',
        '', '| Split | Class | Colliding instances / all instances | Fraction |', '|---|---|---:|---:|']
    for split,row in validation['splits'].items():
        for cls,total in row['counts'].items():
            if row['collisions'].get(cls,0): lines.append(f"| {split} | {cls} | {row['collisions'][cls]}/{total} | {row['collision_fractions'][cls]:.4%} |")
    lines += ['', 'Class3 is line-segment envelope, class2 point, class5 legend label. The approved ceiling is 5% per class per split; validation line segments are 54/1,033 = 5.2275%. Every instance stays in evaluation. Union heatmaps and regression masking are retained. No samples were replaced and the ceiling was not relaxed.',
        f"Rendering supervision marks {manifest['ambiguous_text']} text regions ambiguous because of overlap or clipping. Automated review produced overlays for {review.get('anomaly_charts','unavailable')} anomaly charts. All 26 validation collision-chart thumbnails were visually reviewed; this is not manual certification of every ambiguous text region.",
        '', '## Diagnosis and real annotation coverage','',
        f"The historical cached real diagnostic recovered77/616 valid supplied boxes on20 images (12 horizontal bars, eight lines, zero vertical bars); its macro supported-label recall was {diagnosis['macro_supported_recall50']:.6f}. All83 valid line envelopes thinner than4 native pixels were missed. Failures:500 detection misses,31 localization failures,8 class confusions;239 unmatched predictions remain unknown.",
        'This imbalance prevents a general real-domain estimate. Thin envelope IoU is not stroke recall. Scene-graph unions are not physical detection targets. Source boxes lack complete object/background coverage; AP, precision, F1, real point/tick accuracy and supervised real/mixed fitting remain unavailable.',
        f"Fresh real inventory complete: {real['complete']}. "+('150 official-training identities, development60/validation30/final60, each balanced horizontal bars, vertical bars and lines.' if real['complete'] else 'See acquisition blocker evidence; no final inventory claim.'),
        'Original20 images remain development-only. Grouping uses title, exact source arrays, exact images and dHash<=4, with near-prior groups restricted to development. Publisher/shared-source and pretrained-model overlap remain unresolved. Candidate-pool acquisition is recorded separately from selected identities.',
        'The new mapper retains valid tick boxes with missing transcriptions, leaves ambiguous line-model boxes unsupported, and leaves zero-area boxes unavailable. Axis description single-box arrays are handled only when unambiguous. Horizontal source x_axis/y_axis are variable semantics, not assumed physical Cartesian directions. VisText raster transforms remain unavailable.',
        'Before inference, source overlay review found displaced axis descriptions/untranscribed tick boxes and inconsistent multi-model bar geometry. Supported-label scoring excludes all axis descriptions, untranscribed ticks, and multi-model bar marks. This conservative rule can withhold valid grouped bars too. Raw boxes remain preserved; explicit supported IDs and exclusions are frozen in data_review.json. The resulting metric covers a narrower source-label inventory.',
        'Native resolution, source-box size/aspect and provided-label-density strata are measured; source font identities, true line stroke widths and exhaustive clutter are unavailable rather than guessed. Legend presence is recorded; geometry thumbnails support qualitative layout review.',
        f"Agent review notes: {reviewnotes.get('summary','not yet completed')}",
        '', '## Leakage and evidence boundaries','',
        f"Exact image duplicates: none in the1,700 generated inventory. Prior exact-image/numerical checks covered2,400 historical configurations. {len(review.get('cross_split_near_hash_candidates',[]))} train/validation dHash candidates are recorded for review; different numerical arrays/styles/titles do not prove independent visual distributions. No benchmark test images/results were acquired or used for tuning.",
        'The final real inventory has not been inferred on in a blocked campaign. Agent annotation review precedes model inference and does not count as held-out model evaluation. Validation-only diagnostics cannot substitute for the missing matched comparison.',
        '', '## Reference and OCR validation diagnostics','']
    if diagnostic:
        lines += [f"A-only validation diagnostic: synthetic AP50 {diagnostic['synthetic_ap50']:.6f}; real supported macro recall50 {diagnostic['real_supported_recall50']:.6f}. Frozen reference tensors unchanged: {diagnostic['all_reference_tensors_unchanged']}; repeated offline pixel-only inference equal: {diagnostic['offline_repeat_equal']}.",
                  'Per-class support/recall, matched IoU, proposal counts, failures, timing and memory are in `validation_diagnostic/*metrics.json`. These measurements do not compare adaptation arms.',
                  '', '| Recognizer | GT-crop CER | GT-crop token WER | Exact-region accuracy | Detected-crop supported exact-region recall |', '|---|---:|---:|---:|---:|']
        for kind in ['ctc','trocr']:
            path=d/f'validation_diagnostic/ocr/{kind}.json'
            value=read_json(path) if path.exists() else {}
            if value.get('available'):
                a=value['oracle']['metrics']; b=value['detected']['metrics']
                lines.append(f"| {kind} | {a['CER']:.6f} | {a['WER']:.6f} | {a['exact_region_accuracy']:.6f} | {b['supported_exact_region_recall']:.6f} |")
            else: lines.append(f"| {kind} | unavailable: {value.get('blocker','not run')} | — | — | — |")
        lines += [f"Validation CER preference: {diagnostic['validation_OCR_choice']}. No final evaluation/production recognizer claim follows from this blocked study."]
    else: lines += ['Not executed; no new model evidence is claimed.']
    lines += ['','CTC remains unchanged with32 output positions; repeated characters require blank positions and narrow crops can have fewer usable frames. The same observable geometry-only clockwise90-degree rule is used for both recognizers. It does not resolve arbitrary rotation. TrOCR uses native preprocessing and greedy beam1 decoding capped at128 new tokens. Failures/truncations and numeric/negative/decimal/long/vertical/unseen-transcription subsets are retained.',
        'Historical OCR "word F1"0.3078 is an exact-region F1, not a token-level metric. New CER and token WER preserve case, signs, decimal points and punctuation after NFC normalization. Partial-label unmatched predictions remain unknown; missing supplied text remains in denominators.',
        '', '## Resource accounting and fitting status','',
        f"GPU working-state total {charged/60:.4f}/300 minutes; new work {(charged-initial['prior_gpu_seconds'])/60:.4f} minutes; remaining {(18000-charged)/60:.4f} minutes. New charged transfers {download-initial['download_initial']:,} bytes; lifetime {download:,}/10,000,000,000 bytes.",
        'No B-control/B-realistic fitting or optimizer profile was admitted after the failed annotation gate. The planned five paired epochs, capacity projection, checkpoint selection, curves and paired2,000-resample uncertainty therefore have no executed results. No partial fitting is presented as completion. C/D are ineligible: missing real background supervision and separate matched allocation.',
        'The configuration and resumable matched trainer implement the requested FP32/frozen-BN/update/selection policy, but the complete two-arm GPU campaign remains unexecuted. CPU fixture optimization/gradient tests are wiring evidence only.',
        '', '## Verification and reproduction','',
        'Use the project Python3.11 environment; no dependency upgrade. Commands accept `--config configs/domain_generalization_v1.json`. Retained logs record actual runs; run the full tests with `python -m pytest -q --basetemp experiments/domain_tests_recheck`.',
        '```powershell', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment diagnose', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment prepare --limit 100',
        '# Full generation already exists; do not truncate its manifests with --limit.', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment validate',
        '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment acquire', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment inspect',
        '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment freeze', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment profile # BLOCKED while collision gate fails',
        '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment fit # BLOCKED', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment diagnose-validation # immutable one-attempt diagnostic',
        '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment evaluate # BLOCKED until paired fitting complete', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment report', '.\\.venv\\Scripts\\python.exe -m sensegraph.domain_experiment verify', '```',
        'A continuation needs a separately versioned, reviewed resolution of the collision gate or explicit new approval; never edit this frozen inventory to obtain a pass. Re-profile complete paired capacity against the remaining closed ledger before any fitting. No downstream graph/QA training is authorized by this pilot.',
        '', '## Provenance and terms','',
        '[ChartQA official source](https://github.com/vis-nlp/ChartQA) documents noisy annotations; the author release metadata declares GPL-3.0 while third-party chart rights remain applicable. Only official training members are acquired by budgeted HTTPS ranges, with revision/CRC/SHA/size records.',
        '[Pinned TrOCR model](https://huggingface.co/microsoft/trocr-small-printed/tree/04e994ab854b0089d4929f48c2b4dbe2ce78a340): HF model-card license declaration is absent. The separate official [Microsoft project README](https://github.com/microsoft/unilm/tree/master/trocr) applies the root MIT license and lists printed checkpoints. Its exact retrieved notice, hashes and permission basis are retained; this study does not redistribute weights or charts.',
        '[Transformers4.57.3 TrOCR documentation](https://github.com/huggingface/transformers/blob/v4.57.3/docs/source/en/model_doc/trocr.md) verified explicit processor/model loading. The pinned safetensors checksum is recorded in the source manifest. Frozen source/config/data digests, historical status archive mapping and resource chain are under `experiments/domain_generalization/v1/`.',
        '']
    (root/'research/real_domain_generalization.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
