"""Locked evaluation and evidence-limited reporting for the domain pilot."""
from pathlib import Path
import time,os
import torch
from .io import read_json,read_jsonl,write_json,sha256
from .domain_metrics import select_checkpoint
from .domain_ocr import Recognizer,oracle_ocr,detected_ocr
from .adaptation_training import (directory,verify,DomainSession,load_reference,
    evaluate_synthetic,evaluate_real,latest_ledger)
from .domain_data import DomainDataset
from .relation_training import seed_everything
import numpy as np


def bootstrap_counts(a,b,samples=2000,seed=20261110):
    if [r['id'] for r in a]!=[r['id'] for r in b]: raise ValueError('Unpaired inventories')
    classes=sorted({k for r in a for k in r['classes']}); groups={}
    for i,r in enumerate(a): groups.setdefault(r['source_chart_type'],[]).append(i)
    support=np.asarray([[r['classes'].get(k,{}).get('support',0) for k in classes] for r in a])
    other=np.asarray([[r['classes'].get(k,{}).get('support',0) for k in classes] for r in b])
    if not np.array_equal(support,other): raise ValueError('Paired reference support differs')
    x=np.asarray([[r['classes'].get(k,{}).get('recovered',0) for k in classes] for r in a]); y=np.asarray([[r['classes'].get(k,{}).get('recovered',0) for k in classes] for r in b])
    def delta(indices):
        s=support[indices].sum(0); active=s>0
        return float(((y[indices].sum(0)[active]-x[indices].sum(0)[active])/s[active]).mean())
    rng=np.random.default_rng(seed); values=[delta(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups.values()])) for _ in range(samples)]
    return {'difference':delta(np.arange(len(a))),'ci95':np.quantile(values,[.025,.975]).tolist(),'samples':samples,'seed':seed,
            'scope':'Chart-type-stratified paired chart bootstrap; conditional on one training seed and this source inventory'}


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
        rows=read_json(d/f'{arm}_history.json'); row=select_checkpoint(rows,prof['reference_synthetic_ap50'],cfg['synthetic_guard_drop'])
        eligible=row is not None
        if row is None: row=max(rows,key=lambda r:(r['real_recall'],-r['epoch']))
        candidates.append({**row,'eligible':eligible,'path':str((d/arm/f"epoch_{row['epoch']}.pt").relative_to(root))})
    winner=max([r for r in candidates if r.get('eligible',True)],key=lambda r:(r['real_recall'],-['A','control','realistic'].index(r['arm'])))
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
    comparisons={arm+'_minus_A':bootstrap_counts(cases['A'],rows,seed=cfg['bootstrap_seed']) for arm,rows in cases.items() if arm!='A'}
    if 'control' in cases and 'realistic' in cases: comparisons['realistic_minus_control']=bootstrap_counts(cases['control'],cases['realistic'],seed=cfg['bootstrap_seed'])
    result={'selection':lock,'detectors':results,'paired_bootstrap':comparisons,'OCR_complete':all(v['complete'] for v in ocr.values()),'single_training_seed':cfg['seed'],
            'complete':all(v['complete'] for v in results.values()) and all(v['complete'] for v in ocr.values()),
            'ineligible_arms':[r['arm'] for r in candidates if not r.get('eligible',True)]}
    write_json(out/'result.json',result); return result


def report(root,cfg):
    """Build the report from measured outputs, including rejected/unrun arms."""
    from .io import sha256
    d=directory(root,cfg); initial=read_json(d/'initial.json'); gate=read_json(d/'synthetic_validation.json')
    final=read_json(d/'final/result.json') if (d/'final/result.json').exists() else None
    fit=read_json(d/'fit_status.json') if (d/'fit_status.json').exists() else {'status':'not_started'}
    prof=read_json(d/'profile.json') if (d/'profile.json').exists() else None
    ledger=read_json(latest_ledger(root,cfg)); total=ledger['historical_seconds']+ledger['charged_seconds']
    histories={a:read_json(d/f'{a}_history.json') if (d/f'{a}_history.json').exists() else [] for a in ['control','realistic']}
    result={'complete':bool(final and final['complete']),'fitting':fit,'training_gate_passed':gate['valid'],
            'GPU_total_seconds':total,'GPU_remaining_seconds':18000-total,'GPU_new_seconds':total-initial['prior_gpu_seconds'],
            'matched_epochs':{a:len(v) for a,v in histories.items()},'final':final}
    write_json(d/'outcome.json',result)
    lines=['# Controlled real-domain generalization: version2 continuation','',
      '**Outcome: '+('matched pilot completed.' if result['complete'] else 'incomplete; see measured gate/capacity/fitting status below.')+'**',
      '', '## Preservation and data amendment','',
      'The user authorized resolving the blocker and completing the pilot within the existing lifetime allowance. The entire prior study, code, data, failed gate, diagnostics, ledgers and reports remain preserved. Prior report/status copies with digests are under `experiments/domain_generalization/v2/archive/`; original v1 source guards remain valid.',
      'The version2 generator keeps all1,700 predetermined chart indices, font/family cells, titles, layouts, resolutions and series counts. Only line values change: correlated common trends in75% of line charts, opposing crossing trends in25%. No sample retries, filtering, annotation deletion, threshold increase or model-result-driven generator tuning. Every instance remains in evaluation. Segmentation remains disabled.',
      'Non-line charts intentionally repeat the prior validation diagnostic cohort. This is a versioned repair, not a new independent synthetic holdout. Real validation30 was already observed; final60 was closed until locked inference. No zero-overfitting or zero-leakage claim is justified.',
      '', '## Data and verification gates','',f"All1,700 records: gate passed={gate['valid']}; failures={gate['failures']}",
      '', '| Split | Class | Instances | Colliding instances | Fraction |','|---|---|---:|---:|---:|']
    for split,row in gate['splits'].items():
        for c,n in row['counts'].items(): lines.append(f"| {split} | {c} | {n} | {row['collisions'].get(c,0)} | {row['collision_fractions'][c]:.4%} |")
    lines+=['','Classes:1 bar,2 point,3 line envelope,4 text,5 legend label,6 axis,7 tick. The per-class/per-split gate stays5%. Union heatmaps/masked ambiguous regressions and all-instance scoring are unchanged.',
      'Automated renderer bounds, transforms, source hashes and train/validation identities were checked. Anomaly overlays and near-hash candidates are retained in data_review.json; visual similarity is not proof of shared data. Font sets/title grammar/numerical arrays are separated. Old benchmark tests remain untouched.',
      'Real inventory and conservative supported-label exclusions are byte-identical to v1. Untranscribed ticks, unverified axis descriptions, multi-model bar marks, invalid/zero-area boxes and unaligned line models remain unscored/unknown. No real AP, global precision/F1 or real point/tick accuracy is claimed. Agent review is not exhaustive human relabeling; source annotations remain noisy.',
      '', '## Matched fitting','',f"Fitting status: `{fit['status']}`. Completed arm epochs: {result['matched_epochs']}.",
      'Both arms initialize the same pinned A checkpoint with fresh optimizers. Control uses original synthetic training data; realistic uses the repaired inventory. Architecture640, FP32/TF32 off, C4/C5+FPN+active heads, frozen BN/unused branches, batch1/accumulation8, AdamW1e-4/1e-5, decay1e-4, clip1, warmup/cosine and fixed decoder are matched. No graph/QA/calibration training or OCR fine-tuning occurred.',
      'Training/validation curves and optimizer/RNG/cursor checkpoints are retained. Selection uses real-validation macro supported recall50 and synthetic AP50 >= A-.05, earliest ties; overall A/control/realistic tie order. A five-epoch endpoint with improving validation is a bounded pilot, not demonstrated convergence.',
      '', '| Arm | Epoch | Updates cumulative | Train loss | Synthetic val AP50 | Real val recall50 |','|---|---:|---:|---:|---:|---:|']
    if prof:
        lines.append(f"| A | 0 | 0 | — | {prof['reference_synthetic_ap50']:.6f} | {prof['reference_real_recall']:.6f} |")
        lines+=[]
    for arm,rows in histories.items():
        for r in rows: lines.append(f"| {arm} | {r['epoch']} | {r['optimizer_steps']} | {r['train_loss']:.5f} | {r['synthetic_ap50']:.6f} | {r['real_recall']:.6f} |")
    if prof: lines+=['',f"Profile admission: `{prof['admission']}`. Projected full arm epoch: {prof['projected_epoch_seconds']:.2f}s, including checkpoint I/O and conservative timing margin. Profile invariants: {prof['invariants']}. Profile updates were discarded; scientific arms reload A."]
    lines+=['','## Locked final evaluation','']
    if final:
        selected=final['selection']; lines += [f"Validation-selected detector: `{selected['detector']['arm']}` epoch{selected['detector']['epoch']}; OCR: `{selected['recognizer']}`. Selection digests were written before final inference. Ineligible detector arms (reported but not selectable): {final['ineligible_arms']}.",
          'Final inventory:60 fresh official-training images,20 per orientation/type; this is a held-out source inventory, not official ChartQA benchmark-test performance. No final-result-driven retraining/reselection.',
          '', '| Arm | Macro supported recall50 | Mean proposals | Unknown unmatched predictions | Failures |','|---|---:|---:|---:|---:|']
        for arm,r in final['detectors'].items(): lines.append(f"| {arm} | {r['macro_supported_recall50']:.6f} | {r['mean_proposals']:.2f} | {r['unmatched_predictions_unknown']} | {len(r['failures'])} |")
        for arm,r in final['detectors'].items():
            lines+=['',f'### {arm}: per-class supported recall','', '| Class | Recovered / support | Recall50 | Matched mean IoU |','|---|---:|---:|---:|']
            for name,c in r['per_class'].items(): lines.append(f"| {name} | {c['recovered']}/{c['support']} | {c['recall50']:.6f} | {r['matched_localization'][name]['mean_iou']} |")
            lines += ['',f"Failure categories: {r['failure_categories']}. Thin envelopes: {r['groups'].get('thin')}. Full pass {r['seconds']:.3f}s, including preprocessing/overlays/scoring; not isolated model-kernel latency."]
        lines+=['','### Paired chart-bootstrap comparisons','', 'Differences in supported macro recall50;2,000 stratified chart resamples, seed20261110. Uncertainty is conditional on this one training seed, label coverage and source inventory. No across-seed robustness claim.',
                '', '| Comparison | Difference | Conditional95% interval |','|---|---:|---|']
        for name,r in final['paired_bootstrap'].items(): lines.append(f"| {name} | {r['difference']:.6f} | {r['ci95']} |")
        lines+=['','## OCR: GT crops versus detected crops','', '| Validation recognizer | GT-crop CER | Token WER | Exact-region accuracy |','|---|---:|---:|---:|']
        for kind in ['ctc','trocr']:
            v=read_json(d/f'final/validation_ocr/{kind}.json')
            if v.get('available'):
                m=v['oracle']['metrics']; lines.append(f"| {kind} | {m['CER']:.6f} | {m['WER']:.6f} | {m['exact_region_accuracy']:.6f} |")
        ocr=read_json(d/'final/ocr.json')
        for scope,r in ocr.items(): lines+=['',f"Final selected OCR {scope}: `{r['metrics']}`; truncations={r['truncations']}; failures={r['failures']}."]
        lines+=['','Native preprocessing, same geometry-only rotation rule, pinned TrOCR revision and greedy beam1/max128 tokens remain unchanged. CER/WER retain case/sign/decimal differences; CER can exceed1 through insertions. Missed provided text remains in denominators; unmatched predictions unknown. Historical word F1 was exact-region F1, not token-level F1. Numerical/negative/decimal/long/vertical/unseen-string subsets are in raw oracle JSON; zero support gives no performance claim.']
    else: lines+=['Final not executed. No final result or improvement is fabricated.']
    lines+=['','## Resource accounting and reproducibility','',f"Closed-ledger cumulative GPU work: {total/60:.4f}/300min; remaining {(18000-total)/60:.4f}min; v2 work {(total-initial['prior_gpu_seconds'])/60:.4f}min. No new downloads. Budget chains preserve all prior attempts and working-state timers/idle-sleep holds.",
      'Real/mixed fitting remains ineligible without complete background supervision and a separate allocation. No trained QA or reliable real-chart QA claim follows even if detection recall improves.',
      '```powershell', '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation validate', '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation inspect # before freezing only',
      '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation freeze', '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation profile', '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation fit',
      '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation evaluate # immutable final attempt', '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation report', '.\\.venv\\Scripts\\python.exe -m sensegraph.adaptation verify',
      '.\\.venv\\Scripts\\python.exe -m pytest -q --basetemp experiments/adaptation_recheck_tests', '```',
      'Config: configs/adaptation_v2.json; protocol: research/adaptation_protocol_v2.md; all artifacts: experiments/domain_generalization/v2/. Existing baseline reproduction stays in research/baseline_report.md. The original v1 report is preserved under the v2 archive. Final inference is not rerun by report/verify.', '']
    Path(root,'research/real_domain_generalization.md').write_text('\n'.join(lines),encoding='utf-8')
    return {k:v for k,v in result.items() if k!='final'}
