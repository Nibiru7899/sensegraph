"""Export the sealed controlled comparison and standalone figures from actual runs."""
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensegraph.io import read_json, sha256, write_json
from sensegraph.relation_experiment import require_frozen, history_check
from sensegraph.synthetic import RELATIONS


def csv_file(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    frozen = require_frozen(ROOT)
    directory = ROOT / "experiments/milestone3"
    config = frozen["config"]
    inventory = read_json(directory / "sealed_inventory.json")
    comparison = read_json(directory / "test_comparison.json")
    tests = ET.parse(directory / "tests.xml")
    if tests.findall(".//failure") or tests.findall(".//error"):
        raise RuntimeError("Tests failed; report not finalized")
    tests_passed = len(tests.findall(".//testcase"))
    rows, class_rows, resource_rows, training_rows, errors = [], [], [], [], []
    runs = {}
    for entry in inventory["runs"]:
        location = directory / "runs" / entry["name"]
        train = read_json(location / "result.json")
        if sha256(location / "result.json") != entry["result_sha256"]:
            raise RuntimeError("Selected training artifact changed")
        if not train["complete"]:
            continue
        test = read_json(location / "test.json")
        with np.load(location / "test_predictions.npz", allow_pickle=False) as arrays:
            if not np.isfinite(arrays["scores"]).all() or arrays["scores"].shape != arrays["labels"].shape:
                raise RuntimeError("Invalid saved test scores")
        runs[entry["name"]] = {"train": train, "test": test}
        rows.append({"name": entry["name"], "dataset": train["dataset"], "kind": train["variant"]["architecture"],
                     "variant": train["variant"]["variant"], "seed": train.get("seed"),
                     "macro_f1": test["metrics"]["macro_f1"], "micro_f1": test["metrics"]["micro_f1"], "macro_ap": test["metrics"]["macro_ap"],
                     "parameters": train["parameter_count"], "completed_steps": train.get("completed_steps", 0), "best_step": train.get("best_step", 0),
                     "runtime_seconds": train["elapsed_seconds"], "device": test["device"],
                     "inference_median_seconds": test["latency_median_seconds"], "inference_p95_seconds": test["latency_p95_seconds"],
                     "train_peak_allocated_bytes": train.get("peak_allocated_bytes"), "train_peak_reserved_bytes": train.get("peak_reserved_bytes"),
                     "test_peak_allocated_bytes": test["peak_allocated_bytes"], "checkpoint_bytes": train.get("checkpoint_bytes",0)})
        for label, values in test["metrics"]["per_class"].items():
            class_rows.append({"name": entry["name"], "relation": label, **values})
        for item in train.get("train_log", []):
            training_rows.append({"name": entry["name"], "step": item["step"], "train_loss": item["train_loss"],
                                  "gradient_norm": item["gradient_norm"], "validation_macro_ap": item.get("validation_macro_ap"),
                                  "elapsed_seconds": item["elapsed_seconds"]})
        errors.extend({"name": entry["name"], **item} for item in test["errors"])
    ledger = read_json(directory / "gpu_budget.json")
    for session in ledger["sessions"]:
        resource_rows.append(session)
    if any(s["actual_seconds"] is None for s in ledger["sessions"]):
        raise RuntimeError("An unfinished GPU reservation remains")
    lifetime = ledger["historical_seconds"] + ledger["charged_seconds"]
    if lifetime > ledger["limit_seconds"]:
        raise RuntimeError("GPU resource limit exceeded")
    csv_file(directory / "comparison.csv", rows)
    csv_file(directory / "per_class.csv", class_rows)
    csv_file(directory / "training_log.csv", training_rows)
    csv_file(directory / "gpu_resource.csv", resource_rows)
    csv_file(directory / "error_examples.csv", errors)

    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 4.8), dpi=160)
    FigureCanvasAgg(figure)
    ax_loss, ax_ap = figure.subplots(1,2)
    for name, result in runs.items():
        train = result["train"]
        if train["dataset"] != "v3" or train["variant"]["variant"] != "main" or train["variant"]["architecture"] == "rule":
            continue
        color = "#205a91" if train["variant"]["architecture"] == "mlp" else "#b04e2f"
        log = train["train_log"]
        loss = np.array([r["train_loss"] for r in log])
        smoothed = np.convolve(loss, np.ones(20)/20, mode="valid")
        ax_loss.plot(np.arange(20,len(loss)+1),smoothed,color=color,alpha=.7,label=name.replace("v3_",""))
        evaluated=[r for r in log if "validation_macro_ap" in r]
        ax_ap.plot([r["step"] for r in evaluated],[r["validation_macro_ap"] for r in evaluated],color=color,alpha=.7,marker=".")
    ax_loss.set(title="Training weighted BCE (20-step moving mean)",xlabel="Optimizer steps",ylabel="Loss")
    ax_ap.set(title="Validation macro average precision",xlabel="Optimizer steps",ylabel="Macro AP",ylim=(0,1))
    ax_loss.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures / "learning_curves.png")

    def main_values(architecture, metric):
        return np.array([runs[f"v3_{architecture}_main_s{seed}"]["test"]["metrics"][metric] for seed in config["seeds"]])
    means={name:{key:float(main_values(name,key).mean()) for key in ("macro_f1","micro_f1","macro_ap")} for name in ("mlp","gat")}
    rule=runs["v3_rule"]["test"]["metrics"]
    main_table=[f"| Rule (CPU) | {rule['macro_f1']:.4f} | {rule['micro_f1']:.4f} | {rule['macro_ap']:.4f} | 0 |"]
    for model in ("mlp","gat"):
        mf=main_values(model,"macro_f1"); mi=main_values(model,"micro_f1")
        main_table.append(f"| {model.upper()} (3 seeds) | {mf.mean():.4f} ± {mf.std(ddof=1):.4f} | {mi.mean():.4f} ± {mi.std(ddof=1):.4f} | {means[model]['macro_ap']:.4f} | {frozen['parameter_counts'][model]:,} |")
    class_table=[]
    for label in RELATIONS:
        a=[runs[f"v3_gat_main_s{s}"]["test"]["metrics"]["per_class"][label] for s in config["seeds"]]
        b=[runs[f"v3_mlp_main_s{s}"]["test"]["metrics"]["per_class"][label] for s in config["seeds"]]
        class_table.append(f"| {label} | {a[0]['positives']:,} | {a[0]['negatives']:,} | {np.mean([v['f1'] for v in b]):.4f} | {np.mean([v['f1'] for v in a]):.4f} | {rule['per_class'][label]['f1']:.4f} |")
    seed_table=[]
    for seed in config["seeds"]:
        a=runs[f"v3_gat_main_s{seed}"]["test"]["metrics"];b=runs[f"v3_mlp_main_s{seed}"]["test"]["metrics"]
        seed_table.append(f"| {seed} | {b['macro_f1']:.4f} | {a['macro_f1']:.4f} | {a['macro_f1']-b['macro_f1']:+.4f} | {b['micro_f1']:.4f} | {a['micro_f1']:.4f} |")
    ablations=[]
    for variant in ("main",*config["ablations"]):
        result=runs[f"v3_gat_{variant}_s{config['ablation_seed']}"]["test"]["metrics"]
        ablations.append({"variant":variant,**{key:result[key] for key in ("macro_f1","micro_f1","macro_ap")}})
    csv_file(directory / "ablations.csv",ablations)
    figure=Figure(figsize=(9,4),dpi=160);FigureCanvasAgg(figure);axes=figure.subplots()
    axes.bar([a['variant'] for a in ablations],[a['macro_f1'] for a in ablations],color=["#b04e2f","#999999","#718c9a","#407259"])
    axes.set(ylim=(0,1),ylabel="Test macro F1",title="Preplanned GATv2 ablations, seed 42 only")
    axes.tick_params(axis="x",labelsize=8);figure.tight_layout();figure.savefig(figures / "ablations.png")
    intervals=comparison["paired_chart_bootstrap"]
    paired_lines=[]
    for key,value in intervals.items():
        paired_lines.append(f"- {key}: GAT−MLP **{value['gat_minus_mlp']*100:+.3f} percentage points**, conditional chart-bootstrap 95% interval **[{value['ci95'][0]*100:+.3f}, {value['ci95'][1]*100:+.3f}]**.")
    integrity=history_check(ROOT)
    v3=read_json(directory / "v3_integrity.json")
    near=read_json(directory / "near_duplicate_review.json")
    overfits=[read_json(p) for p in sorted((directory / "runs").glob("*overfit/result.json"))]
    overfit_lines=[f"- {r['variant']['architecture']}: initial fixed-batch BCE {r['initial_fixed_batch_loss']:.6f}; selected-checkpoint fixed-batch BCE {r['final_fixed_batch_loss']:.6f}; last optimizer-step BCE {r['train_log'][-1]['train_loss']:.6f}." for r in overfits]
    all_neural=[value['train'] for value in runs.values() if value['train']['variant']['architecture']!='rule']+overfits
    checks=all(r['gradients_finite'] and r['reload_exact'] and r['repeated_inference_exact'] and r['complete'] for r in all_neural)
    if not checks:
        raise RuntimeError("A neural verification check failed")
    now=datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(timespec="seconds")
    summary={"created_at":now,"kind":"executed_synthetic_oracle_proposal_relation_experiment","tests_passed":tests_passed,
             "main_means":means,"main_seed_results":seed_table,"paired_intervals":intervals,"ablations":ablations,
             "runs":rows,"history_integrity":integrity,"gpu_budget":ledger,"v3_integrity_summary":{k:v for k,v in v3.items() if k!='leakage'},
             "near_duplicate_candidates":len(v3['leakage']['near_duplicates']),"source_hashes":frozen['source_sha256'],
             "all_gradient_reload_repeat_checks_passed":checks,"test_inventory_sha256":sha256(directory / "sealed_inventory.json")}
    write_json(ROOT / "research/milestone3_results.json",summary)
    run_table="\n".join(f"| {r['name']} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} | {r['macro_ap']:.4f} | {r['best_step']} | {r['runtime_seconds']:.2f} | {(r['train_peak_allocated_bytes'] or 0)/2**20:.1f} | {r['inference_median_seconds']*1000:.3f} ({r['device']}) |" for r in rows)
    ablation_table="\n".join(f"| {a['variant']} | {a['macro_f1']:.4f} | {a['micro_f1']:.4f} | {a['macro_ap']:.4f} |" for a in ablations)
    equal_gat=np.mean([runs[f"v3_gat_main_s{s}"]["test"]["metrics"]["per_class"]["equal_to"]["f1"] for s in config['seeds']])
    equal_mlp=np.mean([runs[f"v3_mlp_main_s{s}"]["test"]["metrics"]["per_class"]["equal_to"]["f1"] for s in config['seeds']])
    report=f"""# Milestone 3 — controlled graph neural network comparison

Generated {now} from executed, sealed local experiments. **Complete.** All results below concern **oracle-proposal synthetic structural relationship prediction**. They are not real-image extraction, chart QA, calibrated uncertainty, or accessibility results.

## Main result: untouched v3 test split

| Predictor | Macro F1, mean ± seed SD | Micro F1, mean ± seed SD | Macro AP, mean | Parameters |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(main_table)}

The main three-layer GATv2 has a modest positive seed-mean advantage over the graph-free MLP under this protocol. This supports further investigation of contextual structure with these oracle proposals and engineered geometry/color inputs. It does not establish better real-world chart understanding.

{chr(10).join(paired_lines)}

Intervals resample 300 paired chart clusters, 2,000 draws, seed {config['bootstrap_seed']}, and average paired differences over three fitted seeds. They condition on the fitted models, generator and one held-out style bundle: they **do not** include full training-population or unseen-domain uncertainty. Three seeds are not a large seed study. Individual seeds matter:

| Seed | MLP macro | GAT macro | Macro difference | MLP micro | GAT micro |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(seed_table)}

## Protocol and controls

The experiment protocol/configuration and implementation hashes were frozen before fitting. A documented amendment **before any test evaluation** added orchestration guards requiring a complete inventory and forbidding post-seal fitting/calibration, and clarified identity-derived, label-independent node permutation seeds. Model, training, metric, hyperparameter, seed, threshold-grid and bootstrap implementations/configurations did not change. Initial and amended snapshots are preserved under `experiments/milestone3/source_snapshots` and `protocol_history`; amendment notes are saved separately.

Inputs: the same 11 geometry/crop-RGB features and eight directed pair features for both models. GAT adds six-neighbor, geometry-derived within-chart connectivity, never target semantic adjacency. Oracle node types, numerical values, true relations, answers, evidence, IDs and split metadata are absent from forward inputs. A fixed random permutation removes canonical annotation ordering; targets remain candidate-aligned. Target/metadata mutation and masked-feature invariance tests pass. Candidate pairs remain directed, and graph boundaries are checked.

Both main models use AdamW lr {config['lr']}, weight decay {config['weight_decay']}, batch {config['batch_size']}, maximum {config['v3_steps']} steps, gradient clip {config['gradient_clip']}, dropout {config['dropout']}, and capped train-only BCE positive weights. GAT uses three 128-channel, four-head residual/LayerNorm layers. Main parameter counts differ by {(max(frozen['parameter_counts'].values())/min(frozen['parameter_counts'].values())-1)*100:.2f}%. The MLP has no message passing. Validation macro AP selects checkpoints; per-class thresholds are calibrated on validation only and saved/hash-sealed before test access. No test-based refits or threshold adjustments occurred. Every planned successful experiment is reported; no overall edge accuracy is used as the headline.

## Per-class test performance and support

F1 averages three seeds; supports are the same ground truth for all predictors. All ten classes are retained. See `per_class.csv` for every run's precision/recall/F1/AP and TP/FP/FN/TN, and `error_examples.csv` plus raw scored NPZ files for actual errors.

| Relation | Positive pairs | Negative pairs | MLP mean F1 | GAT mean F1 | Rule F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(class_table)}

`equal_to` remains unreliable: GAT mean F1 **{equal_gat:.4f}**, MLP **{equal_mlp:.4f}**. The graph model does not uniformly improve every relation. Geometry/color cannot always identify semantic roles, and equality is rare; high negative accuracy would conceal its false positives/misses. Capped positive weighting and validation thresholds mitigate imbalance but do not solve it. The fixed rule baseline abstains on `represents_series` and uses only approximate geometry/color heuristics; it is useful but not an exhaustive hand-engineered chart parser.

## Preplanned ablations

All ablations use seed 42, the same 600-step maximum and checkpoint/threshold rules. They are single-seed descriptions, not replicated significance claims.

| GAT variant | Macro F1 | Micro F1 | Macro AP |
| --- | ---: | ---: | ---: |
{ablation_table}

Removing explicit geometry hurts substantially. Its connectivity is also changed to complete so a spatial graph cannot leak coordinates; this coupled change prevents attributing the entire difference to feature masking alone. RGB crop means may still proxy region appearance/size. Removing color changes neither geometry nor spatial connectivity. Complete connectivity outperforms spatial-six-neighbor GAT for this seed, consistent with useful longer-range context, but that interpretation requires replicated follow-up. The winning ablation was not retroactively promoted to the main model after test evaluation.

## Pipeline verification versus larger development experiment

The preserved v2 corpus is **100 charts, 70/15/15** and was used first for one-batch and full small-data wiring checks. Both neural models completed 150 one-batch steps and 300 small-data steps. Their small-data test metrics are in the all-run table and are not real-world evidence.

{chr(10).join(overfit_lines)}

All neural gradient norms/losses were finite; checkpoint round trips and repeated evaluation were exactly equal in the recorded checks. Seeds/deterministic modes and CUDA workspace settings were used. Universal bitwise deterministic training was not established across devices or by full same-seed retraining; unsupported-kernel warnings are allowed and documented. A printed-loss scalar warning in logs did not affect gradients or results.

Only after v2 verification passed was **v3 generated: 2,000 charts, 1,400/300/300**. All image identities, annotation hashes, boxes, transforms, numerical values/relations, QA answers/evidence, and split invariants passed. No exact images or cross-split template/data identities collided; no v2 numerical arrays were reused. There were **{len(v3['leakage']['near_duplicates'])} near-hash candidates, {near['cross_split_candidates']} cross-split**. All candidate data hashes differ; all six cross-split pairs were visually reviewed as distinct values/layouts, and flags remain recorded. No automatic near-hash deduplication or false whole-corpus independence claim is made. Held-out font/palette/layout bundles differ from training. No v1/v2 drafts were merged into v3, and no further expansion occurred. Annotation test-label inspection was for integrity, never relation-model performance tuning.

## All successful experiments, resources and latency

| Run | Test macro F1 | Test micro F1 | Test macro AP | Selected step | Fit/calibration seconds | Peak train allocated MiB | Cached-graph median ms/device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{run_table}

Rule time is CPU validation/calibration, not optimization. Neural time includes GPU loading, fitting and validation scoring; feature-cache preparation is outside these timers and was not timed separately. Latency starts from cached observable tensors, includes batching/device transfer/scoring, and excludes image loading/feature extraction, output serialization, and metric computation. The evaluator also moves labels for scoring, but no targets enter forward. CPU rule and CUDA neural latency are not directly comparable speed claims. Allocated/reserved memory and checkpoint byte sizes are exported per run in `comparison.csv`; these are PyTorch measurements, not total device memory.

Hardware: {frozen['hardware']['gpu']}; PyTorch {frozen['hardware']['torch']}, CUDA {frozen['hardware']['cuda']}; float32 local training. New charged model-session wall time: **{ledger['charged_seconds']:.3f} s ({ledger['charged_seconds']/60:.2f} min)**. With the preserved historical ledger, lifetime charged time is **{lifetime/60:.2f}/60 min**. All reservations have actual completion times. `gpu_resource.csv`/`gpu_budget.json` include full session accounting, also charging test scoring/serialization overhead conservatively. No new downloads, paid APIs, external models, or frontend were used.

Full suite: **{tests_passed} passed**. **{integrity['protected_files']} historical/v2 files unchanged**, including prior baselines, reports, v2 data, old implementation and tests. No old report-generation script was run. The new report does not overwrite previous milestone reports.

## Artifacts and reproduction

- Frozen configs: `configs/relations.json`, `configs/structural_v3.json`; protocol: `research/milestone3_protocol.md`.
- Model/training/metric/CLI modules: `src/sensegraph/relation_*.py`.
- Run logs/checkpoints/validation thresholds/raw test scores/errors: `experiments/milestone3/runs/`.
- CSVs: `comparison.csv`, `per_class.csv`, `training_log.csv`, `error_examples.csv`, `ablations.csv`, `gpu_resource.csv` under `experiments/milestone3/`.
- Standalone figures: `experiments/milestone3/figures/learning_curves.png`, `ablations.png`.
- Visual review: `experiments/milestone3/visual_review.json`; sampled bar/grouped-bar/line overlays align with rendered marks and ticks. This sample review does not replace full automated annotation checks.
- Summary: `research/milestone3_results.json`; complete per-class measurements remain in each `test.json`.

The executed command sequence was `python -m sensegraph.relation_experiment freeze`, `verify-v2`, `expand`, `fit-v3`, final `freeze` after the documented guard amendment, `seal`, and `test`, using the local `.venv/Scripts/python.exe`. Commands and stdout logs are retained under `experiments/milestone3`. Re-running fitting in this sealed namespace is intentionally rejected; create a new protocol namespace for a new scientific study rather than resetting resource ledgers or modifying historical results. `load_checkpoint` plus `infer` expose cached-graph checkpoint inference; test files are reused on report regeneration, not reselected.

## Conclusion and stop boundary

There is a measurable but modest **synthetic oracle-proposal** graph benefit for the predefined main comparison, alongside significant relation-specific weaknesses and seed variation. The 2,000 charts come from a limited generator and one held-out test-style bundle; they do not demonstrate robustness to real-image proposals or learned perception errors. Real datasets' partial annotations, VisText transforms/dates, full QA, evidence reliability, uncertainty calibration and BLV usability remain unresolved/out of scope.

Stop here. The next milestone should investigate frozen learned visual features and imperfect region proposals with new controlled protocols and untouched evaluation data. No claim of improved UniChart/DePlot QA accuracy or accessibility follows from this relation experiment.
"""
    (ROOT / "research/milestone3_report.md").write_text(report,encoding="utf-8")
    status=f"""# SenseGraph project status

Updated {now}. **Milestone 3 complete: controlled oracle-proposal relation learning.** Stopped before learned perception and frontend work.

## Executed results

- v2: preserved 100 charts; both neural models passed one-batch gradient/loss checks, 300-step small-data fits, checkpoint reload and repeated inference verification.
- v3: independently generated/validated **2,000 charts (1,400/300/300)**. No v1/v2 data merged. Zero invalid annotations, exact duplicates, cross-split identity conflicts or reused v2 numerical arrays. 45 near-hash candidates (six cross-split) retained and reviewed as distinct-source cases.
- Three-seed main test means: **MLP macro F1 {means['mlp']['macro_f1']:.4f}, micro F1 {means['mlp']['micro_f1']:.4f}**; **GATv2 macro F1 {means['gat']['macro_f1']:.4f}, micro F1 {means['gat']['micro_f1']:.4f}**. Rule macro F1 {rule['macro_f1']:.4f}, micro F1 {rule['micro_f1']:.4f}.
- Paired chart-bootstrap GAT−MLP macro difference: **{intervals['macro_f1']['gat_minus_mlp']*100:+.3f} percentage points**, conditional 95% interval [{intervals['macro_f1']['ci95'][0]*100:+.3f}, {intervals['macro_f1']['ci95'][1]*100:+.3f}]. This conditions on three fitted seeds and generated charts, not unseen real domains.
- Preplanned geometry/color/connectivity ablations executed at seed 42; all successful results, rare-class support/errors, curves and resources reported. `equal_to` remains weak and is not omitted.
- **{tests_passed} tests passed**, including previous tests. All gradient/reload/repeated-inference checks passed. Historical/v2 integrity: **{integrity['protected_files']} files unchanged**.

## Implemented and evidence

- Fixed rule baseline, graph-free pair MLP and three-layer/four-head/128-channel GATv2; target-independent message connectivity and multi-hot directed labels.
- Train-only capped imbalance weights, identical step budgets, validation macro-AP checkpoint selection and validation-only threshold calibration. Test inventory sealed before evaluation; post-seal fitting/calibration rejected.
- Cached observable node inputs kept separate from targets; no true types, numeric values, semantic adjacency, answers or evidence used by forward methods.
- Configs/protocol: `configs/relations.json`, `configs/structural_v3.json`, `research/milestone3_protocol.md`.
- Actual report/results: `research/milestone3_report.md`, `research/milestone3_results.json`; run artifacts/CSVs/figures under `experiments/milestone3/`.
- New charged model sessions: **{ledger['charged_seconds']/60:.2f} min**; historical plus new total **{lifetime/60:.2f}/60 min**. No new downloads or paid APIs. Original GPU ledger remains immutable; new ledger accounts against its consumed time.

## Limits and next milestone

- Positive evidence is limited to **synthetic oracle-proposal structural prediction**. It does not establish real-image element extraction, chart QA improvements, calibrated uncertainty or accessibility.
- The graph-free baseline receives the same engineered information; inference still needs non-oracle proposals and richer visual/text features. Complete-connectivity ablation is promising but single-seed and not promoted after seeing test results.
- Next: learned visual features and imperfect proposals under a new frozen protocol and independent evaluation. Preserve the existing sealed results. No frontend or additional scaling/training is started after this milestone.
"""
    (ROOT / "PROJECT_STATUS.md").write_text(status,encoding="utf-8")
    print(f"Finalized Milestone 3: {len(rows)} successful evaluated experiments, {tests_passed} tests; lifetime GPU budget {lifetime/60:.2f}/60 min.")


if __name__ == "__main__":
    main()
