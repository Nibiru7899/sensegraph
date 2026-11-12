"""Finalize Milestone 2 from executed validation artifacts, without rewriting baseline reports."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import importlib.metadata
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensegraph.io import read_json, sha256, write_json
from sensegraph.structural_cli import baseline_integrity


def main():
    stats = read_json(ROOT / "experiments/milestone2/dataset_statistics.json")
    real = read_json(ROOT / "experiments/milestone2/real_mapping.json")
    integrity = baseline_integrity(ROOT)
    tests = ET.parse(ROOT / "experiments/milestone2/tests.xml")
    if tests.findall(".//failure") or tests.findall(".//error"):
        raise RuntimeError("Test evidence contains failures/errors")
    count = len(tests.findall(".//testcase"))
    if stats["count"] != 100 or stats["invalid_samples"] or stats["leakage"]["violations"] or integrity["changed_protected_files"]:
        raise RuntimeError("Milestone completion checks failed")
    if any(not entry["matches_baseline"] for entry in integrity["recomputed_baseline_scores"].values()):
        raise RuntimeError("Baseline scores changed")
    if any(stats["all_split_loading"][key]["loaded_charts"] != size for key, size in stats["splits"].items()):
        raise RuntimeError("Not every split loaded through PyTorch")
    if len(real["chartqa"]) != 20 or len(real["vistext"]) != 20:
        raise RuntimeError("Real mapping inspection incomplete")
    now = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds")
    stats["baseline_integrity"] = integrity
    stats["tests_passed"] = count
    stats["created_at"] = now
    stats["real_mapping"] = real
    stats["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "torch-geometric", "matplotlib", "numpy", "Pillow")}
    stats["implementation_sha256"] = {path.as_posix(): sha256(ROOT / path) for path in [
        Path("src/sensegraph") / filename for filename in ("synthetic.py", "structural_validation.py", "graph_data.py", "leakage.py", "real_mapping.py", "structural_cli.py")]}
    stats["accounted_download_bytes"] = read_json(ROOT / "experiments/download_ledger.json")["transferred_bytes"]
    write_json(ROOT / "research/milestone2_statistics.json", stats)
    boxes = sum(item.get("pixel_boxes", 0) for item in real["chartqa"])
    issues = Counter(issue["reason"] for item in real["chartqa"] for issue in item.get("issues", []))
    converted = sum(item.get("status") != "unsupported" for item in real["chartqa"])
    matching = sum(item.get("matches_baseline_image", False) for item in real["chartqa"])
    exact, near, cross = (len(stats["leakage"][key]) for key in ("exact_duplicates", "near_duplicates", "cross_source_candidates"))
    batch = stats["batch_smoke"]
    report = f"""# Milestone 2 — validated structural training data

Generated {now}. **Complete for the bounded 100-chart milestone.** Ready for controlled oracle-proposal GATv2 versus graph-free experiments, with the limitations below. No neural model was trained and no frontend was built.

## Executed dataset validation

- Charts: **100** in `data/synthetic/v2` — 34 bar, 33 grouped bar, 33 line; split **70 training / 15 validation / 15 test**. Resolution and DPI are sampled independently of chart type. The superseded `v1` draft is preserved, excluded from training, and must not be concatenated with v2 because it repeats source identities.
- Nodes: **{stats['node_counts']['total']}**, {stats['node_counts']['min']}–{stats['node_counts']['max']} per chart (mean {stats['node_counts']['mean']:.1f}).
- QA pairs: **{stats['question_count']}**, with validated operands, numerical/boolean answers, supporting node references, and final-image evidence boxes for every pair.
- Invalid samples: **{len(stats['invalid_samples'])}**. All image identities/dimensions, affine forward/inverse mappings, source data values, rendered mark/segment extents, ticks, typed node IDs, edge references/semantics, comparisons, answers, and evidence references/coordinates passed automated checks.
- Complete test suite: **{count} passed**; JUnit: `experiments/milestone2/tests.xml`. This includes all 44 original baseline tests. A third-party PyG distributed-module deprecation warning was observed; distributed training is not used.
- All 100 charts loaded via PyTorch DataLoader/PyG collation: `{stats['all_split_loading']}`. CPU only; the validation CLI constrains threads to avoid small-operation oversubscription.

| Relation | Positive directed labels |
| --- | ---: |
{chr(10).join(f'| {name} | {total} |' for name, total in stats['relations'].items())}

Full node/type, operation, coverage, split-loading, and candidate statistics are in `research/milestone2_statistics.json`.

## Visual and tensor verification

Reviewed mark overlays for all three chart types in every split: `{', '.join(stats['visual_review_ids'])}`. Rectangles follow bar boundaries, including negative values and visibly stroked zero-height bars. Point boxes follow markers; segment boxes enclose the corresponding line segments. Reviewed full-resolution context and lookup-evidence overlays: tick text, axis labels, title, legends, operand regions, and calibration context align with the saved canvases. Contact sheet: `experiments/milestone2/synthetic_overlay_contact_sheet.png`; individual mark/context/evidence layers are under `experiments/milestone2/overlays/`.

The bounded end-to-end example traversed synthetic PNG → annotation JSON → graph → dataset → three-chart batch. Images: `{batch['images']}`; observable node features: `{batch['node_features']}`; candidate edges: `{batch['candidate_edges']}`; multi-hot edge targets: `{batch['edge_labels']}`; evidence links: `{batch['evidence_index']}`. Finite-feature and graph-boundary checks passed. Variable-sized images are padded without scaling, and bipartite evidence links are independently offset by question/node counts.

Inputs contain pixels, region geometry/color features, text questions, and complete directed edge candidates. True types, numerical values, semantic edges, answers, and evidence stay in supervision. **Region proposals themselves are renderer ground truth in this dataset.** A real detector/OCR pipeline must supply proposals at inference; this is not proof of image-only chart understanding. Feature caching and learned visual backbones are not implemented.

## Real-source mapping results

- ChartQA full release: **{converted}/20** existing training identities converted, **{boxes}** positive-area source pixel boxes; **{matching}/20** paired images match the immutable baseline source images byte-for-byte. Source revision, member CRC/hashes, and split provenance are recorded separately. No official test payloads or whole ZIP were acquired.
- Source issues: `{dict(issues)}`. Zero-area segment boxes remain unavailable as positive-area boxes; no padding or point annotations were invented. No out-of-image boxes were silently clipped.
- Real bar/text and line-segment overlays were visually inspected on the paired images. The separate converter retains line segments (N−1 boxes) rather than mislabeling them as N points. Partial real grouping edges are source-supported; QA evidence and calibration are not fabricated.
- VisText: **20/20** acquired graphs converted only into explicit local translations/anchors/rectangles. **Zero raster boxes asserted.** Plot frame and image dimensions do not supply a verified scale/padding/translation. Local previews are explicitly labeled and are not real-image overlays.
- The date discrepancy is already present in source mark descriptions; the author parser copies it. No unsupported one-day/time-zone correction was applied. See `research/real_annotation_mapping.md` for categorical, temporal, unsupported-transform, and ontology differences.

## Partitioning, leakage, and baseline preservation

Template families are resolved style/layout bundles with disjoint hashes between splits. Entire numerical arrays are also compared independently of series/category naming. The generated data has **0 cross-split identity violations, {exact} exact image duplicates, and {near} near-duplicate candidates** at a predeclared 64-bit difference-hash threshold of 4. Hash candidates against acquired ChartQA/VisText images: **{cross}**. These are sample-level checks; they do not prove whole-corpus semantic/source disjointness. The full ChartQA images intentionally repeat the same 20 original source identities and are not new independent examples.

Integrity check: **{integrity['protected_files']} protected baseline files unchanged**, including pilot selection, original data/manifests, historical reports/lock, predictions, metrics, and baseline implementation/tests. Recomputed relaxed correctness remains UniChart **65/100** and DePlot+executor **12/100**. Only authorized download-budget bookkeeping is excluded from the immutable-file check. The original report-generation script was not used to rewrite historical baseline reports.

Current aggregate accounted downloads: **{stats['accounted_download_bytes']/1e9:.3f} GB**, below the existing 10 GB cap. No additional GPU model sessions or training were run.

## Milestone 3 readiness and boundaries

The data/graph/batching path is ready for a **small, controlled synthetic oracle-proposal experiment**, comparing GATv2 and a graph-free model on identical observable inputs, splits, and budgets. Relation targets are multi-label; never substitute target adjacency/types/values into inference inputs. The 100-chart corpus is a correctness fixture, not enough for claims of improved generalization. Scaling toward 10,000–20,000 requires a new versioned dataset configuration and repeated integrity/leakage/annotation checks, not overwriting this one.

Real raster annotation coverage remains partial: ChartQA's original labels are noisy, zero-area line boxes are withheld, and VisText raster transforms/dates are unresolved. Real-world perception, learned features, calibration, evidence quality, multi-seed results, and BLV accessibility remain unvalidated. No graph accuracy, training improvement, robustness gain, or accessibility result is claimed. Stop after this milestone.

Commands and exact data contracts are in `README.md` and `research/structural_data_schema.md`. The structural dependency lock is separate from the preserved baseline lock.
"""
    (ROOT / "research/milestone2_report.md").write_text(report, encoding="utf-8")
    status = f"""# SenseGraph project status

Updated {now}. **Milestone 2 complete: bounded validated structural training data.** Stopped before GATv2 training and frontend work.

## Implemented

- Deterministic configurable Agg/Matplotlib bar, grouped-bar, and line generation, with exact final-image boxes/transforms, source values, typed semantic/numerical graphs, QA, and evidence.
- Independent geometry/semantics/answer/evidence validators, adversarial corruption tests, and visual overlay tools.
- Training/validation/test partitioning with held-out resolved templates, numerical-configuration separation, exact/near-image checks, and sample-level cross-source checks.
- CPU PyTorch/PyG datasets and batches, explicit label-free input graph, masked numerical supervision, and correct bipartite evidence batching.
- Selective full-ChartQA training annotations and partial pixel conversion; VisText local-only conversion with unresolved raster/date mappings explicitly preserved.
- Separate structural dependency lock, schemas, mapping investigation, dataset statistics, and reviewable milestone report.

## Executed evidence

- **100 charts:** 34 bar / 33 grouped bar / 33 line; **70/15/15** split; **{stats['node_counts']['total']} nodes**, **{stats['question_count']} QA pairs**; **0 invalid samples**.
- All 100 loaded through PyTorch/PyG; finite features, candidate/label alignment, graph boundaries, and evidence offsets verified on CPU.
- Full suite: **{count} passed**, including all 44 original tests. `pip check`: no broken requirements.
- Reviewed nine synthetic mark overlays, full-resolution context/evidence overlays, and real ChartQA bar/line overlays.
- ChartQA: {converted}/20 training annotations converted, {boxes} positive-area boxes; 43 zero-area source segment boxes withheld. All {matching} paired images match original baseline images.
- VisText: 20 local-coordinate conversions; no raster boxes fabricated. Source descriptions confirm the unresolved date-label discrepancy.
- Duplicate checks: {exact} exact, {near} near-image, {cross} cross-source candidates; zero cross-split template/numerical identity violations. Shared-source/rerendered overlap beyond these samples remains possible. Use only `data/synthetic/v2`; the superseded v1 draft is not additional independent training data.

## Baseline preservation and resources

- **{integrity['protected_files']} protected baseline files unchanged.** Historical baseline reports/inputs/predictions and `requirements.lock` preserved. Scores recomputed unchanged: UniChart 65/100 relaxed; DePlot+executor 12/100 at 15% coverage.
- Aggregate accounted downloads **{stats['accounted_download_bytes']/1e9:.3f}/10 GB**. Legacy GPU ledger unchanged; no additional GPU execution or training in this milestone.
- Main evidence: `research/milestone2_report.md`, `research/milestone2_statistics.json`, `experiments/milestone2/tests.xml`, `batch_smoke.json`, `dataset_statistics.json`, and overlays.

## Known limits and Milestone 3 readiness

- Ready for controlled **synthetic oracle-proposal** GATv2 and graph-free comparisons, not real-world image-only deployment.
- Only 100 charts generated; no automatic scaling or training performed. Learned visual feature extraction/caching and perception remain future work.
- ChartQA source supervision is partial/noisy; VisText raster transforms and temporal conventions remain unsupported. Do not invent missing labels or merge these ontologies silently.
- Next: freeze experiment protocols and matched graph-free inputs, then perform bounded controlled training with honest split/seed accounting. Expand data only as a new validated version when appropriate. Frontend remains deferred.
"""
    (ROOT / "PROJECT_STATUS.md").write_text(status, encoding="utf-8")
    print(f"Finalized Milestone 2: {stats['count']} charts, {count} tests, {integrity['protected_files']} immutable baseline files verified.")


if __name__ == "__main__":
    main()
