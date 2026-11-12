# SenseGraph literature review

Reviewed 2026-11-03. Sources below are official papers, author sites, or author repositories. Descriptions of published evaluations are historical evidence, not SenseGraph results. Availability means the linked resource was identified; it does not mean its code or full dataset was executed locally. Unknown licenses are explicitly unresolved and are not permission to redistribute.

## Research position

Hypothesis: learned chart relationships, numerical consistency checks, and evidence-aware uncertainty may improve robustness and reduce unsupported answers. This requires matched ablations. Graph-based chart reasoning and accessible graph navigation already exist: neither graphs nor accessibility alone constitute a novelty claim. This milestone evaluates pretrained baselines only.

## ChartOCR (WACV 2021)

- **Identity/source:** Luo, Li, Wang, Lin, [ChartOCR: Data Extraction from Charts Images via a Deep Hybrid Framework](https://www.microsoft.com/en-us/research/publication/chartocr-data-extraction-from-charts-images-via-a-deep-hybrid-framework/), January 2021; [paper](https://www.microsoft.com/en-us/research/uploads/prod/2020/12/WACV_2021_ChartOCR.pdf).
- **Problem/method:** chart-to-data extraction; learned keypoint detection plus chart-specific rules and OCR-based coordinate calibration.
- **Code/data:** author-linked [DeepRule](https://github.com/soap117/DeepRule); ExcelChart400K, FQA, WebData. Code/dataset license and live checkpoint availability not independently verified in this pilot.
- **Metrics:** chart-specific matching scores; line precision/recall/F1 using interpolation; mean extraction error on prior datasets.
- **Limits/relevance:** keypoint and OCR errors propagate; original OCR depends on Microsoft services. Useful numerical perception baseline, but not end-to-end grounded QA or an accessibility validation.

## ChartQA (Findings of ACL 2022)

- **Identity/source:** Masry et al., [ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://aclanthology.org/2022.findings-acl.177/).
- **Problem/method:** visual/logical chart QA using image and tabular representations; T5, VL-T5, and VisionTaPas implementations.
- **Code/data:** [official repository](https://github.com/vis-nlp/ChartQA), GPL-3.0; author-linked [full release](https://huggingface.co/datasets/ahmed-masry/ChartQA). Human and augmented questions; images, tables, and full-release element annotations. GitHub subset lacks annotations. Preserve original splits and third-party chart provenance.
- **Metrics:** relaxed answer accuracy, with numeric tolerance and textual matching; separate human/augmented evaluation.
- **Limits/relevance:** authors explicitly acknowledge noisy/missing SVG-derived annotations and heuristic/manual annotation. Pilot uses validation only; missing labels cannot supervise element or evidence losses.

## DePlot (ICML 2023; preprint December 2022)

- **Identity/source:** Liu et al., [DePlot: One-shot visual language reasoning by plot-to-table translation](https://arxiv.org/abs/2212.10505).
- **Problem/method:** Pix2Struct chart-to-table translation followed by language-model reasoning.
- **Code/data:** [official model](https://huggingface.co/google/deplot), Apache-2.0; [Transformers implementation and usage](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/deplot). Plot-to-table supervision and ChartQA evaluation are central to the paper.
- **Metrics:** table extraction and downstream chart QA accuracy; inspect the paper's exact table metric before claiming reproduction.
- **Limits/relevance:** malformed or incorrect tables can yield unsupported conclusions. SenseGraph uses a conservative deterministic executor, so its pilot is a modified baseline, not the paper's LLM reasoning reproduction.

## UniChart (EMNLP 2023)

- **Identity/source:** Masry et al., [UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning](https://aclanthology.org/2023.emnlp-main.906/).
- **Problem/method:** chart-specific vision encoder/text decoder pretrained on low- and high-level chart tasks; finetuned for QA, summarization, and table extraction.
- **Code/data:** [official repository](https://github.com/vis-nlp/UniChart), MIT code; [ChartQA checkpoint](https://huggingface.co/ahmed-masry/unichart-chartqa-960), GPL-3.0 model metadata. Task-specific checkpoints and author-hosted pretraining datasets are available.
- **Metrics:** QA relaxed accuracy and task-specific summarization/extraction metrics.
- **Limits/relevance:** direct-answer baseline provides no guaranteed visual evidence. Authors pin Transformers 4.28.1 and warn about performance drift. The agreed 4.57.3 pilot must be labeled a modern compatibility implementation rather than exact paper reproduction. ChartQA-finetuned checkpoint performance is in-domain, not evidence of unseen-domain generalization.

## VisText (ACL 2023)

- **Identity/source:** Tang, Boggust, Satyanarayan, [VisText: A Benchmark for Semantically Rich Chart Captioning](https://vis.csail.mit.edu/pubs/vistext/).
- **Problem/method:** semantically controlled chart descriptions using tables, scene graphs, images, and semantic prefix tuning.
- **Code/data:** [official repository](https://github.com/mitvis/vistext), GPL-3.0 code; official tabular/image/scenegraph archives linked in its download script. Captions combine synthetic elemental descriptions and human trend/statistical descriptions. Statista source mappings are supplied; original chart rights are not replaced by the code license.
- **Metrics:** caption-generation automatic metrics and semantic-level analysis; official evaluation scripts require separate dependency inspection before reproduction.
- **Limits/relevance:** linearized/hierarchical scene graphs are not automatically detector-ready pixel boxes. Validate scenegraph/image/data correspondence and shared-source overlap with ChartQA before using training supervision.

## CharXiv (NeurIPS 2024)

- **Identity/source:** Wang et al., [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://charxiv.github.io/); [paper](https://arxiv.org/abs/2406.18521).
- **Problem/method:** human-curated scientific charts with descriptive and reasoning questions, addressing the simplicity of earlier benchmarks.
- **Code/data:** [official repository](https://github.com/princeton-nlp/CharXiv), author-linked dataset; [QA data license](https://github.com/princeton-nlp/CharXiv/blob/main/data/LICENSE) CC BY-SA 4.0. Underlying scientific figures retain their own rights; code license must be checked independently before reuse.
- **Metrics:** descriptive and reasoning answer accuracy using benchmark-specific grading.
- **Limits/relevance:** complex scientific charts test distribution shift; judge-based grading and prompt/version sensitivity require documentation. Reserved for future evaluation, not pilot training or tuning.

## ChartQAPro (Findings of ACL 2025)

- **Identity/source:** Masry et al., [ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering](https://aclanthology.org/2025.findings-acl.978/); [preprint](https://arxiv.org/abs/2504.05506).
- **Problem/method:** more varied real charts and question forms, including conversation, hypothetical, multi-chart, and unanswerable questions.
- **Code/data:** [official repository](https://github.com/vis-nlp/ChartQAPro), author-hosted Hugging Face release. Dataset/code license not verified for local acquisition; evaluation-only candidate.
- **Metrics:** overall and question-type answer accuracy, error analyses and ablations.
- **Limits/relevance:** answers can require context absent from a single extracted table. Useful later for abstention and out-of-domain evaluation; exclude unsupported tasks from claims about the pilot's arithmetic executor.

## Benthic (ASSETS 2025)

- **Identity/source:** Mei, Pollock, Hajas, Zong, Satyanarayan, [Benthic: Perceptually Congruent Structures for Accessible Charts and Diagrams](https://data-and-design.org/publications/benthic/), DOI 10.1145/3663547.3746342.
- **Problem/method:** screen-reader structures that capture containment and adjacency through hypergraphs and reversible traversal.
- **Code/data:** author page associates the work with Olli; a standalone implementation, reusable benchmark dataset, and corresponding licenses were not verified here.
- **Metrics:** task-based study with blind participants, qualitative exploration/structural understanding findings; not a chart-perception QA leaderboard.
- **Limits/relevance:** graph navigation already has accessibility precedent. Learned extraction from raster images and reliable evidence must be independently established. Do not equate model QA accuracy with usable navigation or validated BLV outcomes.

## ChartR (CVPR 2026)

- **Identity/source:** Chen et al., [ChartR: Evaluating Reasoning Accuracy and Robustness in Chart Question Answering](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_ChartR_Evaluating_Reasoning_Accuracy_and_Robustness_in_Chart_Question_Answering_CVPR_2026_paper.pdf).
- **Problem/method:** decomposed chart reasoning chains and controlled blur, noise, watermark, and annotation-removal perturbations.
- **Code/data:** official CVF paper identified; independent code/data release and license not verified. A later direct PDF request returned HTTP 403, so this is a partial paper review using retrieved official text, not an executed reproduction.
- **Metrics:** stepwise/chain correctness and robustness across visual perturbations; exact scorer definitions require checking the released implementation before use.
- **Limits/relevance:** generated charts/templates can impose distribution bias. Useful future robustness protocol; never train on its evaluation questions.

## DRAGON (2026)

- **Identity/source:** Iyengar et al., [DRAGON: A Benchmark for Evidence-Grounded Visual Reasoning over Diagrams](https://arxiv.org/abs/2604.25231), submitted April 28, revised September 27, 2026; [official project](https://coral-lab-asu.github.io/DRAGON/index.html).
- **Problem/method:** given a diagram, question, and verified answer, localize supporting visual regions. Direct grounding, staged element grounding, and verification prompting are compared.
- **Code/data:** project links code and dataset; only the human-verified test release is described as currently available, with train/validation to follow. License verification and artifact acquisition are deferred.
- **Metrics:** Max-Pair IoU, union-region Grounding IoU, and box F1.
- **Limits/relevance:** answer-conditioned grounding differs from generating an answer and its evidence jointly. Its ChartQA-derived examples also create overlap risk; keep held out. Text explanations do not substitute for evidence boxes.

## GraphWhisper / Making Charts Speak (2026)

- **Identity/source:** [CHI 2026 program entry: Making Charts Speak: LLM-Based Conversational Chart Question Answering for Blind and Low-Vision Users](https://programs.sigchi.org/chi/2026/program/content/230277), Das, Tarun, Mueller; describes GraphWhisper. An [author publications page](https://amitkdas-08.github.io/publications.html) separately lists a GraphWhisper manuscript under TVCG review.
- **Problem/method:** image-based conversational chart access using chart-specific LLM prompting, without requiring prestructured input tables.
- **Code/data:** no standalone official repository, downloadable evaluation dataset, or reusable license verified in this review.
- **Metrics:** reported visualization-literacy accuracy, user-question accuracy, and participant ratings. These are author-reported results, not local reproductions.
- **Limits/relevance:** the CHI poster and under-review manuscript should not be silently treated as the same archival publication. Its name does not establish that it uses a learned GNN. Relevant accessibility precedent; requires comparison of task scope, costs, evidence, and human usability.

## Additional graph precedent: MSG-Chart (CIKM 2024)

[Dai, Han, Liu: MSG-Chart: Multimodal Scene Graph for ChartQA](https://arxiv.org/abs/2408.04852), [paper](https://openreview.net/pdf?id=3gdA8Yw3P9), combines visual/textual graphs and injects structural/semantic information into chart QA backbones, including UniChart and VL-T5. Evaluates ChartQA and OpenCQA with task-appropriate answer metrics. Official executable code and reuse license were not verified. Its existence directly rules out claiming scene-graph chart QA as novel by itself. SenseGraph must test specific learned relations, deterministic consistency, grounding, and uncertainty through controlled ablations.

## Consequences for experiments

1. Do not infer GNN improvement, evidence fidelity, calibration, or accessibility usability from this pilot.
2. Use the same frozen records for baseline comparison; document lexical executor eligibility and count failures.
3. Keep reference-table diagnostics separate from image-only inference.
4. Audit VisText/ChartQA shared sources and image hashes before later training/evaluation.
5. Missing annotations disable corresponding losses; heuristic labels must remain explicitly labeled.
6. Resolve unavailable artifacts/licensing before their later acquisition; no restricted downloads or access-control bypasses.
