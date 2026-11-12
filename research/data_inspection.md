# Initial data inspection

Inspected 2026-11-03. Machine-readable reports live in `experiments/data_inspection.json` and `experiments/vistext_inspection.json`; source manifests include hashes and revisions.

## ChartQA

- Acquired 20 training charts with 40 associated questions, and the frozen 100-question validation pilot over 96 distinct images. Images decode and have positive dimensions; acquired CSV tables are rectangular and nonmissing.
- No exact image duplicates were found across these training and validation samples. This is a sample-level hash check, not a full-corpus leakage audit or a perceptual similarity test.
- Reviewed a contact sheet of all 20 training charts. The sample is dominated by Our World in Data horizontal bars and lines because training IDs were selected lexicographically. It is not representative of the complete dataset.
- Spot-checked the first three reference tables against their rendered labels: Nigeria poverty share 43.54 versus rounded 43.54% display; China's third-child ratio 159.36; Armenia HIV-attitude share 73.0 versus 73% display. These checks do not validate every cell.
- Some source CSV category headers are generic (`Country` for birth-order categories, for example). Preserve these labels; do not assume the column name reliably defines semantic type.
- GitHub's initial release does not contain chart-element annotations. Therefore element precision/recall, relation labels, and evidence box metrics are unavailable for this sample. Do not treat missing annotations as empty negative labels. The official full release is a separate future acquisition/inspection task.

## VisText

- Acquired official source split mappings, the training tabular member, and 20 selected training images with their hierarchical scenegraphs. Used HTTPS ZIP range reads; did not download test examples or full archives. Archive directories may list test members, but their payloads were not fetched.
- Normalized all 20 chosen IDs with matching image/scenegraph/tabular IDs; preserved the source captions and linearized tables. No reference table parser is claimed for the linearized VisText representation.
- Reviewed all 20 images in a contact sheet. They include bars, lines, and area charts, dense rotated labels, dark/low-contrast styles, and cropped long titles. Data-quality/visual complexity labels remain unannotated.
- **Date convention requires investigation:** example 1's chart/caption describe ticket prices beginning in 2006, while its linearized table begins `Dec 31, 2005 62.38`. Example 2 shows a similar prior-day/year boundary. This may reflect time-zone/serialization conventions; no correction was inferred or applied.
- Scenegraph values include negative/local coordinates and nested transforms. They are not validated raster-pixel boxes. The normalizer preserves the original tree rather than inventing detector supervision or edge ground truth.
- Source-host connections initially timed out on the NAT64-first route. Selective archive connections now prefer available IPv4 addresses while retaining normal HTTPS hostname/certificate validation; completed member hashes and source archive ETags are retained.

## Overlap and limits

Exact SHA256 comparison found no image overlap between the 20 acquired VisText images and the 116 acquired ChartQA images. Shared Statista data, rerendered images, or caption/semantic overlap are **not ruled out**. Official VisText Statista mappings are preserved for a later source-identity audit before graph-model training.

Contact sheets are local inspection artifacts, not model outputs or evidence-localization results. Unit/integration fixtures establish bounded software behavior; they do not establish annotation correctness or model quality.
