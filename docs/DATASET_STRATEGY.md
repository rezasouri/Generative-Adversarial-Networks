# Industrial dataset strategy

## Decision

Use **VisA as the first implementation target** and **MVTec AD as the second benchmark**.

This ordering balances the project's portfolio/research goals with dataset licensing and implementation complexity. VisA's official registry entry lists CC BY 4.0, while MVTec AD is CC BY-NC-SA 4.0 and explicitly excludes commercial use. The project code's MIT license does not override either dataset license.

## Dataset comparison

| Property | VisA | MVTec AD |
|---|---|---|
| Purpose | Industrial visual anomaly detection and segmentation | Industrial anomaly detection and localization |
| Scale | 10,821 images | 5,354 images |
| Categories | 12 object subsets | 15 object and texture categories |
| Labels | Image-level labels and pixel-level annotations | Normal/defect labels and pixel-precise anomaly masks |
| Training protocol | Official split CSVs and one-class / few-shot preparations | Defect-free training images; mixed normal/anomalous test images |
| Dataset license | CC BY 4.0 | CC BY-NC-SA 4.0 |
| Access | Public AWS Open Data S3 archive | Official MVTec download form |

VisA counts: 9,621 normal and 1,200 anomalous images. Its official tree includes per-object `Data/Images/{Normal,Anomaly}`, `Data/Masks/Anomaly`, and `image_anno.csv`. Normal-image masks are not stored. The official repository provides split CSVs and a preparation utility for common one-class and few-shot protocols.

MVTec AD has 15 object/texture categories, more than 5,000 high-resolution images and pixel-precise anomaly annotations. Its official page provides an evaluation code package and requires a form for dataset download.

## Licensing and redistribution rules

- **VisA:** follow CC BY 4.0 attribution requirements. Preserve the dataset citation and license notices in benchmark documentation. Do not imply that the project owns the data.
- **MVTec AD:** CC BY-NC-SA 4.0; commercial use is not permitted under the published terms. If the intended use may be commercial, obtain clarification/permission from MVTec before use.
- Do not commit either dataset, extracted images, masks, archives or generated samples derived from them to this repository.
- Dataset terms apply independently of the MIT license covering this repository's original code. This document is an engineering note, not legal advice.

## Implementation order

1. Build a VisA adapter that reads the original annotation CSV and maps image paths, image-level labels, anomaly types and mask paths without rewriting the source archive.
2. Validate the official split CSVs and ensure no source image crosses train/validation/test boundaries.
3. Add a normalized internal sample record: image path, category, split, anomaly label, optional mask path, and source identifier.
4. Add the MVTec AD adapter using its official train/test/ground-truth directory layout.
5. Run `forgegan audit-dataset` after extraction and store the JSON report with each experiment's metadata.
6. Keep benchmark loaders read-only; preprocessing outputs must go to a separate derived-data directory.

## Official sources

- [MVTec AD official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) — dataset scale, category structure, annotations, download form and CC BY-NC-SA 4.0 terms.
- [VisA official AWS Open Data entry](https://registry.opendata.aws/visa/) — dataset scale, annotations, archive location and CC BY 4.0 license.
- [Amazon Science SPot-the-Difference repository](https://github.com/amazon-science/spot-diff) — original VisA description, directory layout, split CSVs, preparation utility and dataset citation.
- Zou et al., “SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation,” ECCV 2022.
- Bergmann et al., “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019.

## Recommended benchmark scope

Start with a focused subset of VisA for adapter and training smoke tests, then run the complete 12-subset benchmark only after the pipeline is stable. Use MVTec AD as a second independent benchmark to demonstrate that the pipeline is not tied to one dataset layout.
