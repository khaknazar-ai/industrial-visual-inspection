# Industrial Visual Inspection

A computer-vision project for inspecting images and comparing anomaly-detection
models. It provides a local web interface, REST API and CLI for scores, heatmaps,
PASS/REVIEW/FAIL decisions and saved inspection reports.

The project supports Classical Diff, PatchCore, EfficientAD, DINOv3 memory and a
local SuperADD runtime. It builds on [Anomalib](https://github.com/open-edge-platform/anomalib)
and the [PatchCore method](https://github.com/amazon-science/patchcore-inspection).

## Example inspections

These screenshots show DINOv3 memory inspecting previously used MVTec AD bottle
examples. The first image was classified as normal; the second was classified
as anomalous. The colored overlay highlights the model's anomaly signal.

| Normal example | Defect example |
|---|---|
| ![Normal bottle inspection](assets/showcase/inspection-normal.png) | ![Defective bottle inspection](assets/showcase/inspection-defect.png) |

The corresponding input photographs are in [assets/showcase](assets/showcase/).
Images are from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
by MVTec Software GmbH; the photographs and screenshots containing modified
versions are shared under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
These two examples illustrate the interface, not independent model evaluation.

- [Installation and launch](docs/RUN.md)
- [Experiment results](docs/RESULTS.md)

Datasets and model weights are separate assets; a fresh clone does not contain a
ready-to-use detector. Results are exploratory and depend on the image domain.
