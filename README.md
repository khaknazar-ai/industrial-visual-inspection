# Industrial Visual Inspection

A computer-vision project for inspecting images and comparing anomaly-detection
models. It provides a local web interface, REST API and CLI for scores, heatmaps,
PASS/REVIEW/FAIL decisions and saved inspection reports.

The project supports Classical Diff, PatchCore, EfficientAD, DINOv3 memory and a
local SuperADD runtime. It builds on [Anomalib](https://github.com/open-edge-platform/anomalib)
and the [PatchCore method](https://github.com/amazon-science/patchcore-inspection).

- [Installation and launch](docs/RUN.md)
- [Experiment results](docs/RESULTS.md)

Datasets and model weights are separate assets; a fresh clone does not contain a
ready-to-use detector. Results are exploratory and depend on the image domain.
