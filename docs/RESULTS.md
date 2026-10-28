# Industrial inspection demo — evidence summary

**Scope:** exploratory demonstration evidence for 26 October 2026. These results are not
an independent product qualification and do not support a production accuracy claim.

## MVTec AD matrix

Four families were evaluated across five categories. Training used normal train images,
calibration used normal validation images, and the frozen test split was evaluated once.
The bottle set has 83 images and was reused from earlier exploratory work. Other rows use
the structured MVTec AD mirror recorded in the run manifests. MVTec AD is released under
CC BY-NC-SA 4.0; the dataset images and model bundles are not included here.

| Category | Family | Test n | Image AUC | Image AP | Image F1 | Pixel AUC | Pixel AP* | Dice |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | Classical Diff | 83 | 0.862 | 0.958 | 0.694 | 0.825 | 0.264 | 0.151 |
| bottle | PatchCore | 83 | 1.000 | 1.000 | 0.976 | 0.984 | 0.757 | 0.524 |
| bottle | EfficientAD | 83 | 1.000 | 1.000 | 0.992 | 0.977 | 0.788 | 0.707 |
| bottle | DINOv3 memory | 83 | 0.998 | 0.999 | 0.992 | 0.982 | 0.759 | 0.241 |
| capsule | Classical Diff | 132 | 0.646 | 0.906 | 0.336 | 0.863 | 0.217 | 0.232 |
| capsule | PatchCore | 132 | 0.972 | 0.993 | 0.725 | 0.987 | 0.428 | 0.320 |
| capsule | EfficientAD | 132 | 0.687 | 0.905 | 0.036 | 0.854 | 0.241 | 0.321 |
| capsule | DINOv3 memory | 132 | 0.930 | 0.984 | 0.764 | 0.981 | 0.373 | 0.177 |
| carpet | Classical Diff | 117 | 0.451 | 0.762 | 0.086 | 0.540 | 0.019 | 0.001 |
| carpet | PatchCore | 117 | 0.961 | 0.983 | 0.951 | 0.987 | 0.455 | 0.160 |
| carpet | EfficientAD | 117 | 0.978 | 0.994 | 0.960 | 0.943 | 0.661 | 0.492 |
| carpet | DINOv3 memory | 117 | 0.999 | 1.000 | 0.977 | 0.980 | 0.649 | 0.063 |
| hazelnut | Classical Diff | 110 | 0.892 | 0.944 | 0.614 | 0.940 | 0.469 | 0.431 |
| hazelnut | PatchCore | 110 | 1.000 | 1.000 | 1.000 | 0.988 | 0.572 | 0.477 |
| hazelnut | EfficientAD | 110 | 0.851 | 0.914 | 0.642 | 0.838 | 0.389 | 0.512 |
| hazelnut | DINOv3 memory | 110 | 0.992 | 0.996 | 0.971 | 0.993 | 0.803 | 0.457 |
| screw | Classical Diff | 160 | 0.245 | 0.613 | 0.000 | 0.842 | 0.012 | 0.027 |
| screw | PatchCore | 160 | 0.959 | 0.983 | 0.632 | 0.980 | 0.363 | 0.148 |
| screw | EfficientAD | 160 | 0.888 | 0.960 | 0.494 | 0.973 | 0.377 | 0.320 |
| screw | DINOv3 memory | 160 | 0.776 | 0.891 | 0.111 | 0.886 | 0.070 | 0.084 |

*Pixel AP is the evaluator's 65,536-bin histogram estimate; its run JSON records the
associated approximation bound. Full per-cell receipts are retained in the local experiment artifacts.

## External bottle and assembly experiments

Both models were trained on normal images, calibrated on validation, and evaluated
on held-out images. AUROC describes ranking, while the false-positive rate reflects
the validation-selected operating threshold.

| Dataset | Model | Test images | Image AUROC | Normal false positives |
|---|---|---:|---:|---:|
| Ron88 bottles | Classical Diff | 300 | 0.676 | 121 / 121 |
| Ron88 bottles | DINOv3 memory | 300 | 0.816 | 100 / 121 |
| PCAD switch assembly | Classical Diff | 474 | 0.505 | 58 / 58 |
| PCAD switch assembly | DINOv3 memory | 474 | 0.519 | 55 / 58 |

Ron88 has 600 normal and 900 defective external photographs. The PCAD experiment
uses one category with 288 normal and 521 defective external images. The exact
physical-item and session independence of these splits is not established.
The above operating points have high false-positive rates and are not reliable
automatic acceptance settings. Scores are not probabilities.

## SuperADD on MVTec LOCO

The local SuperADD runtime was evaluated on the held-out `splicing_connectors`
test split: 312 images, including 119 normal and 193 anomalous images. The
image threshold was selected on validation data and fixed at 0.5 in the
normalized score space.

| Image AUROC | Average precision | F1 | Defect recall | Normal false-positive rate |
|---:|---:|---:|---:|---:|
| 0.758 | 0.855 | 0.673 | 55.4% | 15.1% |

At this threshold, 107 defects were detected, 86 were missed, 101 normal images
were accepted and 18 were falsely flagged. This is an exploratory result on
one LOCO category, not a reproduction of official high-resolution SuperADD or
an MVTec AD2 evaluation.

## Scope

These results are exploratory, not independent production qualification. The
bottle MVTec set was reused, and results on one dataset do not establish
performance on other products. The repository contains two illustrative input
images and corresponding screenshots; full datasets and model weights are not
included. Third-party dataset and model terms apply.
