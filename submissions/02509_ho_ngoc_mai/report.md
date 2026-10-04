# DeepWeeds Day 2 report — Hồ Ngọc Mai (02509)

## 1. Summary

This study compares image-classification backbones, controlled training-recipe
ablations, and inference methods for nine-class DeepWeeds classification. All
selection is made with fold-0 validation data. Final test results will be
reported only after the validation configuration is locked and each final seed
has a single full-test prediction file.

No performance number is stated in this report before it is generated from a
real Kaggle run. This avoids presenting planned experiments as measurements.

## 2. Data and setup

The dataset is DeepWeeds, 17,509 RGB images in nine classes. The experiment
uses the supplied fold-0 files without alteration. The verified split sizes are
10,501 train, 3,501 validation, and 3,507 test images; all pairwise filename
intersections are zero and their union contains 17,509 images.

The data archive MD5 is `b7b30f96d466fba86016aa5a26606e0f`. Training uses
random-resized crop and horizontal flip; validation/test use deterministic
centre crop and ImageNet normalization. The principal metric is macro-F1 across
all nine classes; top-1 accuracy, balanced accuracy, per-class F1, ECE, and
GPU-synchronised latency are secondary metrics.

The active Kaggle environment is Python 3.13.15, PyTorch 2.11.0+cu128, timm
1.0.29, and a Tesla T4 GPU. The EDA chart is written by the notebook to
`figures/eda_class_distribution.png`.

## 3. Pipeline checks

The code checks that focal loss with gamma zero equals cross-entropy, that
CutMix returns a valid mixing coefficient, and that Conv–BatchNorm fusion has
maximum output difference below `1e-5`. A further notebook check runs a
forward/backward pass and verifies that loss falls on a fixed small batch before
long training begins.

## 4. Backbone comparison

The validation-only screening set is B01 ResNet-50, B02 ResNeXt-50-32x4d, B03
ConvNeXt-Tiny, B04 DeiT-Small, and B05 MobileNetV3-Large. Every run uses the
same T00 recipe and seed 0. The completed `results.xlsx` will record each
model's pretrained tag, parameters, GMAC, validation macro-F1/top-1, epoch
time, latency, and its curve.

## 5. Controlled training recipe experiments

After selecting a backbone from Section 4 with validation data, T00–T08 test
initialisation, augmentation, loss, CutMix, and EMA one factor at a time. Each
comparison records the change from T00 and is interpreted conservatively: a
one-seed screening difference is not presented as proof of an improvement.

## 6. Inference and latency

Inference methods are evaluated only on validation data before final testing.
The required comparison includes I00 single-view plus at least four methods
among horizontal-flip/multicrop TTA, probability/logit aggregation, temperature
scaling fitted on validation, ensemble, EMA, FP16, and Conv–BatchNorm fusion
where architecture permits. Latency is measured with warmup and GPU
synchronisation and reports p50, p95, p99, batch size, precision, GPU, and
throughput.

## 7. Final protocol

The selected final configuration and the T00/I00 baseline are each trained with
seeds 0, 1, and 2. Test is opened once per seed only after all validation
choices are locked. `eval.py score` produces metrics and per-class tables; the
two groups are compared with `eval.py grade`. The final report will include
mean ± sample standard deviation, a confusion matrix, and precision/recall/F1
for Chinee Apple and Snake Weed.

## 8. Limitations

The supplied split is random rather than location-disjoint, so final scores can
be optimistic for different locations, seasons, illumination, or cameras. The
initial study uses one fold; results should be replicated with other supplied
folds before claiming broader generalisation.
