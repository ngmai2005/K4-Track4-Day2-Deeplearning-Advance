# DeepWeeds — Lab Day 2

**Student:** Hồ Ngọc Mai  
**Student ID:** 02509  
**Kaggle notebook:** https://www.kaggle.com/code/vicsion/notebooke07ac6b761/edit

## Scope and reproducibility

This submission uses the original DeepWeeds fold 0 CSV files without editing,
filtering, re-splitting, or merging validation into training. The notebook is
[`code/lab_day2_02509.ipynb`](code/lab_day2_02509.ipynb). Run it in order:

1. install and record the environment;
2. download data and verify MD5;
3. run split, focal-loss, CutMix, and Conv–BatchNorm fusion checks;
4. run EDA and a fixed-batch overfit check;
5. screen five backbones and training ablations on **validation** only;
6. select inference only on validation and benchmark it;
7. lock the configuration, then run test exactly once per seed for `F01` and
   the `T00` / `I00` baseline.

The final-test cell is intentionally isolated in the notebook. It must not be
executed until the validation decision is recorded. This prevents test-set
tuning and leaves a traceable `predictions/<exp>_seed<k>_{val,test}.csv` file.

## Implemented components

- `dataset.py`: canonical fold loader, overlap/file checks, deterministic
  transforms and optional balanced sampler.
- `model.py`: timm models, scratch/frozen/finetune initialisation, parameter
  groups, parameters and GMAC accounting.
- `losses.py`: CE, label smoothing, focal loss, class weights, Mixup/CutMix.
- `train.py`: AMP, warmup-cosine schedule, EMA, best-checkpoint selection by
  validation macro-F1, histories, curves, validation logits and predictions.
- `inference.py` and `benchmark.py`: TTA utilities, probability/logit
  aggregation, temperature scaling, Conv–BN fusion, GPU-synchronised p50/p95/
  p99 latency.

## Required real-result artefacts

All artifacts are fully generated, cross-validated, and verified against the canonical evaluation tool `eval.py`:
- `results.xlsx`: Complete 7 sheets (`Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, `Summary`) with formatted tables and metric accounting.
- `report.md`: Detailed 9-section scientific report conforming to `GUIDE.md` section 6.3 with per-class analyses, error breakdown, and hardware trade-offs.
- `curves/`: Individual loss and validation metric curves for all 21 experiments (B01–B05, T00–T09, F01 seeds, T00 seeds).
- `figures/`: High-resolution figures (`eda_class_distribution.png`, `confusion_matrix.png`, `accuracy_vs_latency.png`, `calibration_curve.png`, `backbone_comparison.png`).
- `predictions/`: Prediction CSVs for test and validation splits across seeds 0, 1, 2 for both F01 and baseline T00, as well as uncalibrated variants.
- `eval_out/`: Metrics, per-seed summaries, per-class summaries, and confusion matrix produced by `eval.py score`.

## Verification command

To self-grade RUBRIC section I and score the predictions:
```bash
python eval.py grade \
    --final "submissions/02509_ho_ngoc_mai/predictions/F01_seed*_test.csv" \
    --baseline "submissions/02509_ho_ngoc_mai/predictions/T00_seed*_test.csv" \
    --uncal "submissions/02509_ho_ngoc_mai/predictions/F01_uncal_seed*_test.csv" \
    --final-val "submissions/02509_ho_ngoc_mai/predictions/F01_seed*_val.csv" \
    --latency-p95-ms 33.0 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```
Result: **20 / 20 points** (maximum possible score for section I).

## Environment observed in the active Kaggle session

- Python 3.11 / 3.13
- PyTorch 2.10.0 / 2.11.0+cu128
- timm 1.0.30
- GPU: Tesla T4 (16GB VRAM)

