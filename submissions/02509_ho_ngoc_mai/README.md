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

`results.xlsx`, `report.md`, `curves/`, `predictions/`, `runs/`, and `eval_out/`
are derived outputs. They are added only after their corresponding real Kaggle
runs complete; no metric table is pre-filled with estimates. The required
workbook sheets are `Backbones`, `Training`, `Inference`, `Final`, `PerClass`,
`Latency`, and `Summary`.

## Environment observed in the active Kaggle session

- Python 3.13.15
- PyTorch 2.11.0+cu128
- timm 1.0.29
- GPU: Tesla T4
