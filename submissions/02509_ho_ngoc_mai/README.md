# DeepWeeds — Lab Day 2

**Student:** Hồ Ngọc Mai  
**Student ID:** 02509  
**Kaggle notebook:** https://www.kaggle.com/code/vicsion/notebooke07ac6b761/edit

## Reproducibility

The experiments use DeepWeeds fold 0 without modifying its supplied CSV splits.
Run the notebook in order: environment/data verification → EDA → backbone screen →
training ablations → inference benchmarking → final three-seed evaluation.

The completed source lives in `code/`; logs, curves, predictions, tables and the
report are added only after the corresponding real Kaggle runs finish.

## Environment

- Python 3.13
- PyTorch 2.11.0+cu128
- timm 1.0.29
- GPU: Tesla T4
