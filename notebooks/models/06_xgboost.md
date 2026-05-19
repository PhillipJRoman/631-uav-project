---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: uav-efficiency-proj
    language: python
    name: python3
---

# Notebook 06: XGBoost

Binary classification of UAV energy efficiency using Extreme Gradient
Boosting (XGBoost). This notebook follows notebook 05 (linear baselines)
and precedes notebook 07 (Graph Attention Network).

## Goal

Train an XGBoost classifier on the 70-feature engineered dataset and
evaluate against the notebook 05 baselines:

- Persistence (naive): F1 = 0.9105, ROC AUC = 0.9261
- Lasso Logistic Regression: F1 = 0.8182, ROC AUC = 0.9239
- Elastic Net Logistic Regression: F1 = 0.8155, ROC AUC = 0.9229
- Linear Discriminant Analysis (LDA): F1 = 0.7956, ROC AUC = 0.9097

XGBoost is expected to beat the linear models. The open question is
whether it can beat the persistence baseline, which is strong because
lag-1 autocorrelation of the target is 0.838.

## Approach

- GroupKFold cross-validation with 6 folds, grouped by `flight_id`
- Hyperparameter tuning with Optuna (75 trials, Tree-structured Parzen
  Estimator sampler), early stopping inside each trial
- Mean F1 across folds as the Optuna objective
- Feature importance via built-in gain and permutation importance on test
- No feature scaling (XGBoost is invariant to monotonic transforms)
- Random seed: 631

```python
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use("tableau-colorblind10")

SEED = 631
N_TRIALS = 75
N_SPLITS = 6

REPO_ROOT = Path.cwd().parent
DATA_DIR = REPO_ROOT / "data" / "splits"
MODEL_DIR = REPO_ROOT / "models" / "06_xgboost"
RESULTS_DIR = REPO_ROOT / "results" / "06_xgboost"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"XGBoost version: {xgb.__version__}")
print(f"Optuna version: {optuna.__version__}")
print(f"Data dir: {DATA_DIR}")
print(f"Model dir: {MODEL_DIR}")
print(f"Results dir: {RESULTS_DIR}")
```


