---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: uav-efficiency-proj
    language: python
    name: python3
---

# Notebook 08: Graph Attention Network + XGBoost Hybrid

Binary classification of UAV energy efficiency using a hybrid model:
256D Graph Attention Network (GAT) embeddings concatenated with the
70 engineered features, fed to XGBoost. This notebook follows
notebook 07 (Graph Attention Network).

## Goal

Test whether the GAT embeddings carry signal that complements the
raw engineered features. Three outcomes are possible:

- **The hybrid beats both parents.** GAT embeddings encode structure
  the raw features miss. Best case.
- **The hybrid matches XGBoost.** The embeddings re-encode information
  XGBoost already extracts directly. The graph structure adds nothing
  beyond what gradient boosting on raw sensors can find.
- **The hybrid underperforms XGBoost.** The embeddings add noise that
  XGBoost cannot filter out, or the wider feature space causes
  overfitting that Optuna cannot tune away.

Test set comparison targets:

- Persistence (naive): F1 = 0.9105, ROC AUC = 0.9261
- XGBoost: F1 = 0.8692, ROC AUC = 0.9634
- Lasso Logistic Regression: F1 = 0.8182, ROC AUC = 0.9239
- Elastic Net Logistic Regression: F1 = 0.8155, ROC AUC = 0.9229
- Linear Discriminant Analysis: F1 = 0.7956, ROC AUC = 0.9097
- Graph Attention Network: F1 = 0.7241, ROC AUC = 0.8556

XGBoost on raw features is the model to beat. The hybrid's job is to
prove the GAT learned something the raw features did not already
expose.

## Caveat on embedding extraction

The 256D embeddings come from a single fold's GAT model in notebook
07. That model was trained on 5 of 6 folds, so roughly 5/6 of the
training rows were in-sample to the GAT when their embeddings were
generated. This introduces mild label leakage into the training-side
embedding features.

The leakage is small in practice. The GAT's train F1 and test F1
matched within 0.008, so the model was not memorizing training rows.
Cross-validation F1 in this notebook may be slightly optimistic; test
F1 is unaffected because the 17 test flights were never seen by the
GAT or anything downstream. Test F1 is the honest comparator across
all six models.


## Approach

- Concatenate 256D Graph Attention Network embeddings with the 70
  engineered features for a 326-feature hybrid input
- GroupKFold cross-validation with 6 folds, grouped by `flight_id`
- Hyperparameter tuning with Optuna (75 trials, Tree-structured Parzen
  Estimator sampler), early stopping inside each trial
- Mean F1 across folds as the Optuna objective
- Feature importance via built-in gain and permutation importance on
  test, with attention to how the model splits credit between embedding
  features and raw features
- No feature scaling (XGBoost is invariant to monotonic transforms)
- Search space matches notebook 06, with `colsample_bytree` lower bound
  dropped from 0.6 to 0.4 to give the model more flexibility in
  sampling from the wider 326-feature space per tree
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

REPO_ROOT = Path("/Users/phillipromanmacbook/GitHub/631-uav-project")
DATA_DIR = REPO_ROOT / "data" / "splits"
EMB_DIR = REPO_ROOT / "data" / "embeddings" / "07_gat"
MODEL_DIR = REPO_ROOT / "models" / "08_hybrid"
RESULTS_DIR = REPO_ROOT / "results" / "08_hybrid"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"XGBoost version: {xgb.__version__}")
print(f"Optuna version: {optuna.__version__}")
print(f"Data dir: {DATA_DIR}")
print(f"Embeddings dir: {EMB_DIR}")
print(f"Model dir: {MODEL_DIR}")
print(f"Results dir: {RESULTS_DIR}")
```

## Data load and shape audit

Load four files: train and test raw splits (from notebook 05's output)
and train and test embeddings (from notebook 07). Verify row counts
match between the raw splits and the embeddings before any joining.

Expected shapes:

- Raw train: (47,487, 72) including `flight_id` and `inefficient`
- Raw test: (8,724, 72)
- Train embeddings: (47,487, 258) including `flight_id` and `inefficient`
- Test embeddings: (8,724, 258)

```python
train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

train_emb = pd.read_parquet(EMB_DIR / "train_embeddings.parquet")
test_emb = pd.read_parquet(EMB_DIR / "test_embeddings.parquet")

print(f"Raw train shape:       {train_raw.shape}")
print(f"Raw test shape:        {test_raw.shape}")
print(f"Train embeddings shape: {train_emb.shape}")
print(f"Test embeddings shape:  {test_emb.shape}")

print(f"\nRaw train flights:       {train_raw['flight_id'].nunique()}")
print(f"Raw test flights:        {test_raw['flight_id'].nunique()}")
print(f"Train embedding flights: {train_emb['flight_id'].nunique()}")
print(f"Test embedding flights:  {test_emb['flight_id'].nunique()}")

print(f"\nRow count match train: {len(train_raw) == len(train_emb)}")
print(f"Row count match test:  {len(test_raw) == len(test_emb)}")
```

## Drop startup NaN rows and join embeddings

The raw splits contain startup NaN rows from lag and rolling standard
deviation features (roughly 5 rows per flight at the start). Notebook
07 dropped the same rows before building graphs, so the embedding
parquet files match the post-dropna row counts.

Steps:

1. Drop NaN rows on the raw splits (same logic as notebook 06)
2. Audit that row counts and `flight_id` sequences match between raw
   and embeddings
3. Concatenate raw features and embeddings side by side, since both
   are now aligned by position

```python
train_clean = train_raw.dropna().reset_index(drop=True)
test_clean = test_raw.dropna().reset_index(drop=True)

print("After dropna on raw splits:")
print(f"  Train shape: {train_clean.shape}")
print(f"  Test shape:  {test_clean.shape}")

print("\nRow count alignment:")
print(f"  Train raw vs embeddings: {len(train_clean) == len(train_emb)}")
print(f"  Test raw vs embeddings:  {len(test_clean) == len(test_emb)}")

print("\nFlight_id alignment (first 10 rows):")
print(f"  Train raw flight_ids:        {train_clean['flight_id'].head(10).tolist()}")
print(f"  Train embedding flight_ids:  {train_emb['flight_id'].head(10).tolist()}")

print("\nFlight_id sequence match across all rows:")
print(f"  Train: {(train_clean['flight_id'].values == train_emb['flight_id'].values).all()}")
print(f"  Test:  {(test_clean['flight_id'].values == test_emb['flight_id'].values).all()}")

print("\nTarget alignment:")
print(f"  Train: {(train_clean['inefficient'].values == train_emb['inefficient'].values).all()}")
print(f"  Test:  {(test_clean['inefficient'].values == test_emb['inefficient'].values).all()}")
```

**Findings:**

- Raw splits drop to 47,487 train and 8,724 test rows, exactly matching the embedding row counts from notebook 07
- `flight_id` sequences align row-by-row across all rows in both train and test
- Targets align row-by-row across all rows in both train and test
- Concatenating raw features and embeddings by position is safe

```python
emb_cols = [c for c in train_emb.columns if c.startswith("gat_emb_")]

train_full = pd.concat(
    [train_clean.reset_index(drop=True), train_emb[emb_cols].reset_index(drop=True)],
    axis=1,
)
test_full = pd.concat(
    [test_clean.reset_index(drop=True), test_emb[emb_cols].reset_index(drop=True)],
    axis=1,
)

print(f"Train hybrid shape: {train_full.shape}")
print(f"Test hybrid shape:  {test_full.shape}")
print(f"\nExpected columns: 70 features + flight_id + inefficient + 256 embeddings = 328")
print(f"Actual columns:   {train_full.shape[1]}")

print(f"\nEmbedding columns: {len(emb_cols)}")
print(f"First 3 embedding column names: {emb_cols[:3]}")
print(f"Last 3 embedding column names:  {emb_cols[-3:]}")

print(f"\nNaN check after concat:")
print(f"  Train NaN cells: {train_full.isna().sum().sum()}")
print(f"  Test NaN cells:  {test_full.isna().sum().sum()}")
```

**Findings:**

- Hybrid dataframes assembled at the expected 328 columns: 70 raw features, 256 GAT embeddings, plus `flight_id` and `inefficient`
- Embedding columns named consistently from `gat_emb_000` to `gat_emb_255`
- No NaN values introduced by the concatenation


## Feature matrix, target, and group vector

Build model inputs from the hybrid dataframe:

- `X_train`, `X_test`: 326 feature columns (70 raw + 256 GAT embeddings)
- `y_train`, `y_test`: `inefficient` target
- `groups_train`: `flight_id` for GroupKFold cross-validation

The group vector is only needed on the training set since GroupKFold
operates on training data during cross-validation. The test set is held
out wholesale.

```python
feature_cols = [c for c in train_full.columns if c not in ("flight_id", "inefficient")]

X_train = train_full[feature_cols].copy()
y_train = train_full["inefficient"].copy()
groups_train = train_full["flight_id"].copy()

X_test = test_full[feature_cols].copy()
y_test = test_full["inefficient"].copy()

raw_feature_count = len([c for c in feature_cols if not c.startswith("gat_emb_")])
emb_feature_count = len([c for c in feature_cols if c.startswith("gat_emb_")])

print(f"Total feature count: {len(feature_cols)}")
print(f"  Raw engineered features: {raw_feature_count}")
print(f"  GAT embedding features:  {emb_feature_count}")
print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"groups_train unique flights: {groups_train.nunique()}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")
```

**Findings:**

- 326 features confirmed: 70 raw engineered + 256 GAT embeddings
- 47,487 train rows across 94 flights, 8,724 test rows
- Target and group vectors align with feature matrices


## Sanitize feature names for XGBoost

XGBoost rejects column names containing `[`, `]`, or `<` because they
conflict with its internal parsing. The raw sensor columns from the
IDF-DS dataset use bracket notation for vector components (for example,
`gyro_rad[0]_f104` for the x-axis of the gyroscope). The GAT embedding
columns are already safe (`gat_emb_000` through `gat_emb_255`), but the
raw features need cleaning. Apply the same transformation to train and
test to keep names aligned.

```python
def sanitize_columns(df):
    df = df.copy()
    df.columns = [c.replace("[", "_").replace("]", "").replace("<", "_lt_") for c in df.columns]
    return df

X_train = sanitize_columns(X_train)
X_test = sanitize_columns(X_test)
feature_cols = list(X_train.columns)

print(f"Feature count: {len(feature_cols)}")
print("\nFirst 5 raw feature names after sanitization:")
for col in [c for c in feature_cols if not c.startswith("gat_emb_")][:5]:
    print(f"  {col}")
print("\nFirst 3 embedding column names (unchanged):")
for col in [c for c in feature_cols if c.startswith("gat_emb_")][:3]:
    print(f"  {col}")
print("\nAny brackets remaining?")
print(f"  Train: {any('[' in c or ']' in c or '<' in c for c in X_train.columns)}")
print(f"  Test:  {any('[' in c or ']' in c or '<' in c for c in X_test.columns)}")
```

**Findings:**

- 326 feature names cleaned, brackets replaced with underscores in raw columns
- Embedding column names already XGBoost-safe, unchanged
- No problematic characters remaining in either split


## Baseline XGBoost (default hyperparameters)

Train an Extreme Gradient Boosting classifier with default
hyperparameters across the GroupKFold splits. This establishes the
hybrid's reference score before tuning and gives a direct comparison
against notebook 06's untuned XGBoost baseline (mean F1 0.8643 on 70
features).

If the hybrid baseline beats notebook 06's baseline, the embeddings
carry useful signal on top of the raw features. If it matches or
trails, the embeddings either duplicate existing information or add
noise the model has not yet learned to filter.

Cross-validation setup:

- GroupKFold with 6 folds, grouped by `flight_id`
- Each fold holds out roughly 15-16 flights from the 94 training flights
- Metric: F1 score (primary), ROC AUC (secondary)
- Aggregation: mean across folds

XGBoost defaults used:

- `n_estimators=100`, `max_depth=6`, `learning_rate=0.3`
- `tree_method="hist"` for speed
- `random_state=631`
- `eval_metric="logloss"` to silence the default-changed warning

```python
cv = GroupKFold(n_splits=N_SPLITS)

baseline_f1_scores = []
baseline_auc_scores = []
fold_sizes = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train), start=1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    val_flights = groups_train.iloc[val_idx].nunique()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        tree_method="hist",
        random_state=SEED,
        eval_metric="logloss",
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    baseline_f1_scores.append(f1)
    baseline_auc_scores.append(auc)
    fold_sizes.append(val_flights)

    print(f"Fold {fold_idx}: {val_flights} flights | F1 = {f1:.4f} | ROC AUC = {auc:.4f}")

print(f"\nMean F1:      {np.mean(baseline_f1_scores):.4f} (std {np.std(baseline_f1_scores):.4f})")
print(f"Mean ROC AUC: {np.mean(baseline_auc_scores):.4f} (std {np.std(baseline_auc_scores):.4f})")
print(f"\nNotebook 06 baseline mean F1: 0.8643 (on 70 features)")
```

**Findings:**

- Mean F1: 0.8637, Mean ROC AUC: 0.9619
- Untuned hybrid lands within 0.0006 of notebook 06's untuned XGBoost on F1, and within 0.001 on ROC AUC. Effectively a tie at default settings.
- Fold spread similar to notebook 06 (F1 std 0.0154, no outlier folds)
- Early signal: the 256 embedding features do not lift performance at default hyperparameters. Either the embeddings duplicate signal XGBoost already extracts from raw features, or default settings cannot exploit them. Tuning will tell us which.


## Hyperparameter tuning with Optuna

Tune Extreme Gradient Boosting hyperparameters with Optuna using the
Tree-structured Parzen Estimator sampler. 75 trials, mean F1 across
the 6 GroupKFold folds as the objective. Folds are fixed across
trials so trial scores are directly comparable.

Each trial runs early stopping inside every fold with the held-out
fold as the evaluation set. This tunes `n_estimators` implicitly: the
best iteration on the evaluation set is recorded per fold, and the
final model uses the mean of those best iterations.

Search space matches notebook 06 with one change. `colsample_bytree`
lower bound drops from 0.6 to 0.4 to give the model more flexibility
in sampling from the wider 326-feature space per tree.

Search space:

- `max_depth`: 3 to 10
- `learning_rate`: 0.01 to 0.3 (log scale)
- `min_child_weight`: 1 to 10
- `subsample`: 0.6 to 1.0
- `colsample_bytree`: 0.4 to 1.0 (wider than notebook 06)
- `reg_alpha`: 1e-3 to 10 (log scale)
- `reg_lambda`: 1e-3 to 10 (log scale)
- `scale_pos_weight`: 1.0 to 2.0
- `n_estimators`: fixed at 2000 with early stopping (patience 50)

```python
cv = GroupKFold(n_splits=N_SPLITS)
fold_splits = list(cv.split(X_train, y_train, groups_train))


def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 2.0),
        "n_estimators": 2000,
        "tree_method": "hist",
        "random_state": SEED,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
    }

    fold_f1s = []
    fold_best_iters = []

    for train_idx, val_idx in fold_splits:
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        fold_f1s.append(f1_score(y_val, y_pred))
        fold_best_iters.append(model.best_iteration)

    trial.set_user_attr("mean_best_iteration", int(np.mean(fold_best_iters)))
    return float(np.mean(fold_f1s))


sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest trial: {study.best_trial.number}")
print(f"Best mean F1: {study.best_value:.4f}")
print(f"Best mean iterations: {study.best_trial.user_attrs['mean_best_iteration']}")
print(f"\nNotebook 06 best mean F1: 0.8760")
print("\nBest params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
```

**Findings:**

- Best mean cross-validation F1: 0.8740. Notebook 06 hit 0.8760 on the 70-feature input. The hybrid lands 0.002 below.
- Lift over the hybrid's untuned baseline: +0.010, similar to the +0.012 lift Optuna gave notebook 06.
- Best trial used 919 trees at learning rate 0.037, deeper into the search than notebook 06 (which used 436 trees at 0.049). The hybrid needed more, slower trees to reach a similar score, which is consistent with the wider feature space.
- `colsample_bytree` landed at 0.68, comfortably above the new 0.4 floor. The widened range did not change the outcome.
- `reg_alpha` at 3.06 is on the higher side of the search range, suggesting the model is pruning embedding features that do not pull their weight.
- Cross-validation read: the 256 GAT embeddings do not meaningfully add to what XGBoost already extracts from the 70 raw features. Test set will confirm whether this holds.


## Final model fit and test set evaluation

Refit Extreme Gradient Boosting on the full training set (47,487 rows,
all 94 flights) using the best Optuna hyperparameters and the mean best
iteration count from cross-validation as `n_estimators`. Evaluate on
the held-out test set (8,724 rows, 17 flights).

Reporting:

- F1 score and ROC AUC at the default 0.5 threshold
- Comparison against notebook 06 (XGBoost on 70 features) and the
  other baselines

The 0.5 threshold is used to match notebook 06's primary reported
number, keeping the comparison clean.

```python
best_params = study.best_params.copy()
best_n_estimators = study.best_trial.user_attrs["mean_best_iteration"]

final_params = {
    **best_params,
    "n_estimators": best_n_estimators,
    "tree_method": "hist",
    "random_state": SEED,
    "eval_metric": "logloss",
}

final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X_train, y_train)

y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("Hybrid test set performance (threshold = 0.5):")
print(f"  F1:      {test_f1:.4f}")
print(f"  ROC AUC: {test_auc:.4f}")

print("\nComparison to prior models on test set:")
print(f"  {'Model':<40} {'F1':>8} {'ROC AUC':>10}")
print(f"  {'Persistence (naive)':<40} {0.9105:>8.4f} {0.9261:>10.4f}")
print(f"  {'XGBoost (70 features)':<40} {0.8692:>8.4f} {0.9634:>10.4f}")
print(f"  {'Lasso Logistic Regression':<40} {0.8182:>8.4f} {0.9239:>10.4f}")
print(f"  {'Elastic Net Logistic Regression':<40} {0.8155:>8.4f} {0.9229:>10.4f}")
print(f"  {'Linear Discriminant Analysis':<40} {0.7956:>8.4f} {0.9097:>10.4f}")
print(f"  {'Graph Attention Network':<40} {0.7241:>8.4f} {0.8556:>10.4f}")
print(f"  {'GAT + XGBoost hybrid':<40} {test_f1:>8.4f} {test_auc:>10.4f}")

model_path = MODEL_DIR / "hybrid_xgboost.pkl"
joblib.dump(final_model, model_path)
print(f"\nModel saved to: {model_path}")

results = {
    "model": "gat_xgboost_hybrid",
    "random_seed": SEED,
    "n_features": len(feature_cols),
    "n_raw_features": raw_feature_count,
    "n_embedding_features": emb_feature_count,
    "cv": {
        "n_splits": N_SPLITS,
        "strategy": "GroupKFold by flight_id",
        "baseline_mean_f1": float(np.mean(baseline_f1_scores)),
        "baseline_mean_roc_auc": float(np.mean(baseline_auc_scores)),
        "tuned_mean_f1": float(study.best_value),
    },
    "optuna": {
        "n_trials": N_TRIALS,
        "best_trial": int(study.best_trial.number),
        "best_mean_cv_f1": float(study.best_value),
        "best_mean_iterations": int(best_n_estimators),
        "best_params": best_params,
    },
    "test": {
        "n_rows": int(len(y_test)),
        "n_flights": int(test_full["flight_id"].nunique()),
        "f1": float(test_f1),
        "roc_auc": float(test_auc),
    },
}

results_path = RESULTS_DIR / "hybrid_xgboost.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {results_path}")
```

**Findings:**

- Hybrid test F1: 0.8666, ROC AUC: 0.9642
- Hybrid trails XGBoost on F1 by 0.0026 (0.8666 vs 0.8692) and beats XGBoost on ROC AUC by 0.0008 (0.9642 vs 0.9634). Both differences are within cross-validation noise.
- The 256 GAT embeddings did not improve the model. The hybrid is statistically indistinguishable from XGBoost on raw features alone.
- Persistence still leads on F1 by a wide margin (0.9105) thanks to the 0.838 lag-1 autocorrelation in the target.
- Hybrid does beat the GAT alone by 0.14 F1, confirming that XGBoost is doing the heavy lifting in this hybrid, not the embeddings.
- Reasonable hypothesis: the 70 engineered features (especially throttle, which dominated notebook 06's permutation importance) already encode most of the signal. The GAT embeddings re-express that signal in a different basis without adding new information.
- Feature importance in the next section will tell us how much the model actually used the embeddings versus the raw features.


## Threshold tuning on out-of-fold predictions

XGBoost's default 0.5 threshold may not be optimal for F1. Sweep
thresholds from 0.1 to 0.9 against out-of-fold predictions from the
tuned model, pick the threshold that maximizes F1, and apply it to
the test set.

Out-of-fold predictions are generated by refitting the tuned model
on each fold's training rows and predicting the held-out rows. This
keeps threshold selection honest: the test set is never used to
choose the threshold.

Notebook 06 ran the same procedure and found the tuned threshold
(0.46) underperformed 0.5 on the test set by 0.0026, confirming
XGBoost probabilities were already well-calibrated. The hybrid is
expected to behave similarly.

```python
oof_proba = np.zeros(len(y_train))

for train_idx, val_idx in fold_splits:
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]

    fold_model = xgb.XGBClassifier(**final_params)
    fold_model.fit(X_tr, y_tr)
    oof_proba[val_idx] = fold_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.1, 0.91, 0.01)
threshold_f1s = np.array([f1_score(y_train, (oof_proba >= t).astype(int)) for t in thresholds])

best_idx = int(np.argmax(threshold_f1s))
best_threshold = float(round(thresholds[best_idx], 2))
best_oof_f1 = float(threshold_f1s[best_idx])
oof_f1_at_default = float(threshold_f1s[np.argmin(np.abs(thresholds - 0.5))])

print(f"Out-of-fold F1 at threshold 0.5:           {oof_f1_at_default:.4f}")
print(f"Out-of-fold F1 at best threshold ({best_threshold}): {best_oof_f1:.4f}")
print(f"OOF F1 lift from threshold tuning:         {best_oof_f1 - oof_f1_at_default:+.4f}")

y_test_pred_tuned = (y_test_proba >= best_threshold).astype(int)
test_f1_tuned = f1_score(y_test, y_test_pred_tuned)

print(f"\nTest set F1 at threshold 0.5:           {test_f1:.4f}")
print(f"Test set F1 at tuned threshold ({best_threshold}): {test_f1_tuned:.4f}")
print(f"Test F1 change from threshold tuning:   {test_f1_tuned - test_f1:+.4f}")

print(f"\nROC AUC is unchanged by threshold tuning: {test_auc:.4f}")
```

**Findings:**

- Out-of-fold F1 is flat across thresholds 0.5 and 0.55 (both 0.8745). The maximum is essentially tied across a wide range.
- Test F1 nudges up by 0.0018 at threshold 0.55 (0.8684 vs 0.8666). Within fold noise, not a real lift.
- ROC AUC unchanged at 0.9642, as expected (threshold has no effect on ranking metrics).
- Same story as notebook 06: hybrid probabilities are well-calibrated, threshold tuning is not a meaningful lever.
- Primary reported number remains F1 0.8666 at threshold 0.5 for clean comparison against notebook 06's primary number (0.8692 at threshold 0.5). The tuned number (0.8684) is reported for transparency.


## Feature importance: built-in gain and permutation

Two complementary views of feature importance, with a third question
unique to this notebook: how does the model split credit between the
70 raw features and the 256 GAT embeddings?

- **Built-in gain**: average improvement in the loss function when a
  feature is used to split a node, summed across all trees. Fast and
  free from the trained model. Biased toward features used early in
  tree construction.
- **Permutation importance**: drop in test set F1 when a feature's
  values are randomly shuffled. Measures what the model actually
  relies on at prediction time on held-out data. Slow but honest.

The key question for the hybrid: do GAT embeddings appear in the top
features, or did XGBoost effectively ignore them and rely on the raw
sensors? If embeddings are absent from the permutation top 15, the
GAT did not contribute to the final hybrid prediction.

Permutation importance uses 10 repeats on the test set, scored on F1
at threshold 0.5.

```python
gain_importance = pd.DataFrame({
    "feature": final_model.feature_names_in_,
    "gain": final_model.feature_importances_,
}).sort_values("gain", ascending=False).reset_index(drop=True)
gain_importance["is_embedding"] = gain_importance["feature"].str.startswith("gat_emb_")


def f1_at_default_threshold(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return f1_score(y, preds)


perm_result = permutation_importance(
    final_model,
    X_test,
    y_test,
    scoring=f1_at_default_threshold,
    n_repeats=10,
    random_state=SEED,
    n_jobs=-1,
)

perm_importance = pd.DataFrame({
    "feature": X_test.columns,
    "perm_mean": perm_result.importances_mean,
    "perm_std": perm_result.importances_std,
}).sort_values("perm_mean", ascending=False).reset_index(drop=True)
perm_importance["is_embedding"] = perm_importance["feature"].str.startswith("gat_emb_")

print("Top 15 features by built-in gain:")
print(gain_importance.head(15).to_string(index=False))

print("\nTop 15 features by permutation importance:")
print(perm_importance.head(15).to_string(index=False))

gain_emb_in_top15 = gain_importance.head(15)["is_embedding"].sum()
perm_emb_in_top15 = perm_importance.head(15)["is_embedding"].sum()
print(f"\nGAT embeddings in gain top 15:        {gain_emb_in_top15} of 15")
print(f"GAT embeddings in permutation top 15: {perm_emb_in_top15} of 15")

total_gain_emb = gain_importance[gain_importance["is_embedding"]]["gain"].sum()
total_gain_raw = gain_importance[~gain_importance["is_embedding"]]["gain"].sum()
print(f"\nTotal gain attributed to embeddings: {total_gain_emb:.4f} ({total_gain_emb / (total_gain_emb + total_gain_raw) * 100:.1f}%)")
print(f"Total gain attributed to raw features: {total_gain_raw:.4f} ({total_gain_raw / (total_gain_emb + total_gain_raw) * 100:.1f}%)")

total_perm_emb = perm_importance[perm_importance["is_embedding"]]["perm_mean"].clip(lower=0).sum()
total_perm_raw = perm_importance[~perm_importance["is_embedding"]]["perm_mean"].clip(lower=0).sum()
total_perm = total_perm_emb + total_perm_raw
if total_perm > 0:
    print(f"\nTotal positive permutation importance from embeddings: {total_perm_emb:.4f} ({total_perm_emb / total_perm * 100:.1f}%)")
    print(f"Total positive permutation importance from raw features: {total_perm_raw:.4f} ({total_perm_raw / total_perm * 100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

top_n = 15
gain_plot = gain_importance.head(top_n).iloc[::-1]
gain_colors = ["C2" if e else "C0" for e in gain_plot["is_embedding"]]
axes[0].barh(gain_plot["feature"], gain_plot["gain"], color=gain_colors)
axes[0].set_xlabel("Gain")
axes[0].set_title("Top 15 features by built-in gain")
axes[0].grid(axis="x", alpha=0.3)

perm_plot = perm_importance.head(top_n).iloc[::-1]
perm_colors = ["C2" if e else "C1" for e in perm_plot["is_embedding"]]
axes[1].barh(perm_plot["feature"], perm_plot["perm_mean"], xerr=perm_plot["perm_std"], color=perm_colors)
axes[1].set_xlabel("Mean F1 drop when permuted")
axes[1].set_title("Top 15 features by permutation importance")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()
```

**Findings:**

This is the most informative split in the project. Gain and permutation tell very different stories about the embeddings, and the disagreement is the whole point.

**Gain says embeddings dominate (66.6%).** Ten of the top 15 features by gain are GAT embeddings. The model used them heavily during training as split features.

**Permutation says embeddings barely matter (12.2%).** Only 5 embeddings appear in the permutation top 15, and the largest embedding contribution (`gat_emb_171`) is 0.0011, about 287 times smaller than throttle (`output_0_f3` at 0.3042). Scrambling embeddings at prediction time does almost nothing to test F1.

**What this means.** The embeddings carry redundant information, not new information. XGBoost used them during training because they offered easy splits (smooth, dense, 256 of them to choose from), but when an embedding gets scrambled at prediction time, the model falls back on the raw features and recovers most of its accuracy. The embeddings re-encode signal that throttle, wind, pressure, temperature, and time-since-start already contain.

**Throttle still dominates everything.** `output_0_f3` permutation importance is 0.30. The next non-throttle feature is `time_since_start_of_flight` at 0.029, an order of magnitude smaller. Same story as notebook 06.

**Environmental features hold their place.** `wind_magnitude`, `baro_pressure_pa_f117`, and `ambient_temperature_f117` appear in the permutation top 5, confirming notebook 04's environmental engineering work was not wasted.

**Lag features fade.** Only one lag feature (`output_2_f3_lag2`) appears in the permutation top 15, with importance of 0.0011. Consistent with notebook 06's finding that lags are redundant with their raw counterparts at prediction time.

**Why the hybrid did not beat XGBoost.** The GAT learned to encode telemetry structure in a 256D space, but the structure it captured overlaps almost entirely with what XGBoost already extracts from the raw features. Adding the embeddings gave the model more places to split, not more information to split on.


## Summary

GAT + XGBoost hybrid on the 326-feature input (70 raw engineered
features + 256 GAT embeddings), tuned with Optuna over 75 trials,
evaluated on 17 held-out flights.

**Final test scores:**

- F1: 0.8666 (at threshold 0.5)
- ROC AUC: 0.9642

**Full four-tier comparison:**

| Model | F1 | ROC AUC |
| --- | --- | --- |
| Persistence (naive) | 0.9105 | 0.9261 |
| Lasso Logistic Regression | 0.8182 | 0.9239 |
| Elastic Net Logistic Regression | 0.8155 | 0.9229 |
| Linear Discriminant Analysis | 0.7956 | 0.9097 |
| XGBoost (70 features) | 0.8692 | 0.9634 |
| Graph Attention Network | 0.7241 | 0.8556 |
| GAT + XGBoost hybrid | 0.8666 | 0.9642 |

**Takeaways:**

- The hybrid lands within 0.003 F1 of XGBoost on raw features, and within 0.001 ROC AUC. Effectively a tie on test set performance.
- Feature importance reveals why. The model used embeddings heavily during training (66.6% of total gain) but barely relied on them at prediction time (12.2% of permutation importance). The embeddings re-encoded signal the raw features already carried.
- Throttle (`output_0_f3`) dominates both XGBoost and the hybrid. Environmental features (wind, pressure, temperature) and time-since-start contribute meaningfully. Lag features and GAT embeddings add little at prediction time.
- Persistence still leads on F1 (0.9105). The 0.838 lag-1 autocorrelation in the target makes "predict the previous row" hard to beat at a fixed threshold. Persistence wins F1 but loses ROC AUC by a wide margin because it produces no useful probability ranking.
- Caveat acknowledged from the notebook intro: GAT embeddings were extracted from a single fold's model, so training-side embeddings have mild label leakage. The leakage is small in practice (GAT train F1 and test F1 matched within 0.008) and the test F1 reported above is unaffected because the 17 test flights were never seen by the GAT.

**What this tells us about the four-tier comparison:**

- Linear models (Lasso, Elastic Net, LDA) underperform because the relationship between sensors and inefficiency is non-linear
- XGBoost captures the non-linearity directly from raw features and is the strongest single model on ROC AUC
- The GAT alone underperforms because graph attention over 16 sensor nodes is not the right inductive bias for this task. The sensors do not form a natural graph that benefits from message passing
- The hybrid neither helps nor hurts XGBoost meaningfully. The embeddings are redundant
- For this dataset, the best modeling story is gradient boosting on well-engineered features

**Artifacts saved:**

- `models/08_hybrid/hybrid_xgboost.pkl`
- `results/08_hybrid/hybrid_xgboost.json`

```python
results["threshold_tuning"] = {
    "method": "out-of-fold predictions, F1 maximization",
    "best_threshold": best_threshold,
    "oof_f1_at_0.5": oof_f1_at_default,
    "oof_f1_at_best_threshold": best_oof_f1,
    "test_f1_at_tuned_threshold": float(test_f1_tuned),
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results updated at: {results_path}")
print(f"\nFinal results summary:")
print(f"  Test F1 (threshold 0.5):  {test_f1:.4f}")
print(f"  Test F1 (threshold {best_threshold}): {test_f1_tuned:.4f}")
print(f"  Test ROC AUC:             {test_auc:.4f}")
```
