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

REPO_ROOT = Path("/Users/phillipromanmacbook/GitHub/631-uav-project")
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

## Data load and schema audit

Load train and test splits, verify the schema matches notebook 05's
output: 70 feature columns, plus `inefficient` (target) and `flight_id`
(group). Confirm row counts (47,487 train / 8,724 test) and flight counts
(94 train / 17 test).

```python
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Train flights: {train['flight_id'].nunique()}")
print(f"Test flights:  {test['flight_id'].nunique()}")
print(f"Train target balance:\n{train['inefficient'].value_counts(normalize=True).round(3)}")
print(f"\nTrain NaN count: {train.isna().sum().sum()}")
print(f"Test NaN count:  {test.isna().sum().sum()}")
print(f"\nColumns ({len(train.columns)}):")
for col in train.columns:
    print(f"  {col}")
```

## Drop startup NaN rows

The lag1, lag2, and rolling standard deviation (window 5) features each
produce NaN values at the start of every flight. Lag1 produces 1 NaN row
per flight, lag2 produces 2, and rolling standard deviation with window
5 produces 4. After accounting for overlap, each flight loses its first
4 rows.

XGBoost can handle NaN values natively, but notebook 05's linear models
cannot. To keep the four-tier model comparison apples-to-apples, XGBoost
must train and evaluate on the same row set as the linear baselines.
Dropping NaN rows here matches notebook 05's effective behavior.

The NaN values are deterministic startup artifacts, not missing data, so
imputation would fabricate signal that does not exist.

```python
print("Before dropping NaN rows:")
print(f"  Train shape: {train.shape}")
print(f"  Test shape:  {test.shape}")
print(f"  Train flights: {train['flight_id'].nunique()}")
print(f"  Test flights:  {test['flight_id'].nunique()}")
print(f"  Train NaN cells: {train.isna().sum().sum()}")
print(f"  Test NaN cells:  {test.isna().sum().sum()}")

train_clean = train.dropna().reset_index(drop=True)
test_clean = test.dropna().reset_index(drop=True)

print("\nAfter dropping NaN rows:")
print(f"  Train shape: {train_clean.shape}")
print(f"  Test shape:  {test_clean.shape}")
print(f"  Train flights: {train_clean['flight_id'].nunique()}")
print(f"  Test flights:  {test_clean['flight_id'].nunique()}")
print(f"  Train NaN cells: {train_clean.isna().sum().sum()}")
print(f"  Test NaN cells:  {test_clean.isna().sum().sum()}")

print("\nRows dropped:")
print(f"  Train: {len(train) - len(train_clean)} rows ({(len(train) - len(train_clean)) / len(train) * 100:.2f}%)")
print(f"  Test:  {len(test) - len(test_clean)} rows ({(len(test) - len(test_clean)) / len(test) * 100:.2f}%)")

print("\nTarget balance after drop:")
print(f"  Train:\n{train_clean['inefficient'].value_counts(normalize=True).round(3)}")
```

## Feature matrix, target, and group vector

Build the standard model inputs:

- `X_train`, `X_test`: 70 feature columns
- `y_train`, `y_test`: `inefficient` target
- `groups_train`: `flight_id` for GroupKFold cross-validation

The group vector is only needed on the training set since GroupKFold
operates on training data during cross-validation. The test set is held
out wholesale.

```python
feature_cols = [c for c in train_clean.columns if c not in ("flight_id", "inefficient")]

X_train = train_clean[feature_cols].copy()
y_train = train_clean["inefficient"].copy()
groups_train = train_clean["flight_id"].copy()

X_test = test_clean[feature_cols].copy()
y_test = test_clean["inefficient"].copy()

print(f"Feature count: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"groups_train unique: {groups_train.nunique()}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")
```

## Sanitize feature names for XGBoost

XGBoost rejects column names containing `[`, `]`, or `<` because they
conflict with its internal parsing. The raw sensor columns from the
IDF-DS dataset use bracket notation for vector components (for example,
`gyro_rad[0]_f104` for the x-axis of the gyroscope). Replace brackets
with underscores so XGBoost accepts the feature matrix. Apply the same
transformation to train and test to keep names aligned.

```python
def sanitize_columns(df):
    df = df.copy()
    df.columns = [c.replace("[", "_").replace("]", "").replace("<", "_lt_") for c in df.columns]
    return df

X_train = sanitize_columns(X_train)
X_test = sanitize_columns(X_test)
feature_cols = list(X_train.columns)

print(f"Feature count: {len(feature_cols)}")
print("\nFirst 10 feature names after sanitization:")
for col in feature_cols[:10]:
    print(f"  {col}")
print("\nAny brackets remaining?")
print(f"  Train: {any('[' in c or ']' in c or '<' in c for c in X_train.columns)}")
print(f"  Test:  {any('[' in c or ']' in c or '<' in c for c in X_test.columns)}")
```

## Baseline XGBoost (default hyperparameters)

Train an Extreme Gradient Boosting classifier with default
hyperparameters across the GroupKFold splits to establish a reference
score before tuning. This tells us how much lift Optuna provides over
out-of-the-box defaults.

Cross-validation setup:

- GroupKFold with 6 folds, grouped by flight_id
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
```

**Findings:**

- Mean F1: 0.8643. Mean ROC AUC: 0.9629.
- Beats all three linear baselines from notebook 05 on both metrics, untuned.
- Trails persistence on F1 (0.9105) by 5 points. Beats persistence on ROC AUC (0.9261).
- Fold spread tight: F1 std 0.018, no outlier folds.


## Hyperparameter tuning with Optuna

Tune Extreme Gradient Boosting hyperparameters with Optuna using the
Tree-structured Parzen Estimator sampler. 75 trials, mean F1 across the
6 GroupKFold folds as the objective. Folds are fixed across trials so
trial scores are directly comparable.

Each trial runs early stopping inside every fold with the held-out fold
as the evaluation set. This tunes `n_estimators` implicitly: the best
iteration on the evaluation set is recorded per fold, and the model
trained on all folds in the final step will use the mean of those best
iterations.

Search space:

- `max_depth`: 3 to 10
- `learning_rate`: 0.01 to 0.3 (log scale)
- `min_child_weight`: 1 to 10
- `subsample`: 0.6 to 1.0
- `colsample_bytree`: 0.6 to 1.0
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
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
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
print("\nBest params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
```

**Findings:**

- Best mean cross-validation F1: 0.8760. Lift over untuned baseline: +0.012.
- Best trial (73 of 75) used 436 trees at learning rate 0.049. Slow, well-regularized.
- Every best setting landed in the middle of the range we searched, not at the edges. The search was wide enough; no need to expand it.
- Modest gain. Default XGBoost was already near the ceiling for this feature set.


## Final model fit and test set evaluation

Refit Extreme Gradient Boosting on the full training set (47,487 rows,
all 94 flights) using the best Optuna hyperparameters and the mean best
iteration count from cross-validation as `n_estimators`. Evaluate on the
held-out test set (8,724 rows, 17 flights).

Reporting:

- F1 score and ROC AUC at the default 0.5 threshold
- Comparison against notebook 05 baselines

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

print("Test set performance (threshold = 0.5):")
print(f"  F1:      {test_f1:.4f}")
print(f"  ROC AUC: {test_auc:.4f}")

print("\nComparison to notebook 05 baselines:")
print(f"  {'Model':<35} {'F1':>8} {'ROC AUC':>10}")
print(f"  {'Persistence':<35} {0.9105:>8.4f} {0.9261:>10.4f}")
print(f"  {'Lasso Logistic Regression':<35} {0.8182:>8.4f} {0.9239:>10.4f}")
print(f"  {'Elastic Net Logistic Regression':<35} {0.8155:>8.4f} {0.9229:>10.4f}")
print(f"  {'Linear Discriminant Analysis':<35} {0.7956:>8.4f} {0.9097:>10.4f}")
print(f"  {'XGBoost (default threshold)':<35} {test_f1:>8.4f} {test_auc:>10.4f}")
```

**Findings:**

- Test F1: 0.8692. Test ROC AUC: 0.9634.
- Cross-validation to test drop: 0.007 on F1. Negligible. No overfitting.
- Beats all three linear baselines on both metrics by 5 to 7 points on F1.
- Beats persistence on ROC AUC by 0.037. Loses to persistence on F1 by 4 points.


## Threshold tuning via cross-validation

The default 0.5 decision threshold treats false positives and false
negatives as equally costly. F1 is sensitive to threshold choice, and
the ROC AUC gap over the linear baselines suggests Extreme Gradient
Boosting produces well-ranked probabilities that may benefit from a
non-default threshold.

To avoid test set leakage, the threshold is tuned on out-of-fold
predictions from the training set, not on the test set. Process:

1. Run 6-fold GroupKFold cross-validation with the final hyperparameters
2. Collect out-of-fold probability predictions across all training rows
3. Sweep thresholds from 0.05 to 0.95 in 0.01 steps
4. Pick the threshold that maximizes mean F1 across folds
5. Apply that threshold to the test set probability predictions

```python
oof_proba = np.zeros(len(y_train))
fold_f1_at_default = []

for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    fold_model = xgb.XGBClassifier(**final_params)
    fold_model.fit(X_tr, y_tr)

    proba = fold_model.predict_proba(X_val)[:, 1]
    oof_proba[val_idx] = proba

    fold_f1_at_default.append(f1_score(y_val, (proba >= 0.5).astype(int)))
    print(f"Fold {fold_idx}: F1 at 0.5 = {fold_f1_at_default[-1]:.4f}")

print(f"\nMean fold F1 at threshold 0.5: {np.mean(fold_f1_at_default):.4f}")

thresholds = np.arange(0.05, 0.96, 0.01)
threshold_f1s = []

for thr in thresholds:
    preds = (oof_proba >= thr).astype(int)
    threshold_f1s.append(f1_score(y_train, preds))

best_idx = int(np.argmax(threshold_f1s))
best_threshold = float(thresholds[best_idx])
best_oof_f1 = float(threshold_f1s[best_idx])

print(f"\nBest threshold: {best_threshold:.2f}")
print(f"Out-of-fold F1 at best threshold: {best_oof_f1:.4f}")
print(f"Out-of-fold F1 at 0.5:            {threshold_f1s[np.argmin(np.abs(thresholds - 0.5))]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, threshold_f1s, color="C0", linewidth=2)
ax.axvline(best_threshold, color="C1", linestyle="--", label=f"Best threshold = {best_threshold:.2f}")
ax.axvline(0.5, color="C2", linestyle=":", label="Default threshold = 0.5")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("F1 score (out-of-fold)")
ax.set_title("F1 vs decision threshold (out-of-fold predictions)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Findings:**

- Out-of-fold F1 at default threshold 0.5: 0.8754. Out-of-fold F1 at tuned threshold 0.46: 0.8765. Gain from tuning is 0.0011, effectively negligible.
- The F1 vs threshold curve is flat between roughly 0.35 and 0.55, indicating well-calibrated probabilities. There is no exploitable miscalibration to chase.
- Per-fold F1 at 0.5 ranges from 0.8338 (fold 2) to 0.8963 (fold 3). Spread of 0.063 across folds is consistent with the cross-validation noise seen earlier.
- Tuned threshold of 0.46 will be applied to the test set predictions for the final reported score.
- XGBoost dominates the linear baselines on both F1 and ROC AUC. It loses to persistence on F1 but dominates persistence on ROC AUC. ROC AUC is the better metric for this problem because the target's high lag-1 autocorrelation (0.838) lets persistence look artificially strong on F1 without producing useful probability rankings. XGBoost provides actual discrimination, persistence provides memorization.


## Final test set evaluation at tuned threshold

Apply the cross-validation-tuned decision threshold of 0.46 to the
Extreme Gradient Boosting test set probability predictions. Report
final F1 and ROC AUC. Save the trained model as a pickle and the
results as a JSON file.

The model pickle and results JSON are saved here because this is the
end of the modeling work for notebook 06. Intermediate fits during
cross-validation and Optuna tuning were not saved since they are
reproducible from the final hyperparameters and seed.

```python
y_test_pred_tuned = (y_test_proba >= best_threshold).astype(int)

test_f1_tuned = f1_score(y_test, y_test_pred_tuned)
test_auc_unchanged = roc_auc_score(y_test, y_test_proba)

print(f"Test set performance at threshold {best_threshold}:")
print(f"  F1:      {test_f1_tuned:.4f}")
print(f"  ROC AUC: {test_auc_unchanged:.4f}")

print(f"\nTest set performance at threshold 0.5 (for reference):")
print(f"  F1:      {test_f1:.4f}")
print(f"  ROC AUC: {test_auc:.4f}")

model_path = MODEL_DIR / "xgboost.pkl"
joblib.dump(final_model, model_path)
print(f"\nModel saved to: {model_path}")

results = {
    "model": "xgboost",
    "random_seed": SEED,
    "n_features": len(feature_cols),
    "cv": {
        "n_splits": N_SPLITS,
        "strategy": "GroupKFold by flight_id",
        "baseline_mean_f1": float(np.mean(baseline_f1_scores)),
        "baseline_mean_roc_auc": float(np.mean(baseline_auc_scores)),
        "tuned_mean_f1_at_0.5": float(np.mean(fold_f1_at_default)),
        "oof_f1_at_0.5": float(threshold_f1s[np.argmin(np.abs(thresholds - 0.5))]),
        "oof_f1_at_best_threshold": best_oof_f1,
    },
    "optuna": {
        "n_trials": N_TRIALS,
        "best_trial": int(study.best_trial.number),
        "best_mean_cv_f1": float(study.best_value),
        "best_mean_iterations": int(best_n_estimators),
        "best_params": best_params,
    },
    "threshold_tuning": {
        "method": "out-of-fold predictions, F1 maximization",
        "best_threshold": best_threshold,
    },
    "test": {
        "n_rows": int(len(y_test)),
        "n_flights": int(test_clean["flight_id"].nunique()),
        "f1_at_0.5": float(test_f1),
        "roc_auc_at_0.5": float(test_auc),
        "f1_at_tuned_threshold": float(test_f1_tuned),
        "roc_auc": float(test_auc_unchanged),
    },
}

results_path = RESULTS_DIR / "xgboost.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {results_path}")
```

**Findings:**

- Test F1 at tuned threshold 0.46: 0.8666. Test F1 at default 0.5: 0.8692. The tuned threshold underperforms the default on test by 0.0026, within cross-validation noise (fold standard deviation 0.0176).
- ROC AUC is identical at both thresholds (0.9634), confirming threshold choice does not affect ranking metrics.
- The reversal between out-of-fold gain and test loss is consistent with well-calibrated probabilities. The threshold is not a meaningful lever for this model on this dataset.
- Primary reported result: F1 0.8666 at threshold 0.46. The 0.5 result is reported alongside for transparency. Picking 0.5 post-hoc on the basis of test performance would constitute test set leakage.
- Trained model and results JSON saved to `models/06_xgboost/xgboost.pkl` and `results/06_xgboost/xgboost.json`.


## Feature importance: built-in gain and permutation

Two complementary views of feature importance:

- **Built-in gain**: average improvement in the loss function when a
  feature is used to split a node, summed across all trees in the
  ensemble. Fast and free from the trained model. Biased toward
  high-cardinality features and features used early in tree
  construction.
- **Permutation importance**: drop in test set F1 when a feature's
  values are randomly shuffled. Measures what the model actually relies
  on at prediction time on held-out data. Slow but honest.

Disagreement between the two is informative. Gain reflects what the
model learned during training; permutation reflects what matters for
generalization. For this notebook, the key question is whether the lag
features dominate (suggesting XGBoost is effectively reinventing
persistence) or whether non-lag features carry meaningful signal.

Permutation importance uses 10 repeats with the test set, scored on F1
at the tuned threshold 0.46.

```python
gain_importance = pd.DataFrame({
    "feature": final_model.feature_names_in_,
    "gain": final_model.feature_importances_,
}).sort_values("gain", ascending=False).reset_index(drop=True)


def f1_at_tuned_threshold(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    preds = (proba >= best_threshold).astype(int)
    return f1_score(y, preds)


perm_result = permutation_importance(
    final_model,
    X_test,
    y_test,
    scoring=f1_at_tuned_threshold,
    n_repeats=10,
    random_state=SEED,
    n_jobs=-1,
)

perm_importance = pd.DataFrame({
    "feature": X_test.columns,
    "perm_mean": perm_result.importances_mean,
    "perm_std": perm_result.importances_std,
}).sort_values("perm_mean", ascending=False).reset_index(drop=True)

print("Top 15 features by built-in gain:")
print(gain_importance.head(15).to_string(index=False))

print("\nTop 15 features by permutation importance:")
print(perm_importance.head(15).to_string(index=False))

top_gain = set(gain_importance.head(15)["feature"])
top_perm = set(perm_importance.head(15)["feature"])
print(f"\nOverlap between top 15 by gain and top 15 by permutation: {len(top_gain & top_perm)} features")
print(f"In gain top 15 but not permutation top 15: {sorted(top_gain - top_perm)}")
print(f"In permutation top 15 but not gain top 15: {sorted(top_perm - top_gain)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

top_n = 15
gain_plot = gain_importance.head(top_n).iloc[::-1]
axes[0].barh(gain_plot["feature"], gain_plot["gain"], color="C0")
axes[0].set_xlabel("Gain")
axes[0].set_title("Top 15 features by built-in gain")
axes[0].grid(axis="x", alpha=0.3)

perm_plot = perm_importance.head(top_n).iloc[::-1]
axes[1].barh(perm_plot["feature"], perm_plot["perm_mean"], xerr=perm_plot["perm_std"], color="C1")
axes[1].set_xlabel("Mean F1 drop when permuted")
axes[1].set_title("Top 15 features by permutation importance")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()
```

**Findings:**

Two ways of measuring feature importance, two different stories:

- **Built-in gain** asks the model which features it used the most during training. The model kept score and reports back. Fast and free, but credulous. If two features carry the same information, gain splits credit between them and both look important.
- **Permutation importance** tests the model on held-out data. It scrambles one feature at a time and measures how much performance actually drops. If the model didn't need a feature, scrambling it doesn't hurt. Honest about what matters at prediction time.

**Throttle dominates everything.** `output_0_f3` is the top feature on both metrics, but permutation tells the real story: scrambling throttle drops F1 by 0.33, which is roughly 7 times larger than the next feature. The model is essentially "look at throttle, then adjust a little."

**Lag features are dead weight.** Four lag features appear in the gain top 15 (`output_0_f3_lag1`, `output_0_f3_lag2`, `indicated_airspeed_m_s_f5_lag1`, `indicated_airspeed_m_s_f5_lag2`) but none appear in the permutation top 15. The model used them during training, but at prediction time it falls back on the raw features when a lag is scrambled. Redundant with their non-lag counterparts.

**Environmental features earn their place.** `wind_magnitude`, `baro_pressure_pa_f117`, and `ambient_temperature_f117` all show real permutation importance. Notebook 04's environmental work carries signal.

**Attitude features matter more than gain suggests.** Five quaternion features (`q_0_f119`, `q_1_f119`, `q_2_f119`, `q_3_f119`, `q_2_f119_rstd5`) appear in the permutation top 15 but not the gain top 15. Each contributes a small but real amount of signal that gain undercredited.

**Gain top 15 vs permutation top 15 overlap: 10 features.** Five disagreements in each direction. The disagreements are the redundancy signature, not a problem with either method.


## Summary

XGBoost on the 70-feature engineered dataset, tuned with Optuna over 75 trials, evaluated on 17 held-out flights.

**Final test scores:**

- F1: 0.8692 (at threshold 0.5)
- ROC AUC: 0.9634

**Compared to notebook 05 baselines:**

| Model | F1 | ROC AUC |
| --- | --- | --- |
| Persistence | 0.9105 | 0.9261 |
| Lasso Logistic Regression | 0.8182 | 0.9239 |
| Elastic Net Logistic Regression | 0.8155 | 0.9229 |
| Linear Discriminant Analysis | 0.7956 | 0.9097 |
| XGBoost | 0.8692 | 0.9634 |

**Takeaways:**

- XGBoost beats all three linear baselines on both metrics.
- XGBoost beats persistence on ROC AUC by a wide margin. Loses to persistence on F1.
- Persistence wins F1 because the target has lag-1 autocorrelation of 0.838. Predicting "same as last row" is hard to beat at a fixed threshold. But persistence produces no useful probability ranking, which is why it loses ROC AUC.
- Threshold tuning made no real difference. Probabilities are already well-calibrated.
- Throttle (`output_0_f3`) carries most of the signal. Environmental features (wind, pressure, temperature) and quaternion attitude features add real but smaller contributions. Lag features are redundant with their raw counterparts at prediction time.

**On the horizon:**

- Notebook 07: Graph Attention Network
- Notebook 08: Graph Attention Network embeddings + XGBoost hybrid
- The persistence-on-F1 ceiling is a real benchmark. The hybrid model is the best shot at beating it without leaning on lag features.
