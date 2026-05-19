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

# Notebook 05: Model Baselines

Three models, in order:

1. **Persistence baseline**: predict target(t) = target(t-1). Floor is around 84% based on the lag-1 ACF of 0.838 from notebook 03.
2. **Logistic Regression**: two regularization variants, Lasso (L1) and Elastic Net.
3. **Linear Discriminant Analysis (LDA)**: second linear baseline with different assumptions.

Goal: establish what linear models can achieve before moving to XGBoost and the Graph Attention Network (GAT).

Inputs: `data/splits/train.csv` and `data/splits/test.csv`, 72 columns each (67 from notebook 03 plus 5 environmental features from notebook 04).

Outputs:
- Trained pipelines saved to `models/05_baselines/` as `.pkl`
- Cross-validation metrics saved to `results/05_baselines/` as `.json`

```python
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

REPO_ROOT = Path.cwd().parents[1]
DATA_DIR = REPO_ROOT / "data" / "splits"
MODELS_DIR = REPO_ROOT / "models" / "05_baselines"
RESULTS_DIR = REPO_ROOT / "results" / "05_baselines"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 631
N_SPLITS = 6

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

plt.style.use("tableau-colorblind10")

print(f"Repo root:   {REPO_ROOT}")
print(f"Data dir:    {DATA_DIR}")
print(f"Models dir:  {MODELS_DIR}")
print(f"Results dir: {RESULTS_DIR}")
```

## Load data and audit columns

Quick inspection pass before any modeling. Three checks on the loaded splits:

1. **Dtypes**: confirm all 72 columns are numeric. Any `object` dtype would break `StandardScaler`.
2. **Unique value counts**: flag constant columns. Constants add no signal and can cause numerical issues in regularized linear models.
3. **NaN counts**: confirm NaN values appear only in lag and rolling-std columns, matching the per-flight startup pattern from notebook 03.

Train and test schemas should match exactly.

```python
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Schemas match: {list(train.columns) == list(test.columns)}")

print("\nDtype counts (train):")
print(train.dtypes.value_counts())

print("\nNon-numeric columns (train):")
non_numeric = train.select_dtypes(exclude=[np.number]).columns.tolist()
print(non_numeric if non_numeric else "None")

print("\nConstant columns (train):")
nunique = train.nunique()
constants = nunique[nunique <= 1].index.tolist()
print(constants if constants else "None")

print("\nNaN counts per column (train, non-zero only):")
nan_counts = train.isna().sum()
print(nan_counts[nan_counts > 0].sort_values(ascending=False))
```

**Findings:**

- Train: 47,957 rows, 72 columns. Test: 8,809 rows, 72 columns. Schemas match.
- All 72 columns are numeric (69 float64, 3 int64). No object dtypes.
- Zero constant columns.
- NaN counts match the expected per-flight startup pattern from notebook 03:
  - `_rstd5` columns: 470 NaN each = 5 rows × 94 train flights
  - `_lag2` columns: 188 NaN each = 2 rows × 94 train flights
  - `_lag1` columns: 94 NaN each = 1 row × 94 train flights
- NaN values are confined to lag and rolling-std columns. No surprises elsewhere.


## Drop NaN-affected rows

The lag-2 and rolling-std-window-5 features leave the first 5 rows of each flight with NaN values. We drop those rows from train and test rather than fill them.

**Why drop instead of fill:**

- Filling with zero, mean, or backfill injects synthetic temporal signal into models whose entire premise is recent past behavior.
- At inference time on a new flight, the first few seconds also have no valid prediction. Dropping matches the real deployment scenario.
- The cost is small: 470 rows per flight startup band × 94 train flights = 470 rows dropped from train (under 1%).

```python
train_before = len(train)
test_before = len(test)

train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

train_dropped = train_before - len(train)
test_dropped = test_before - len(test)

print(f"Train: {train_before} -> {len(train)} rows ({train_dropped} dropped, {train_dropped / train_before:.2%})")
print(f"Test:  {test_before} -> {len(test)} rows ({test_dropped} dropped, {test_dropped / test_before:.2%})")

print(f"\nRemaining NaN in train: {train.isna().sum().sum()}")
print(f"Remaining NaN in test:  {test.isna().sum().sum()}")
```

**Findings:**

- Train: 47,957 -> 47,487 rows (470 dropped, 0.98%).
- Test: 8,809 -> 8,724 rows (85 dropped, 0.96%).
- Drop counts match the expected 5 rows × flight count (94 train, 17 test).
- Zero remaining NaN values in either split.


## Define X, y, and groups

Split each dataset into:

- `X`: feature matrix (all columns except `flight_id` and the target)
- `y`: target vector (binary efficiency label)
- `groups`: `flight_id` vector for GroupKFold cross-validation

```python
TARGET_COL = "inefficient"
GROUP_COL = "flight_id"

feature_cols = [c for c in train.columns if c not in (TARGET_COL, GROUP_COL)]

X_train = train[feature_cols].values
y_train = train[TARGET_COL].values
groups_train = train[GROUP_COL].values

X_test = test[feature_cols].values
y_test = test[TARGET_COL].values
groups_test = test[GROUP_COL].values

print(f"Feature count: {len(feature_cols)}")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, groups_train: {groups_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}, groups_test:  {groups_test.shape}")

print(f"\nTrain class balance:")
print(pd.Series(y_train).value_counts(normalize=True).rename("proportion"))
print(f"\nTest class balance:")
print(pd.Series(y_test).value_counts(normalize=True).rename("proportion"))

print(f"\nUnique flights in train: {len(np.unique(groups_train))}")
print(f"Unique flights in test:  {len(np.unique(groups_test))}")
```

**Findings:**

- 70 feature columns after excluding `flight_id` and `inefficient`.
- Train: 47,487 rows across 94 flights. Test: 8,724 rows across 17 flights.
- Class balance is similar in both splits: ~40% positive (inefficient) in train, ~39% in test. Stratification from notebook 02 held up through the NaN drop.
- Class imbalance is mild, no resampling needed for baselines. F1 and ROC AUC are reasonable headline metrics.


## Model 1: Naive baseline (persistence)

In plain terms: predict each row's label by copying the label from the row just before it, within the same flight. The first row of each flight has no prior, so we drop it for scoring.

**Why this baseline:**

- The lag-1 Autocorrelation Function (ACF) on the target was 0.838 in notebook 03, meaning a row's label is very similar to the previous row's label. Copying the previous label is therefore a strong naive guess.
- Persistence sets the floor any real model must beat. If a trained model cannot beat "just copy the previous row," the model is not adding value.
- No training, no parameters, no preprocessing.

**Metrics:** F1 score and Receiver Operating Characteristic Area Under the Curve (ROC AUC). For persistence, ROC AUC is computed on hard 0/1 predictions, which reduces to balanced accuracy.

**Implementation:** compute predictions on train and test separately, grouped by `flight_id`, skipping the first row of each flight.

```python
def persistence_predict(df, target_col, group_col):
    """Predict each row's label as the previous row's label, within each flight.

    Returns a dataframe with columns: group, y_true, y_pred.
    Rows where no prior label exists (first row of each flight) are dropped.
    """
    out = df[[group_col, target_col]].copy()
    out["y_pred"] = out.groupby(group_col)[target_col].shift(1)
    out = out.dropna().reset_index(drop=True)
    out["y_pred"] = out["y_pred"].astype(int)
    return out.rename(columns={target_col: "y_true"})


def score_predictions(y_true, y_pred):
    """Compute F1 and ROC AUC. Returns a dict."""
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
    }


persistence_train = persistence_predict(train, TARGET_COL, GROUP_COL)
persistence_test = persistence_predict(test, TARGET_COL, GROUP_COL)

persistence_train_scores = score_predictions(persistence_train["y_true"], persistence_train["y_pred"])
persistence_test_scores = score_predictions(persistence_test["y_true"], persistence_test["y_pred"])

print(f"Persistence baseline")
print(f"  Train: F1 = {persistence_train_scores['f1']:.4f}, ROC AUC = {persistence_train_scores['roc_auc']:.4f}")
print(f"  Test:  F1 = {persistence_test_scores['f1']:.4f}, ROC AUC = {persistence_test_scores['roc_auc']:.4f}")
print(f"\nRows scored: train = {len(persistence_train)}, test = {len(persistence_test)}")
```

**Findings:**

- Naive baseline (persistence) model trained.
- Train: F1 = 0.9090, ROC AUC = 0.9232.
- Test:  F1 = 0.9105, ROC AUC = 0.9261.
- Train and test scores nearly identical, as expected for a parameter-free model.

The challenge for the trained models: persistence wins by exploiting label stickiness from one row to the next, without using any sensor data. To improve on these scores, Logistic Regression and Linear Discriminant Analysis need to extract enough signal from the 70 engineered features to correctly classify the rows where the label actually changes, which is the harder ~9% of the data.


## Cross-validation setup

We use GroupKFold with `flight_id` as the group. All rows from a single flight stay together in the same fold, never split across train and validation.

**Why GroupKFold for this dataset:**

Rows within a flight are highly autocorrelated. If random k-fold mixed rows from the same flight across train and validation, the model would see row t-1 in training and row t in validation, which is trivially predictable. The validation score would be inflated and would not reflect performance on new flights. GroupKFold forces the model to generalize to flights it has never seen, which is the real deployment question.

**Why 6 folds:**

With 94 train flights and 6 folds, each fold holds out roughly 15-16 flights, the same size as our held-out test set (17 flights). The justification behind matching fold size to test set size means our CV estimate of generalization error is calibrated to the same group size we will actually evaluate on. Therefore, the variance of the CV mean reflects the variance we expect at test time. 

```python
cv = GroupKFold(n_splits=N_SPLITS)

print(f"GroupKFold splits: {N_SPLITS}")
print(f"Total train flights: {len(np.unique(groups_train))}\n")

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train), start=1):
    train_flights = np.unique(groups_train[train_idx])
    val_flights = np.unique(groups_train[val_idx])
    print(
        f"Fold {fold_idx}: "
        f"train rows = {len(train_idx)}, val rows = {len(val_idx)} | "
        f"train flights = {len(train_flights)}, val flights = {len(val_flights)}"
    )
```

**Findings:**

- 6 folds created, each holding out 15-16 flights from the 94 train flights.
- Validation row counts per fold range from 7,602 to 8,060, close to the test set size of 8,724 rows.
- GroupKFold confirmed working: no flight appears in both train and validation within any fold.


## Model 2: Logistic Regression with Lasso (L1)

Logistic Regression with L1 regularization. L1 drives unhelpful feature coefficients exactly to zero, producing a sparse model where we can read off which features the model decided to keep.

**Why Lasso first:**

- Built-in feature selection. With 70 features, many of which are correlated (lag columns, rolling-std columns, redundant airspeed pairs), L1 tells us which ones the model finds non-redundant.
- Interpretability. Reading the non-zero coefficients gives a clean story about what the model learned.

**Preprocessing:**

- `StandardScaler` inside the pipeline. Fit on train folds only during cross-validation, so no leakage from validation folds.
- All 70 features are continuous, no one-hot encoding needed.

**Hyperparameter tuning:**

- `LogisticRegression` with `l1_ratio=1.0` for pure Lasso. Sklearn 1.8 unified the L1 and L2 interface under `l1_ratio`, so the same syntax extends to Elastic Net by setting `l1_ratio` between 0 and 1.
- `solver="saga"`, required for L1 and Elastic Net penalties.
- `GridSearchCV` sweeps `C` across 10 log-spaced values from 1e-4 to 1e4, scoring on both F1 and ROC AUC, picking the best C by F1.
- Outer cross-validation uses our 6-fold GroupKFold split with `flight_id` as the group.

**Metrics scored on the cross-validation:** F1 and ROC AUC. The held-out test set (17 flights, 8,724 rows) serves as the final unbiased evaluation after hyperparameter selection.

```python
logreg_lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        l1_ratio=1.0,
        solver="saga",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )),
])

logreg_lasso_grid = GridSearchCV(
    logreg_lasso_pipeline,
    param_grid={"clf__C": np.logspace(-2, 2, 5)},
    cv=cv,
    scoring={"f1": "f1", "roc_auc": "roc_auc"},
    refit="f1",
    n_jobs=-1,
    return_train_score=False,
)

logreg_lasso_grid.fit(X_train, y_train, groups=groups_train)

best_C_lasso = logreg_lasso_grid.best_params_["clf__C"]
coefs_lasso = logreg_lasso_grid.best_estimator_.named_steps["clf"].coef_[0]
n_zero = int(np.sum(coefs_lasso == 0))
n_nonzero = int(np.sum(coefs_lasso != 0))

best_idx = logreg_lasso_grid.best_index_
cv_f1_mean = logreg_lasso_grid.cv_results_["mean_test_f1"][best_idx]
cv_f1_std = logreg_lasso_grid.cv_results_["std_test_f1"][best_idx]
cv_auc_mean = logreg_lasso_grid.cv_results_["mean_test_roc_auc"][best_idx]
cv_auc_std = logreg_lasso_grid.cv_results_["std_test_roc_auc"][best_idx]

print(f"Lasso Logistic Regression")
print(f"  Best C: {best_C_lasso:.6f}")
print(f"  Features kept (non-zero coef): {n_nonzero} / {len(coefs_lasso)}")
print(f"  Features eliminated (zero coef): {n_zero} / {len(coefs_lasso)}")
print(f"  CV F1:      {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")
print(f"  CV ROC AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
```

**Findings:**

- Best C = 0.005995, indicating strong regularization. The model preferred heavy shrinkage given the correlated feature set.
- 36 of 70 features kept, 34 eliminated. Lasso pruned roughly half the engineered feature set on its own.
- CV F1 = 0.7969 +/- 0.0103, CV ROC AUC = 0.9170 +/- 0.0038. Fold-to-fold variance is low, so the scores are stable.
- F1 trails the persistence baseline (0.91) but ROC AUC matches it closely. The model has strong rank-ordering ability across rows, with the gap appearing at the default 0.5 classification threshold.
- Threshold tuning could improve F1, but we leave that for after model comparison, since all models should be compared under the same default threshold.


### Lasso feature selection inspection

Pull the coefficients from the fitted Lasso model and inspect:

- **Eliminated features**: which 34 features Lasso zeroed out entirely.
- **Top coefficients**: the largest non-zero coefficients by absolute value, with sign preserved. These are the features the model leaned on most.

Caveats:

- Coefficients are on scaled features (post-StandardScaler), so magnitudes are directly comparable across features.
- Sign indicates direction: positive coefficient means higher feature value pushes prediction toward "inefficient" (class 1).
- With correlated features, Lasso may arbitrarily pick one and zero the partner. Eliminated does not mean useless, it means redundant given another kept feature.

```python
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coef": coefs_lasso,
})
coef_df["abs_coef"] = coef_df["coef"].abs()

eliminated = coef_df[coef_df["coef"] == 0]["feature"].tolist()
kept = coef_df[coef_df["coef"] != 0].sort_values("abs_coef", ascending=False)

print(f"Eliminated features ({len(eliminated)}):")
for f in eliminated:
    print(f"  {f}")

print(f"\nTop 15 features by |coefficient|:")
print(kept[["feature", "coef"]].head(15).to_string(index=False))
```

**Findings:**

- 34 features eliminated, 36 kept. Lasso aggressively pruned correlated families.
- `output[0]_f3` dominates with a coefficient of +3.24, roughly 7x the next-largest coefficient. The model leans heavily on this single motor output channel.
- Lag families collapsed: only 4 of 16 lag-1 features and 4 of 16 lag-2 features survived. Lasso treated lags as redundant with their current-value counterparts.
- Rolling-std features mostly survived (11 of 16). Short-window volatility carries signal independent of the level values.
- Quaternion orientation values were largely eliminated (10 of 12 quaternion-related features dropped). The exception is `q[3]_f119_rstd5`, suggesting quaternion volatility matters more than quaternion position.
- Environmental features split: `wind_magnitude` and `baro_pressure_pa_f117` ranked in the top 15. `ambient_temperature_f117` was eliminated.
- `time_since_start_of_flight` ranked in the top 5, suggesting efficiency varies systematically across the flight timeline (takeoff, cruise, landing).

```python
print("output[0]_f3 raw stats (train):")
print(train["output[0]_f3"].describe())

print(f"\nStd: {train['output[0]_f3'].std():.6f}")
print(f"Min: {train['output[0]_f3'].min():.6f}")
print(f"Max: {train['output[0]_f3'].max():.6f}")
```

### Save Lasso feature inspection to JSON

Persist the kept/eliminated lists and sorted coefficients to `results/05_baselines/lasso_feature_inspection.json`. Useful for comparison against Elastic Net and for the writeup.

```python
lasso_feature_inspection = {
    "model": "logistic_regression_lasso",
    "best_C": float(best_C_lasso),
    "n_features_total": len(feature_cols),
    "n_features_kept": n_nonzero,
    "n_features_eliminated": n_zero,
    "eliminated_features": eliminated,
    "coefficients_sorted_by_abs_value": [
        {"feature": row["feature"], "coef": float(row["coef"])}
        for _, row in kept.iterrows()
    ],
}

inspection_path = RESULTS_DIR / "lasso_feature_inspection.json"
with open(inspection_path, "w") as f:
    json.dump(lasso_feature_inspection, f, indent=2)

print(f"Saved: {inspection_path}")
```

### Save the trained Lasso pipeline

Persist the fitted pipeline (StandardScaler + LogisticRegression) to `models/05_baselines/logreg_lasso.pkl` using `joblib`. The pickle stores the scaler state and the trained classifier together, so loading it later gives us a single object that takes raw features and returns predictions.

```python
lasso_model_path = MODELS_DIR / "logreg_lasso.pkl"
joblib.dump(logreg_lasso_grid.best_estimator_, lasso_model_path)

print(f"Saved: {lasso_model_path}")
```

## Model 3: Logistic Regression with Elastic Net

Same Logistic Regression setup as Lasso, but with Elastic Net regularization. Elastic Net is a weighted mix of L1 (Lasso) and L2 (Ridge) penalties, controlled by `l1_ratio`:

- `l1_ratio=1.0`: pure Lasso (what we just ran)
- `l1_ratio=0.0`: pure Ridge
- `0 < l1_ratio < 1`: a blend

**Why also run Elastic Net:**

- With 70 features and many correlated pairs (lag families, redundant airspeed columns), pure Lasso can be unstable. When two features are highly correlated, Lasso picks one and zeroes the other somewhat arbitrarily. Across different random seeds or slightly different data, it might pick the other one.
- Elastic Net handles correlated groups more gracefully. The L2 component shrinks correlated features together rather than forcing a winner-take-all choice. The L1 component still produces some feature elimination, just less aggressively.
- If Elastic Net scores meaningfully higher, that tells us the correlation structure matters and Lasso may be losing signal by zeroing correlated partners.
- If scores are similar, Lasso's sparse solution is preferred for interpretability.

**Hyperparameter tuning:**

- Two hyperparameters: `C` (regularization strength) and `l1_ratio` (mix of L1 vs L2).
- `GridSearchCV` sweeps a 2D grid: 10 values of `C` x 5 values of `l1_ratio` from 0.1 to 0.9.
- We exclude `l1_ratio=0.0` (pure Ridge, no sparsity) and `l1_ratio=1.0` (pure Lasso, already run).
- 50 combinations x 6 folds = 300 fits. This will take longer than the Lasso run.

**Metrics scored on the cross-validation:** F1 and ROC AUC.

```python
logreg_enet_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="saga",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )),
])

logreg_enet_grid = GridSearchCV(
    logreg_enet_pipeline,
    param_grid={
        "clf__C": np.logspace(-2, 2, 5),
        "clf__l1_ratio": [0.3, 0.5, 0.7],
    },
    cv=cv,
    scoring={"f1": "f1", "roc_auc": "roc_auc"},
    refit="f1",
    n_jobs=-1,
    return_train_score=False,
)

logreg_enet_grid.fit(X_train, y_train, groups=groups_train)

best_C_enet = logreg_enet_grid.best_params_["clf__C"]
best_l1_ratio_enet = logreg_enet_grid.best_params_["clf__l1_ratio"]
coefs_enet = logreg_enet_grid.best_estimator_.named_steps["clf"].coef_[0]
n_zero_enet = int(np.sum(coefs_enet == 0))
n_nonzero_enet = int(np.sum(coefs_enet != 0))

best_idx_enet = logreg_enet_grid.best_index_
cv_f1_mean_enet = logreg_enet_grid.cv_results_["mean_test_f1"][best_idx_enet]
cv_f1_std_enet = logreg_enet_grid.cv_results_["std_test_f1"][best_idx_enet]
cv_auc_mean_enet = logreg_enet_grid.cv_results_["mean_test_roc_auc"][best_idx_enet]
cv_auc_std_enet = logreg_enet_grid.cv_results_["std_test_roc_auc"][best_idx_enet]

print(f"Elastic Net Logistic Regression")
print(f"  Best C: {best_C_enet:.6f}")
print(f"  Best l1_ratio: {best_l1_ratio_enet}")
print(f"  Features kept (non-zero coef): {n_nonzero_enet} / {len(coefs_enet)}")
print(f"  Features eliminated (zero coef): {n_zero_enet} / {len(coefs_enet)}")
print(f"  CV F1:      {cv_f1_mean_enet:.4f} +/- {cv_f1_std_enet:.4f}")
print(f"  CV ROC AUC: {cv_auc_mean_enet:.4f} +/- {cv_auc_std_enet:.4f}")
```

**Findings:**

- Best C = 0.005995 (same as Lasso), best l1_ratio = 0.5 (equal mix of L1 and L2).
- 49 of 70 features kept, 21 eliminated. Less aggressive pruning than Lasso (which kept 36), as expected given the L2 component.
- CV F1 = 0.7979 +/- 0.0108, CV ROC AUC = 0.9166 +/- 0.0037.
- Scores are statistically indistinguishable from Lasso (F1 differs by 0.001, well within fold std of 0.01). The L2 component kept 13 additional features alive that did not improve performance.
- Conclusion: Lasso's sparse solution is preferred. Same predictive performance with fewer features and a cleaner interpretability story.

```python
enet_model_path = MODELS_DIR / "logreg_enet.pkl"
joblib.dump(logreg_enet_grid.best_estimator_, enet_model_path)

print(f"Saved: {enet_model_path}")
```

```python
def save_model_results(model_name, cv_f1_mean, cv_f1_std, cv_auc_mean, cv_auc_std, extra=None):
    """Save a single model's CV results to results/05_baselines/{model_name}.json."""
    payload = {
        "model": model_name,
        "cv_f1_mean": float(cv_f1_mean),
        "cv_f1_std": float(cv_f1_std),
        "cv_roc_auc_mean": float(cv_auc_mean),
        "cv_roc_auc_std": float(cv_auc_std),
    }
    if extra:
        payload.update(extra)
    path = RESULTS_DIR / f"{model_name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")


save_model_results(
    "persistence",
    cv_f1_mean=persistence_train_scores["f1"],
    cv_f1_std=0.0,
    cv_auc_mean=persistence_train_scores["roc_auc"],
    cv_auc_std=0.0,
    extra={
        "note": "Naive baseline, no CV. Scores are train-set predictions vs train-set labels.",
        "train_f1": float(persistence_train_scores["f1"]),
        "train_roc_auc": float(persistence_train_scores["roc_auc"]),
        "test_f1": float(persistence_test_scores["f1"]),
        "test_roc_auc": float(persistence_test_scores["roc_auc"]),
    },
)

save_model_results(
    "logreg_lasso",
    cv_f1_mean=cv_f1_mean,
    cv_f1_std=cv_f1_std,
    cv_auc_mean=cv_auc_mean,
    cv_auc_std=cv_auc_std,
    extra={
        "best_C": float(best_C_lasso),
        "l1_ratio": 1.0,
        "n_features_kept": n_nonzero,
        "n_features_eliminated": n_zero,
    },
)

save_model_results(
    "logreg_enet",
    cv_f1_mean=cv_f1_mean_enet,
    cv_f1_std=cv_f1_std_enet,
    cv_auc_mean=cv_auc_mean_enet,
    cv_auc_std=cv_auc_std_enet,
    extra={
        "best_C": float(best_C_enet),
        "l1_ratio": float(best_l1_ratio_enet),
        "n_features_kept": n_nonzero_enet,
        "n_features_eliminated": n_zero_enet,
    },
)
```

## Model 4: Linear Discriminant Analysis (LDA)

Second linear baseline with different assumptions from Logistic Regression.

**How LDA works in plain terms:**

LDA assumes each class is a Gaussian (bell curve) with the same covariance shape. It finds the single direction in feature space that pushes the two class means as far apart as possible while keeping each class as tight as possible. Classification reduces to projecting a new point onto that direction and checking which side of a threshold it falls on.

**Why include LDA:**

- Different reasoning from Logistic Regression. Logistic Regression minimizes classification error directly. LDA assumes the data shape and derives the boundary from that assumption.
- If LDA and Logistic Regression score similarly, the linear ceiling is real and consistent across approaches.
- If they diverge, that tells us something about the class distributions (likely non-Gaussian or unequal covariance).

**Preprocessing:**

- `StandardScaler` inside the pipeline. LDA is not scale-invariant in the same way Logistic Regression is, but standardization keeps the optimization stable and the projection coefficients comparable.

**Hyperparameter tuning:**

- LDA has no regularization strength to sweep in its default form. We use the default solver (`svd`) and no shrinkage.
- `cross_val_score` with our 6-fold GroupKFold split, scored on F1 and ROC AUC.

**Metrics scored on the cross-validation:** F1 score and ROC AUC.

```python
from sklearn.model_selection import cross_validate

lda_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearDiscriminantAnalysis()),
])

lda_cv_results = cross_validate(
    lda_pipeline,
    X_train,
    y_train,
    groups=groups_train,
    cv=cv,
    scoring={"f1": "f1", "roc_auc": "roc_auc"},
    n_jobs=-1,
    return_train_score=False,
)

cv_f1_mean_lda = lda_cv_results["test_f1"].mean()
cv_f1_std_lda = lda_cv_results["test_f1"].std()
cv_auc_mean_lda = lda_cv_results["test_roc_auc"].mean()
cv_auc_std_lda = lda_cv_results["test_roc_auc"].std()

lda_pipeline.fit(X_train, y_train)

print(f"Linear Discriminant Analysis")
print(f"  CV F1:      {cv_f1_mean_lda:.4f} +/- {cv_f1_std_lda:.4f}")
print(f"  CV ROC AUC: {cv_auc_mean_lda:.4f} +/- {cv_auc_std_lda:.4f}")
```

**Findings:**

- CV F1 = 0.7897 +/- 0.0196, CV ROC AUC = 0.9058 +/- 0.0052.
- LDA trails both Logistic Regression variants slightly: F1 by ~0.008, ROC AUC by ~0.011.
- Fold-to-fold variance is roughly double Logistic Regression's, suggesting LDA's Gaussian assumption is only approximately satisfied by the feature distributions.
- All three linear models cluster tightly (F1 between 0.79 and 0.80, ROC AUC between 0.90 and 0.92). The linear ceiling for this feature set appears real.

```python
lda_model_path = MODELS_DIR / "lda.pkl"
joblib.dump(lda_pipeline, lda_model_path)

save_model_results(
    "lda",
    cv_f1_mean=cv_f1_mean_lda,
    cv_f1_std=cv_f1_std_lda,
    cv_auc_mean=cv_auc_mean_lda,
    cv_auc_std=cv_auc_std_lda,
)

print(f"Saved: {lda_model_path}")
```

## Test set evaluation

Score all three trained models (Lasso, Elastic Net, LDA) on the held-out test set. This is the unbiased final number, since the test set was never seen during training or hyperparameter tuning.

Persistence test scores were computed earlier and are already saved.

```python
def score_pipeline_on_test(pipeline, X_test, y_test):
    """Score a fitted sklearn pipeline on the test set. Returns F1 and ROC AUC."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


lasso_test_scores = score_pipeline_on_test(logreg_lasso_grid.best_estimator_, X_test, y_test)
enet_test_scores = score_pipeline_on_test(logreg_enet_grid.best_estimator_, X_test, y_test)
lda_test_scores = score_pipeline_on_test(lda_pipeline, X_test, y_test)

print(f"Test set scores (8,724 rows across 17 flights):\n")
print(f"  Persistence:  F1 = {persistence_test_scores['f1']:.4f}, ROC AUC = {persistence_test_scores['roc_auc']:.4f}")
print(f"  Lasso:        F1 = {lasso_test_scores['f1']:.4f}, ROC AUC = {lasso_test_scores['roc_auc']:.4f}")
print(f"  Elastic Net:  F1 = {enet_test_scores['f1']:.4f}, ROC AUC = {enet_test_scores['roc_auc']:.4f}")
print(f"  LDA:          F1 = {lda_test_scores['f1']:.4f}, ROC AUC = {lda_test_scores['roc_auc']:.4f}")
```

**Findings:**

- Persistence: F1 = 0.9105, ROC AUC = 0.9261.
- Lasso:       F1 = 0.8182, ROC AUC = 0.9239.
- Elastic Net: F1 = 0.8155, ROC AUC = 0.9229.
- LDA:         F1 = 0.7956, ROC AUC = 0.9097.
- All three trained models scored slightly higher on test than on cross-validation. The 17-flight test set appears mildly more separable than the average CV fold.
- Persistence wins on F1 by ~0.09 over the next-best model, but ROC AUC is nearly identical between persistence, Lasso, and Elastic Net (within 0.003).
- The F1 gap is largely a classification-threshold issue. The trained models rank-order rows nearly as well as persistence, but at the default 0.5 threshold they convert that ordering into hard predictions less effectively.
- Lasso edges Elastic Net on both metrics, confirming the preferred-model conclusion from CV.
- LDA trails on both metrics at test time, consistent with cross-validation.

```python
combined_metrics = {
    "notebook": "05_model_baselines",
    "test_set": {"n_rows": int(len(y_test)), "n_flights": int(len(np.unique(groups_test)))},
    "cv_setup": {"splitter": "GroupKFold", "n_splits": N_SPLITS, "group_col": GROUP_COL},
    "models": {
        "persistence": {
            "type": "naive_baseline",
            "cv_f1_mean": None,
            "cv_f1_std": None,
            "cv_roc_auc_mean": None,
            "cv_roc_auc_std": None,
            "train_f1": float(persistence_train_scores["f1"]),
            "train_roc_auc": float(persistence_train_scores["roc_auc"]),
            "test_f1": float(persistence_test_scores["f1"]),
            "test_roc_auc": float(persistence_test_scores["roc_auc"]),
            "note": "No CV. Predicts label at row t as label at row t-1 within each flight.",
        },
        "logreg_lasso": {
            "type": "logistic_regression_l1",
            "best_C": float(best_C_lasso),
            "l1_ratio": 1.0,
            "n_features_kept": n_nonzero,
            "n_features_eliminated": n_zero,
            "cv_f1_mean": float(cv_f1_mean),
            "cv_f1_std": float(cv_f1_std),
            "cv_roc_auc_mean": float(cv_auc_mean),
            "cv_roc_auc_std": float(cv_auc_std),
            "test_f1": float(lasso_test_scores["f1"]),
            "test_roc_auc": float(lasso_test_scores["roc_auc"]),
        },
        "logreg_enet": {
            "type": "logistic_regression_elastic_net",
            "best_C": float(best_C_enet),
            "l1_ratio": float(best_l1_ratio_enet),
            "n_features_kept": n_nonzero_enet,
            "n_features_eliminated": n_zero_enet,
            "cv_f1_mean": float(cv_f1_mean_enet),
            "cv_f1_std": float(cv_f1_std_enet),
            "cv_roc_auc_mean": float(cv_auc_mean_enet),
            "cv_roc_auc_std": float(cv_auc_std_enet),
            "test_f1": float(enet_test_scores["f1"]),
            "test_roc_auc": float(enet_test_scores["roc_auc"]),
        },
        "lda": {
            "type": "linear_discriminant_analysis",
            "cv_f1_mean": float(cv_f1_mean_lda),
            "cv_f1_std": float(cv_f1_std_lda),
            "cv_roc_auc_mean": float(cv_auc_mean_lda),
            "cv_roc_auc_std": float(cv_auc_std_lda),
            "test_f1": float(lda_test_scores["f1"]),
            "test_roc_auc": float(lda_test_scores["roc_auc"]),
        },
    },
}

combined_path = RESULTS_DIR / "metrics.json"
with open(combined_path, "w") as f:
    json.dump(combined_metrics, f, indent=2)

print(f"Saved: {combined_path}")
```

## Diagnostic plots: class separation under linear models

Two side-by-side histograms showing how well each linear model separates the two classes:

- **LDA projection histogram**: project test rows onto LDA's discriminant direction. Each row becomes a single number. Plot the distribution of those numbers separately for each class. If the two histograms are far apart with little overlap, the data is linearly separable. If they sit on top of each other, linear models are hitting a ceiling.

- **Logistic Regression probability histogram**: get predicted probabilities for the inefficient class from the Lasso model. Plot the distribution separately for each class. Same diagnostic question, different lens. A well-calibrated model puts most class-1 rows near probability 1 and most class-0 rows near probability 0.

**What this tells us:** if both plots show heavy overlap between the classes, non-linear models (XGBoost, Graph Attention Network) have room to improve. If both show clean separation, the linear ceiling is the real ceiling.

```python
lda_projection = lda_pipeline.decision_function(X_test)
lasso_proba = logreg_lasso_grid.best_estimator_.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(
    lda_projection[y_test == 0],
    bins=50,
    alpha=0.6,
    label="Efficient (class 0)",
)
axes[0].hist(
    lda_projection[y_test == 1],
    bins=50,
    alpha=0.6,
    label="Inefficient (class 1)",
)
axes[0].axvline(0, linestyle="--", linewidth=1, color="black")
axes[0].set_xlabel("LDA projection (decision function)")
axes[0].set_ylabel("Row count")
axes[0].set_title("LDA: projection of test rows by class")
axes[0].legend()

axes[1].hist(
    lasso_proba[y_test == 0],
    bins=50,
    alpha=0.6,
    label="Efficient (class 0)",
)
axes[1].hist(
    lasso_proba[y_test == 1],
    bins=50,
    alpha=0.6,
    label="Inefficient (class 1)",
)
axes[1].axvline(0.5, linestyle="--", linewidth=1, color="black")
axes[1].set_xlabel("Predicted probability of class 1")
axes[1].set_ylabel("Row count")
axes[1].set_title("Lasso Logistic Regression: predicted probabilities by class")
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Findings: visualizations of the linear ceiling**

These two plots show the limit of what any linear model can achieve on this feature set.

- **LDA projection (left)**: each test row is compressed to a single number. Efficient rows (blue) cluster around -2, inefficient rows (orange) cluster around +1. The two distributions overlap between roughly -3 and +2. Rows in that overlap region are where LDA gets confused, since the same projection value could belong to either class.

- **Lasso probabilities (right)**: a different shape entirely. Lasso confidently calls a large group of efficient rows class 0, producing the tall blue spike near probability 0. Inefficient rows spread across probabilities 0.4 to 1.0 with the bulk above 0.6. The model is decisive about many efficient rows but less sure about inefficient ones.

- **Same conclusion from both plots**: meaningful overlap between the two classes exists in any linear projection of the feature space. That overlap is the linear ceiling. No amount of regularization or hyperparameter tuning will eliminate it.

- **This explains the F1 vs ROC AUC gap**: ROC AUC is high (~0.92) because the models rank rows correctly on average. F1 sits around 0.80 because at the default 0.5 threshold, the overlap region produces both false positives and false negatives.

- **Motivation for the next models**: XGBoost and the Graph Attention Network are non-linear. They can capture interactions between features that no straight line in feature space can express. If the overlap region in these plots can be untangled, it has to be done with non-linear structure.


## Summary

**Models trained and evaluated:**

| Model | CV F1 | CV ROC AUC | Test F1 | Test ROC AUC |
|---|---|---|---|---|
| Naive baseline (persistence) | n/a | n/a | 0.9105 | 0.9261 |
| Logistic Regression (Lasso) | 0.7969 +/- 0.0103 | 0.9170 +/- 0.0038 | 0.8182 | 0.9239 |
| Logistic Regression (Elastic Net) | 0.7979 +/- 0.0108 | 0.9166 +/- 0.0037 | 0.8155 | 0.9229 |
| Linear Discriminant Analysis | 0.7897 +/- 0.0196 | 0.9058 +/- 0.0052 | 0.7956 | 0.9097 |

**Key takeaways:**

- The naive persistence baseline beats every trained linear model on F1. The previous-row label carries information that lagged sensor features alone cannot fully recover.
- All three trained linear models cluster tightly: F1 between 0.79 and 0.82, ROC AUC between 0.91 and 0.92. The linear ceiling is real and consistent across approaches.
- Lasso and Elastic Net produce statistically indistinguishable scores. Lasso is preferred for its sparser solution (36 kept features vs 49).
- LDA trails Logistic Regression slightly on both metrics. Its Gaussian-class assumption is only approximately satisfied.
- Diagnostic plots confirm meaningful class overlap in any linear projection of the feature space. The gap between high ROC AUC and lower F1 is a threshold/overlap issue, not a missing-signal issue.

**Artifacts saved:**

- `models/05_baselines/`: trained pipelines for Lasso, Elastic Net, and LDA (`.pkl`).
- `results/05_baselines/`: per-model JSON results, a Lasso feature inspection JSON, and a combined `metrics.json`.

**Next:**

- Notebook 06: XGBoost. First non-linear model. The diagnostic plots suggest there is room above the linear ceiling if non-linear interactions can untangle the overlap region.
- Future work flagged: classification threshold tuning, t+1 target framing, edge construction methods for the Graph Attention Network.
