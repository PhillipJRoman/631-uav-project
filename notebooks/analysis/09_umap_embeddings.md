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

# Notebook 09: UMAP analysis of GAT embeddings

Post-modeling visualization. The hybrid result in notebook 08 showed
the Graph Attention Network embeddings carried redundant signal:
XGBoost used them 66.6% of the time during training (built-in gain)
but only 12.2% at prediction time (permutation importance). This
notebook visualizes that finding.

## Goal

Project the 70 raw engineered features and the 256 GAT embeddings
into 2D using UMAP, color by the efficiency label, and compare. The
question is whether the GAT learned structure that the raw features
do not already expose.

Three possible outcomes:

- **GAT embeddings show cleaner class separation.** The graph attention
  layer found structure the raw features hide. Would partially
  contradict notebook 08's permutation finding.
- **Both spaces show similar separation.** GAT re-encoded what was
  already there. Confirms notebook 08.
- **Raw features show cleaner separation.** The GAT compressed and
  lost signal during message passing. Also consistent with notebook 08.

## Approach

- Load train and test splits and the 256D Graph Attention Network
  embeddings from notebook 07
- Sample 5000 rows from each (full dataset is too dense to read on a
  scatter plot; sampling preserves the shape while keeping the
  visualization legible)
- Fit two UMAP models with identical hyperparameters: one on raw
  features, one on embeddings
- Plot side by side, colored by `inefficient` label
- Compute a quantitative class-separation score on each projection to
  back up the visual story

```python
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use("tableau-colorblind10")

SEED = 631
SAMPLE_SIZE = 5000

REPO_ROOT = Path("/Users/phillipromanmacbook/GitHub/631-uav-project")
DATA_DIR = REPO_ROOT / "data" / "splits"
EMB_DIR = REPO_ROOT / "data" / "embeddings" / "07_gat"
RESULTS_DIR = REPO_ROOT / "results" / "09_umap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

print(f"UMAP version: {umap.__version__}")
print(f"Data dir: {DATA_DIR}")
print(f"Embeddings dir: {EMB_DIR}")
print(f"Results dir: {RESULTS_DIR}")
```

## Load data and prepare feature matrices

Load the train splits and the GAT embeddings from notebook 07. We
work with train data only here since the test set is small (8,724
rows) and we want a clean representative sample of the full data
distribution.

After loading:

- Drop startup NaN rows (same logic as every other notebook)
- Confirm row alignment between raw and embedding files
- Build two feature matrices:
  - `X_raw`: the 70 engineered features
  - `X_emb`: the 256 GAT embedding columns
- Capture the `inefficient` label for coloring

```python
train_raw = pd.read_csv(DATA_DIR / "train.csv")
train_emb = pd.read_parquet(EMB_DIR / "train_embeddings.parquet")

train_raw = train_raw.dropna().reset_index(drop=True)

print(f"Raw train shape after dropna: {train_raw.shape}")
print(f"Embeddings shape: {train_emb.shape}")
print(f"Row counts aligned: {len(train_raw) == len(train_emb)}")

emb_cols = [c for c in train_emb.columns if c.startswith("gat_emb_")]
raw_feature_cols = [c for c in train_raw.columns if c not in ("flight_id", "inefficient")]

X_raw = train_raw[raw_feature_cols].values
X_emb = train_emb[emb_cols].values
y = train_raw["inefficient"].values

print(f"\nX_raw shape: {X_raw.shape}")
print(f"X_emb shape: {X_emb.shape}")
print(f"y shape: {y.shape}")
print(f"Class balance: {y.mean():.3f} inefficient")
```

**Findings:**

- Raw features matrix: 47,487 rows x 70 features
- GAT embeddings matrix: 47,487 rows x 256 features
- Row counts aligned between the two files
- Class balance 40.7% inefficient, same as upstream notebooks


## Sample and scale

UMAP on all 47,487 points is dense and slow. We sample ~50 rows per
flight, stratified by the `inefficient` label within each flight.
This covers all 94 flights, preserves class balance, and avoids
oversampling any single flight's behavior.

Features are then standardized (mean 0, std 1) since UMAP is
distance-based.

```python
ROWS_PER_FLIGHT = 50

flight_ids_all = train_raw["flight_id"].values

rng = np.random.default_rng(SEED)
sample_idx_list = []

for fid in np.unique(flight_ids_all):
    flight_mask = flight_ids_all == fid
    flight_rows = np.where(flight_mask)[0]
    flight_labels = y[flight_rows]

    pos_rows = flight_rows[flight_labels == 1]
    neg_rows = flight_rows[flight_labels == 0]
    flight_pos_rate = len(pos_rows) / len(flight_rows)

    n_pos = int(round(ROWS_PER_FLIGHT * flight_pos_rate))
    n_neg = ROWS_PER_FLIGHT - n_pos

    n_pos = min(n_pos, len(pos_rows))
    n_neg = min(n_neg, len(neg_rows))

    if n_pos > 0:
        sample_idx_list.append(rng.choice(pos_rows, size=n_pos, replace=False))
    if n_neg > 0:
        sample_idx_list.append(rng.choice(neg_rows, size=n_neg, replace=False))

sample_idx = np.concatenate(sample_idx_list)

X_raw_sample = X_raw[sample_idx]
X_emb_sample = X_emb[sample_idx]
y_sample = y[sample_idx]
flight_sample = flight_ids_all[sample_idx]

raw_scaler = StandardScaler()
emb_scaler = StandardScaler()
X_raw_scaled = raw_scaler.fit_transform(X_raw_sample)
X_emb_scaled = emb_scaler.fit_transform(X_emb_sample)

print(f"Sample size: {len(sample_idx)}")
print(f"Unique flights in sample: {len(np.unique(flight_sample))}")
print(f"Sample class balance: {y_sample.mean():.3f}")
print(f"Original class balance: {y.mean():.3f}")
print(f"X_raw_scaled shape: {X_raw_scaled.shape}")
print(f"X_emb_scaled shape: {X_emb_scaled.shape}")
```

**Findings:**

- 4700 rows sampled, 50 per flight (some flights had less of one class so the sample is just under target)
- All 94 flights represented
- Sample class balance 40.4%, within 0.003 of the original 40.7%
- Both matrices scaled and ready for UMAP


## Fit UMAP and project to 2D

One UMAP per feature matrix, both with default settings, seeded for
reproducibility.

```python
umap_raw = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=SEED,
)
umap_emb = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=SEED,
)

print("Fitting UMAP on raw features...")
proj_raw = umap_raw.fit_transform(X_raw_scaled)

print("Fitting UMAP on GAT embeddings...")
proj_emb = umap_emb.fit_transform(X_emb_scaled)

print(f"\nproj_raw shape: {proj_raw.shape}")
print(f"proj_emb shape: {proj_emb.shape}")
```

## Plot both projections

Two scatter plots, raw features on the left and GAT embeddings on
the right. Points colored by label.

```python
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, proj, title in [
    (axes[0], proj_raw, "Raw features (70D)"),
    (axes[1], proj_emb, "GAT embeddings (256D)"),
]:
    for label, color, name in [(0, "C0", "efficient"), (1, "C1", "inefficient")]:
        mask = y_sample == label
        ax.scatter(proj[mask, 0], proj[mask, 1], c=color, s=5, alpha=0.5, label=name)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", framealpha=0.9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "umap_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Findings:**

**What the two plots show:**

- **Raw features (left):** Two disconnected blobs, each a roughly 50/50
  mix of efficient and inefficient. UMAP grouped the data by overall
  sensor profile, not by label. The blobs likely reflect two flight
  regimes (such as cruise versus maneuver).
- **GAT embeddings (right):** One connected region with a clear
  top-to-bottom color split. Inefficient sits in the upper half,
  efficient in the lower half. The regime split is gone.

**What this means:**

- Raw features encode efficiency categorically. The two flight regimes
  dominate the geometry, with efficiency mixed within each.
- GAT embeddings encode efficiency continuously. The model pulled the
  two classes apart along a single axis, the same way a learned
  representation should.
- The GAT smoothed out regime differences (one manifold instead of two
  blobs) and emphasized the efficiency gradient. This makes sense
  because the GAT was trained on efficiency labels, not regime labels.

**How this fits with notebook 08:**

- This does not contradict the notebook 08 finding that GAT embeddings
  contributed only 12.2% of permutation importance to the hybrid. It
  refines it.
- The GAT did learn structure the raw features do not expose directly:
  a continuous efficiency axis.
- XGBoost can build that same axis on its own by combining raw sensors
  in tree splits. The GAT's contribution is real geometrically but
  redundant predictively.
- Put differently: the GAT made the structure visible to humans, while
  XGBoost did not need that help because it can find non-linear
  combinations of raw sensors directly.
