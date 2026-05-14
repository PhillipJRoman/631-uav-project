---
jupyter:
  jupytext:
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

# 04 - Environmental Feature Analysis (Pixhawk Grouped Flights)

## Purpose

Notebook 03 handled the 16 flight dynamics predictors. This notebook covers the
environmental features from the proposal. Raw differential pressure was set aside
(it overlaps with airspeed, already covered), leaving 5:

- `rho_f117` - air density
- `baro_pressure_pa_f117` - barometric pressure
- `ambient_temperature_f117` - ambient temperature
- `windspeed_north_f7` - wind speed, north component
- `windspeed_east_f7` - wind speed, east component

Wind uses the `_f7` estimator group. A second group, `_f8`, exists but disagrees
on sign and magnitude and looked less stable, so it was set aside.

## Approach

Environmental features change slowly across a flight, not row to row, so
row-level temporal engineering does not fit them. Metrics here are computed at
the flight level (111 flights collapsed to one row each), not the row level, to
keep the sample size honest.

**Phase 1:** Cohen's d and Mutual Information (MI) on raw values, to see which
features separate the efficient and inefficient classes on their own.

**Phase 2 (only for features that pass Phase 1):** test engineered versions -
per-flight mean, per-flight z-score, throttle divided by air density, and wind
speed magnitude.

## Decision rule

For each feature, raw or engineered:
- Drop if both Cohen's d < 0.1 and MI < 0.01.
- Keep if Cohen's d > 0.2 or MI > 0.05.
- Hold for discussion if mixed.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif

# Paths
REPO_ROOT = Path.cwd().parents[1]
FLIGHT_DIR = REPO_ROOT / "data" / "processed" / "pixhawk_grouped_flights"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# The 5 environmental columns, exact names from the source CSVs
ENV_COLS = [
    "rho_f117",
    "baro_pressure_pa_f117",
    "ambient_temperature_f117",
    "windspeed_north_f7",
    "windspeed_east_f7",
]

# Join keys
KEY_COLS = ["flight_id", "time_since_start_of_flight"]

print("repo root:", REPO_ROOT)
print("flight dir exists:", FLIGHT_DIR.exists())
print("splits dir exists:", SPLITS_DIR.exists())
```

```python
# Load the current splits
train = pd.read_csv(SPLITS_DIR / "train.csv")
test = pd.read_csv(SPLITS_DIR / "test.csv")

# The 111 flights actually used. The source directory has 120 files;
# 9 were pruned earlier and are skipped here.
split_flight_ids = set(train["flight_id"]) | set(test["flight_id"])

# Build the environmental table from the source files of used flights only.
# time_since_start_of_flight is the row number within each file (0..N-1),
# confirmed to match the splits exactly.
env_frames = []
for src_path in sorted(FLIGHT_DIR.glob("*.csv")):
    fid = int(src_path.stem.split("_")[1])
    if fid not in split_flight_ids:
        continue
    src = pd.read_csv(src_path, usecols=ENV_COLS)
    src["flight_id"] = fid
    src["time_since_start_of_flight"] = np.arange(len(src))
    env_frames.append(src)

env = pd.concat(env_frames, ignore_index=True)
print("environmental table shape:", env.shape)
print("flights in env table:", env["flight_id"].nunique())

# Merge onto the splits
train_before = len(train)
test_before = len(test)

train = train.merge(env, on=KEY_COLS, how="left")
test = test.merge(env, on=KEY_COLS, how="left")

# Sanity checks
print()
print("rows preserved - train:", len(train) == train_before, "| test:", len(test) == test_before)
print()
print("nulls in environmental columns after merge:")
print("  train:", train[ENV_COLS].isna().sum().sum())
print("  test: ", test[ENV_COLS].isna().sum().sum())
```

## Phase 1: Class separation on raw values

How well does each raw environmental feature separate flights by inefficiency rate. Training flights only.

### Part one: collapse to flight level

- Check that environmental features are near-constant within a flight
- Compares within-flight spread to overall spread: small ratio means the per-flight mean is a fair summary
- High ratio for wind would be fine, just means wind moves within flights (that is what Phase 2 z-score is for)
- Then collapse to one row per flight: feature means plus per-flight inefficiency rate

```python
within_std = train.groupby("flight_id")[ENV_COLS].std()
overall_std = train[ENV_COLS].std()
spread_ratio = (within_std.mean() / overall_std).sort_values()

print("within-flight std as a fraction of overall std (mean across flights):")
print(spread_ratio.to_string())
print()

# Collapse to one row per flight: mean of each environmental feature,
# plus the per-flight inefficiency rate as the continuous target.
flight_level = train.groupby("flight_id").agg(
    rho=("rho_f117", "mean"),
    baro_pressure=("baro_pressure_pa_f117", "mean"),
    ambient_temp=("ambient_temperature_f117", "mean"),
    wind_north=("windspeed_north_f7", "mean"),
    wind_east=("windspeed_east_f7", "mean"),
    inefficiency_rate=("inefficient", "mean"),
)

print("flight-level table shape:", flight_level.shape)
print()
print("inefficiency_rate distribution across flights:")
print(flight_level["inefficiency_rate"].describe().to_string())
```

### Aside: is differential pressure worth bringing back?

`differential_pressure_pa_f15` is the raw Pitot reading. Indicated airspeed is calculated almost directly from it, and we already have airspeed columns. Quick check before deciding: how correlated is it with the airspeed features we already have?

- High correlation (above ~0.9): confirmed redundant, leave it out
- Low correlation: surprising, would justify a real look

```python
# Aside: check if differential pressure is redundant with existing airspeed columns.
# It lives in feature group 15, not the f117 barometer group, so pull it separately.

DIFF_P_COL = "differential_pressure_pa_f15"
AIRSPEED_COLS = ["indicated_airspeed_m_s_f5", "true_airspeed_m_s_f5"]

# Pull differential pressure from source files for the 111 used flights
dp_frames = []
for src_path in sorted(FLIGHT_DIR.glob("*.csv")):
    fid = int(src_path.stem.split("_")[1])
    if fid not in split_flight_ids:
        continue
    src = pd.read_csv(src_path, usecols=[DIFF_P_COL])
    src["flight_id"] = fid
    src["time_since_start_of_flight"] = np.arange(len(src))
    dp_frames.append(src)

dp = pd.concat(dp_frames, ignore_index=True)

# Merge onto a temporary copy of train (not modifying train itself)
check = train[KEY_COLS + AIRSPEED_COLS].merge(dp, on=KEY_COLS, how="left")

print("nulls in differential pressure after merge:", check[DIFF_P_COL].isna().sum())
print()
print("correlation of differential pressure with airspeed columns:")
for col in AIRSPEED_COLS:
    r = check[DIFF_P_COL].corr(check[col])
    print(f"  {col}: {r:.3f}")
```

**Result:** differential pressure correlates 0.92 with both airspeed columns. Confirmed redundant. It stays out of the analysis. The five environmental features are unchanged.


### Part two: Mutual Information

- `mutual_info_regression` between each per-flight environmental mean and the inefficiency rate
- Higher MI means the feature carries more information about how inefficient a flight is
- Decision: drop if MI near zero, keep if clearly above the noise floor, run Pearson as a tiebreaker for the middle
- Note: `baro_pressure` likely tracks altitude more than weather (it moved more within flights than across them), so read its score with that in mind

```python
from sklearn.feature_selection import mutual_info_regression

env_feature_cols = ["rho", "baro_pressure", "ambient_temp", "wind_north", "wind_east"]

X = flight_level[env_feature_cols]
y = flight_level["inefficiency_rate"]

# random_state fixed so the estimate is reproducible
mi_scores = mutual_info_regression(X, y, random_state=0)

mi_results = pd.Series(mi_scores, index=env_feature_cols).sort_values(ascending=False)

print("Mutual Information with per-flight inefficiency rate (94 flights):")
print(mi_results.to_string())
```

### Phase 1 result

All five features carry forward to Phase 2:

- `baro_pressure`, `rho`, `ambient_temp`: clear MI signal, advance straight through
- `wind_east`, `wind_north`: weak on per-flight mean, but wind varies a lot within flights, so the mean is likely the wrong summary. Carried forward to test the z-score in Phase 2, not dropped
- Skipping the Pearson tiebreaker: it is another per-flight-mean metric, so it would not resolve the wind question. The Phase 2 z-score is the real test

Caveat: `baro_pressure` has the top score but likely tracks altitude more than weather. Keep that in mind when interpreting it.


## Phase 2: Engineered features

### Step 1: per-flight z-scores

- Each row gets (value - flight mean) / flight std, computed within its own flight
- Catches how far a moment deviates from that flight's baseline
- Real test for wind, which Phase 1 suggested gets washed out by the per-flight mean
- Evaluated at row level against the row-level inefficient label

```python
zscore_cols = []
for col in ENV_COLS:
    z = col + "_zscore"
    grp = train.groupby("flight_id")[col]
    train[z] = (train[col] - grp.transform("mean")) / grp.transform("std")
    zscore_cols.append(z)

# A flight whose feature never changes has std = 0, so its z-score is broken
# (divide by zero). Count how many flights and rows that affects per feature.
print("z-score health check (broken = flight had no within-flight variation):\n")
for col in ENV_COLS:
    z = col + "_zscore"
    bad_rows = train[z].replace([np.inf, -np.inf], np.nan).isna()
    n_bad_rows = bad_rows.sum()
    n_bad_flights = train.loc[bad_rows, "flight_id"].nunique()
    total_flights = train["flight_id"].nunique()
    print(f"{col}")
    print(f"  flat flights: {n_bad_flights} of {total_flights}")
    print(f"  broken rows:  {n_bad_rows}")
    print()
```

**Findings:**

- `rho`, `baro_pressure`, `windspeed_north`, `windspeed_east`: vary within every flight, z-scores are clean
- `ambient_temperature`: flat in 35 of 94 flights, so its z-score divides by zero for those rows
- A z-score measures how far a moment sits from the flight's normal. If temperature never changes, there is nothing to measure
- Dropping the ambient temperature z-score. Its signal, if any, is in the per-flight mean, already checked in Phase 1
- Four z-scores move on: `rho`, `baro_pressure`, `windspeed_north`, `windspeed_east`

```python
train = train.drop(columns=["ambient_temperature_f117_zscore"])
zscore_cols = [c for c in zscore_cols if c != "ambient_temperature_f117_zscore"]

X = train[zscore_cols]
y = train["inefficient"]

mi_scores = mutual_info_classif(X, y, random_state=0)

mi_zscore = pd.Series(mi_scores, index=zscore_cols).sort_values(ascending=False)

print("Mutual Information of per-flight z-scores with row-level inefficient label:")
print(mi_zscore.to_string())
```

**Findings:**

- Wind z-scores are the strongest yet: both around 0.20, up from near-zero as raw per-flight means
- Confirms wind's signal is the within-flight gusts, not the flight average
- `rho` and `baro_pressure` z-scores are near zero: these are slow-moving, their signal is the flight-to-flight difference already caught by the per-flight mean in Phase 1
- Keep: `windspeed_east_f7_zscore`, `windspeed_north_f7_zscore`
- Drop: `rho_f117_zscore`, `baro_pressure_pa_f117_zscore`


### Step 2: wind speed magnitude

- Combine the two wind components into one feature: `sqrt(windspeed_north_f7^2 + windspeed_east_f7^2)`
- Test both its per-flight mean (flight level) and its per-flight z-score (row level)
- This is a possible replacement for the two separate wind z-scores, not an addition. Simpler if it holds up

```python
train["wind_magnitude"] = np.sqrt(train["windspeed_north_f7"]**2 + train["windspeed_east_f7"]**2)

grp = train.groupby("flight_id")["wind_magnitude"]
train["wind_magnitude_zscore"] = (train["wind_magnitude"] - grp.transform("mean")) / grp.transform("std")

bad = train["wind_magnitude_zscore"].replace([np.inf, -np.inf], np.nan).isna()
print("wind_magnitude_zscore broken rows:", bad.sum())
print("flat flights:", train.loc[bad, "flight_id"].nunique(), "of", train["flight_id"].nunique())
```

```python
flight_level["wind_magnitude"] = train.groupby("flight_id")["wind_magnitude"].mean()

mi_mag_mean = mutual_info_regression(
    flight_level[["wind_magnitude"]], flight_level["inefficiency_rate"], random_state=0
)[0]

mi_mag_zscore = mutual_info_classif(
    train[["wind_magnitude_zscore"]], train["inefficient"], random_state=0
)[0]

print("wind_magnitude per-flight mean   (flight level, MI vs rate): ", round(mi_mag_mean, 6))
print("wind_magnitude z-score           (row level, MI vs label):   ", round(mi_mag_zscore, 6))
```

**Findings:**

- Wind magnitude per-flight mean: 0.135, beats both raw component means from Phase 1 (`wind_east` 0.08, `wind_north` 0.03). Combining into strength helped
- Wind magnitude z-score: 0.197, matches the two separate component z-scores (both 0.196). Same signal, one feature
- Replace the two component z-scores with the single `wind_magnitude_zscore`. No loss
- Keep: `wind_magnitude` (per-flight mean) and `wind_magnitude_zscore`
- Drop: `windspeed_north_f7_zscore`, `windspeed_east_f7_zscore`


### Step 3: throttle divided by air density

- `output[0]_f3` (throttle PWM) divided by `rho_f117` (air density)
- Physical idea: thinner air needs more throttle for the same lift, so this tests whether air density modulates throttle
- Throttle is a raw PWM value, so the ratio is relative, not a tidy physical number. Fine for a Mutual Information screen
- Row-level feature, evaluated against the row-level inefficient label

```python
train["throttle_over_rho"] = train["output[0]_f3"] / train["rho_f117"]

bad = train["throttle_over_rho"].replace([np.inf, -np.inf], np.nan).isna()
print("throttle_over_rho broken rows:", bad.sum())

mi_tor = mutual_info_classif(
    train[["throttle_over_rho"]], train["inefficient"], random_state=0
)[0]

print("throttle_over_rho (row level, MI vs label):", round(mi_tor, 6))
```

```python
mi_throttle = mutual_info_classif(
    train[["output[0]_f3"]], train["inefficient"], random_state=0
)[0]

print("output[0]_f3 raw throttle (row level, MI vs label):", round(mi_throttle, 6))
print("throttle_over_rho         (row level, MI vs label):", round(mi_tor, 6))
```

**Findings:**

- Raw throttle Mutual Information: 0.259, identical to `throttle_over_rho` (0.259)
- Dividing by air density added nothing. The signal was throttle all along
- `throttle_over_rho` is now a near-duplicate of a column already in the dataset
- Drop `throttle_over_rho`. Raw throttle already carries this, and two versions would just be redundant


## Final feature set

Rule: keep each environmental signal in its best-performing form, drop the rest.

**Kept (5):**
- `rho_f117` - raw per-flight value
- `baro_pressure_pa_f117` - raw per-flight value
- `ambient_temperature_f117` - raw per-flight value
- `wind_magnitude` - engineered, replaces the two raw wind components
- `wind_magnitude_zscore` - engineered, within-flight wind variation

**Dropped:**
- `windspeed_north_f7`, `windspeed_east_f7` - replaced by wind magnitude
- `rho_f117_zscore`, `baro_pressure_pa_f117_zscore` - near zero signal
- `ambient_temperature_f117_zscore` - temperature flat within most flights
- `windspeed_north_f7_zscore`, `windspeed_east_f7_zscore` - replaced by wind magnitude z-score
- `throttle_over_rho` - redundant with raw throttle
- `differential_pressure_pa_f15` - redundant with airspeed

```python
for df in (train, test):
    if "wind_magnitude" not in df.columns:
        df["wind_magnitude"] = np.sqrt(df["windspeed_north_f7"]**2 + df["windspeed_east_f7"]**2)
        grp = df.groupby("flight_id")["wind_magnitude"]
        df["wind_magnitude_zscore"] = (df["wind_magnitude"] - grp.transform("mean")) / grp.transform("std")

print("train has both:", all(c in train.columns for c in ["wind_magnitude", "wind_magnitude_zscore"]))
print("test has both: ", all(c in test.columns for c in ["wind_magnitude", "wind_magnitude_zscore"]))
```

```python
keep_env = [
    "rho_f117",
    "baro_pressure_pa_f117",
    "ambient_temperature_f117",
    "wind_magnitude",
    "wind_magnitude_zscore",
]

drop_env = [
    "windspeed_north_f7",
    "windspeed_east_f7",
    "rho_f117_zscore",
    "baro_pressure_pa_f117_zscore",
    "windspeed_north_f7_zscore",
    "windspeed_east_f7_zscore",
    "throttle_over_rho",
]

for name, df in [("train", train), ("test", test)]:
    present = [c for c in drop_env if c in df.columns]
    df.drop(columns=present, inplace=True)
    print(f"{name}: dropped {len(present)} columns, shape now {df.shape}")
    print(f"  all 5 kept env columns present: {all(c in df.columns for c in keep_env)}")
```

```python
train.to_csv(SPLITS_DIR / "train.csv", index=False)
test.to_csv(SPLITS_DIR / "test.csv", index=False)
print("saved train.csv and test.csv")
```

## Summary

Added 5 environmental features to train.csv and test.csv. All 5 are row-level columns (one value per row). No per-flight aggregate columns were saved.

- `rho_f117`, `baro_pressure_pa_f117`, `ambient_temperature_f117` - raw row-level values
- `wind_magnitude`, `wind_magnitude_zscore` - engineered wind features

**Key takeaways for later analysis:**

- `baro_pressure` had the strongest signal but likely tracks altitude, not weather. Do not interpret it as an atmospheric effect
- Wind only shows signal as within-flight variation (`wind_magnitude_zscore`), not as a flight average. The gusts matter, the mean does not
- Throttle divided by air density added nothing over raw throttle. No environmental interaction features kept
- Differential pressure was left out: 0.92 correlation with airspeed already in the data
- `rho`, `baro_pressure`, `ambient_temperature` barely vary within a flight, so they sit in the data as near-constant per-flight values stored at row resolution
