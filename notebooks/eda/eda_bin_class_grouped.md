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

# EDA: Binary Classification of UAV Energy Efficiency (Pixhawk Grouped)

Exploring the Pixhawk grouped dataset to understand the data for classifying flight segments as efficient or inefficient using the 0.1 m/W threshold from the paper.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RANDOM_SEED = 631

plt.style.use('tableau-colorblind10')
```

## Load All Flights

```python
DATA_DIR = '../../data/processed/pixhawk_grouped_flights/'

all_flights = []
for flight_id in range(1, 121):
    df = pd.read_csv(f'{DATA_DIR}vuelo_{flight_id}_sync.csv')
    df['flight_id'] = flight_id
    all_flights.append(df)

combined = pd.concat(all_flights, ignore_index=True)
print(f'Total flights: {len(all_flights)}')
print(f'Total rows: {len(combined)}')
print(f'Columns: {combined.shape[1]}')
```

```python
print(combined.columns[:20].tolist())
```

```python
# Check if all flights have the same number of columns
col_counts = [len(df.columns) for df in all_flights]
print(f'Min columns: {min(col_counts)}')
print(f'Max columns: {max(col_counts)}')
print(f'Flights with 3003 columns: {col_counts.count(3003)}')
```

```python
key_cols = [
    'voltage_v_f9', 'current_a_f9',
    'vx_f131', 'vy_f131', 'vz_f131',
    'gyro_rad[0]_f104', 'gyro_rad[1]_f104', 'gyro_rad[2]_f104',
    'accelerometer_m_s2[0]_f104', 'accelerometer_m_s2[1]_f104', 'accelerometer_m_s2[2]_f104',
    'indicated_airspeed_m_s_f5', 'true_airspeed_m_s_f5',
    'q[0]_f119', 'q[1]_f119', 'q[2]_f119', 'q[3]_f119',
    'output[0]_f3', 'output[1]_f3', 'output[2]_f3', 'output[3]_f3',
]

for col in key_cols:
    nulls = combined[col].isnull().sum()
    print(f'{col}: {nulls} nulls out of {len(combined)}')
```

## Variable Reference

**Target variable (used to create the label):**

| Column | What it means |
|--------|--------------|
| `voltage_v_f9` | How much juice the battery has (volts) |
| `current_a_f9` | How much power the battery is delivering right now (amps) |
| `vx_f131` | How fast the drone is moving north/south (m/s) |
| `vy_f131` | How fast the drone is moving east/west (m/s) |
| `vz_f131` | How fast the drone is moving up/down (m/s) |

**Predictor variables (model inputs):**

| Column | What it means |
|--------|--------------|
| `gyro_rad[0]_f104` | How fast the drone is rotating side to side (roll rate) |
| `gyro_rad[1]_f104` | How fast the drone is tilting nose up/down (pitch rate) |
| `gyro_rad[2]_f104` | How fast the drone is spinning left/right (yaw rate) |
| `accelerometer_m_s2[0]_f104` | Force pushing the drone forward/backward |
| `accelerometer_m_s2[1]_f104` | Force pushing the drone left/right |
| `accelerometer_m_s2[2]_f104` | Force pushing the drone up/down (includes gravity) |
| `indicated_airspeed_m_s_f5` | How fast air is flowing past the drone (raw Pitot reading) |
| `true_airspeed_m_s_f5` | Airspeed corrected for altitude and temperature |
| `q[0]_f119` | Orientation quaternion w (how the drone is tilted, part 1 of 4) |
| `q[1]_f119` | Orientation quaternion x (part 2 of 4) |
| `q[2]_f119` | Orientation quaternion y (part 3 of 4) |
| `q[3]_f119` | Orientation quaternion z (part 4 of 4) |
| `output[0]_f3` | Throttle command sent to the motor |
| `output[1]_f3` | Command sent to the ailerons (roll control) |
| `output[2]_f3` | Command sent to the elevator (pitch control) |
| `output[3]_f3` | Command sent to the rudder (yaw control) |

```python
# Which flights have nulls in vx_f131?
for df in all_flights:
    fid = df['flight_id'].iloc[0]
    if 'vx_f131' not in df.columns or df['vx_f131'].isnull().any():
        print(f'Flight {fid}: {len(df)} rows, vx_f131 exists: {"vx_f131" in df.columns}')
```

**Finding:** 8 flights have issues with the `vx_f131` (vehicle_local_position) column:

- **Flights 7, 8, 9:** Missing f131 columns entirely. These flights have a different column structure than the rest (~557-590 rows each, so they're full flights, just logged differently).
- **Flight 113:** Only 2 rows. Not a real flight, likely a startup or abort.
- **Flights 117-120:** Missing f131 and also have uncalibrated Pitot data (already flagged in the dataset README).

All 8 flights need to be excluded, leaving 112 usable flights out of 120.

```python
for df in all_flights:
    fid = df['flight_id'].iloc[0]
    if len(df) < 100 or 'vx_f131' not in df.columns:
        print(f'Flight {fid}: {len(df)} rows | f131: {"vx_f131" in df.columns} | f104: {"gyro_rad[0]_f104" in df.columns} | f119: {"q[0]_f119" in df.columns}')
```

**Finding:** 9 flights have issues:

- **Flights 7, 8, 9:** Missing f131, f104, and f119 entirely. Full-length flights (~557-590 rows) but logged with a different column structure.
- **Flight 16:** Only 14 rows. Likely a startup or abort.
- **Flight 113:** Only 2 rows. Same as above.
- **Flights 117-120:** Missing f131, f104, f119 and have uncalibrated Pitot data (flagged in dataset README).

All 9 flights need to be excluded, leaving 111 usable flights out of 120.


**Initial Missing Value Conclusion:** 111 usable flights remain after excluding 9 problematic ones. At ~500 rows per flight, that gives us roughly 55,000 data points with an approximately 50/50 class split. This is sufficient for all four model tiers in the DOE.

**Updated train/test plan:**
- 85/15 split by flight: 94 training, 17 testing
- 6-fold CV: ~15-16 flights per fold

**Flights to exclude:** 7, 8, 9, 16, 113, 117, 118, 119, 120


## Compute Target Variable (Energy Efficiency Label)

Using the paper's methodology, we compute the efficiency metric for each row (time step) and apply the 0.1 m/W threshold to create the binary label.

```python
# Filter to usable flights only
exclude = [7, 8, 9, 16, 113, 117, 118, 119, 120]
df = combined[~combined['flight_id'].isin(exclude)].copy()
print(f'Usable flights: {df["flight_id"].nunique()}')
print(f'Usable rows: {len(df)}')

# Compute power, ground speed, efficiency, and label
df['power_w'] = df['voltage_v_f9'] * df['current_a_f9']
df['ground_speed_ms'] = np.sqrt(df['vx_f131']**2 + df['vy_f131']**2 + df['vz_f131']**2)
df['efficiency_mw'] = df['ground_speed_ms'] / df['power_w']
df['inefficient'] = (df['efficiency_mw'] < 0.1).astype(int)

print(f'\nClass distribution:')
print(df['inefficient'].value_counts())
print(f'\nRow (time step) % inefficient: {df["inefficient"].mean() * 100:.1f}%')
```

**Finding:** Across all 111 usable flights, 40.2% of rows are labeled inefficient (22,811 out of 56,766). That's not a perfect 50/50 like our 3-sample EDA suggested, but it's still a healthy split -- no severe class imbalance. A 60/40 ratio doesn't require special handling like oversampling or class weighting, though we may want to revisit this during modeling.


## Class Distribution by Flight

Let's check if the 60/40 split is consistent across flights, or if some flights are mostly efficient and others mostly inefficient.

```python
flight_pcts = df.groupby('flight_id')['inefficient'].mean() * 100

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(flight_pcts.index, flight_pcts.values)
ax.axhline(y=40.2, color='red', linestyle='--', label='Overall avg (40.2%)')
ax.set_xlabel('Flight ID')
ax.set_ylabel('% Inefficient')
ax.set_title('Percentage of Inefficient Rows per Flight')
ax.legend()
plt.tight_layout()
plt.show()

print(f'Min: {flight_pcts.min():.1f}%')
print(f'Max: {flight_pcts.max():.1f}%')
print(f'Std: {flight_pcts.std():.1f}%')
```

**Clarification:** The model will predict whether each individual moment (row) is efficient or inefficient. It does not predict a flight-level percentage. The flight-level percentages below are only used to understand the data and to make sure our train/test split is balanced.

**Finding:** Each flight has a different mix of efficient and inefficient moments, ranging from 18.9% to 63.0% inefficient with a standard deviation of 10%. No flight is entirely one class.

Earlier flights (roughly 1-30) tend to have more inefficient moments, while later flights (roughly 80-120) tend to have fewer. This could reflect changing wind conditions, pilot adjustments, or seasonal differences (the paper notes flights were conducted across different days). Because of this pattern, we need to stratify our train/test split rather than splitting sequentially -- otherwise we'd train on mostly-inefficient flights and test on mostly-efficient ones.

```python
bins = pd.cut(flight_pcts, bins=[0, 35, 45, 100], labels=['low (<35%)', 'medium (35-45%)', 'high (>45%)'])
print(bins.value_counts())
```

**Finding:** Adjusting to 3 bins with a 35% cutoff:

| Bin | Range | Flights |
|-----|-------|---------|
| Low | under 35% inefficient | 36 |
| Medium | 35-45% inefficient | 39 |
| High | above 45% inefficient | 36 |

Nearly equal distribution across bins. This is the stratification we should use for train/test splitting and CV folds.

```python
# Get flight-level inefficiency bins for stratification
flight_pcts = df.groupby('flight_id')['inefficient'].mean() * 100
flight_bins = pd.cut(flight_pcts, bins=[0, 35, 45, 100], labels=['low', 'medium', 'high'])

# Split flight IDs, stratified by inefficiency bin
flight_ids = flight_bins.index.values
flight_labels = flight_bins.values

train_ids, test_ids = train_test_split(
    flight_ids, test_size=0.15, random_state=RANDOM_SEED, stratify=flight_labels
)

# Split the data
train_df = df[df['flight_id'].isin(train_ids)].copy()
test_df = df[df['flight_id'].isin(test_ids)].copy()

print(f'Train: {len(train_ids)} flights, {len(train_df)} rows')
print(f'Test: {len(test_ids)} flights, {len(test_df)} rows')
print(f'\nTrain % inefficient: {train_df["inefficient"].mean() * 100:.1f}%')
print(f'Test % inefficient: {test_df["inefficient"].mean() * 100:.1f}%')
```

```python
train_bins = flight_bins[train_ids].value_counts()
test_bins = flight_bins[test_ids].value_counts()

print('Train bins:')
print(train_bins)
print(f'\nTest bins:')
print(test_bins)
```

**Finding:** Train/test split is well-balanced:

| | Train | Test |
|---|---|---|
| Flights | 94 | 17 |
| Rows | 47,957 | 8,809 |
| % Inefficient | 40.4% | 39.2% |
| Low bin | 30 | 6 |
| Medium bin | 33 | 6 |
| High bin | 31 | 5 |

Stratification preserved the distribution of flight behaviors across both sets. Now let's save these splits.

```python
train_df.to_csv('../../data/splits/train.csv', index=False)
test_df.to_csv('../../data/splits/test.csv', index=False)

print(f'Train saved: {len(train_df)} rows')
print(f'Test saved: {len(test_df)} rows')
```

Quick check of testset for nulls in predictor variables. 

```python
test = pd.read_csv('../../data/splits/test.csv')
print('Test set nulls:')
print(test[predictor_cols].isnull().sum().to_string())
```

## Explore Predictor Variables (Training Set Only)

From this point forward, all EDA is done on the training set to avoid leaking information from the test set into our analysis and modeling decisions.

```python
# Load training set
train = pd.read_csv('../../data/splits/train.csv')

# Define predictor columns
predictor_cols = [
    'gyro_rad[0]_f104', 'gyro_rad[1]_f104', 'gyro_rad[2]_f104',
    'accelerometer_m_s2[0]_f104', 'accelerometer_m_s2[1]_f104', 'accelerometer_m_s2[2]_f104',
    'indicated_airspeed_m_s_f5', 'true_airspeed_m_s_f5',
    'q[0]_f119', 'q[1]_f119', 'q[2]_f119', 'q[3]_f119',
    'output[0]_f3', 'output[1]_f3', 'output[2]_f3', 'output[3]_f3',
]

print(f'Training rows: {len(train)}')
print(f'Predictor columns: {len(predictor_cols)}')
print(f'\nBasic stats:')
train[predictor_cols].describe().round(3)
```

**Finding:** All 16 predictor columns have zero nulls in the training set (47,957 rows each). Key observations:

- **Gyro rates** (rotation): Centered near zero, symmetric range. Makes sense -- the drone isn't constantly spinning in one direction.
- **Accelerometer:** The z-axis (up/down) averages around -10.3 m/s2, which assume is gravity. The x and y axes fluctuate around zero. This is expected for level flight.
- **Airspeed:** Averages ~24 m/s with a range of 13-38 m/s. The paper states cruise speed is 14-17 m/s, so these values seem high -- worth investigating.
- **Quaternions:** A way to describe which direction the drone is pointing using four numbers instead of the more familiar roll/pitch/yaw angles. Values range between -1 and 1. In our data, q[0] (w) and q[3] (z) have wide ranges, meaning the drone's heading (which compass direction it faces) varies a lot -- expected since it's flying a circuit and constantly turning. q[1] (x) and q[2] (y) are tighter, meaning the drone doesn't tilt or pitch very aggressively -- also expected for a stable fixed-wing aircraft in cruise.
- **Actuator outputs:** All in the 1000-1860 Pulse Width Modulation (PWM) range. Throttle (output[0]) has the widest spread. Control surfaces (output[1-3]) are tighter, centered around their midpoints. Pulse Width Modulation is how the flight controller talks to the motor and control surfaces (ailerons, elevator, rudder). The number (like 1000-1860) represents the width of an electrical pulse in microseconds. Higher number = more deflection or more throttle. 1000 is typically the minimum (off/neutral) and 2000 is the maximum (full throttle/full deflection).

The scales are very different across features (gyro in radians vs PWM in 1000s). Scaling will be necessary before modeling, which aligns with the DOE's "Raw + Scaling" baseline.


## Variable Reference

**Target variable (used to create the label):**

| Column | What it means |
|--------|--------------|
| `voltage_v_f9` | How much juice the battery has (volts) |
| `current_a_f9` | How much power the battery is delivering right now (amps) |
| `vx_f131` | How fast the drone is moving north/south (m/s) |
| `vy_f131` | How fast the drone is moving east/west (m/s) |
| `vz_f131` | How fast the drone is moving up/down (m/s) |

**Predictor variables (model inputs):**

| Column | What it means |
|--------|--------------|
| `gyro_rad[0]_f104` | How fast the drone is rotating side to side (roll rate) |
| `gyro_rad[1]_f104` | How fast the drone is tilting nose up/down (pitch rate) |
| `gyro_rad[2]_f104` | How fast the drone is spinning left/right (yaw rate) |
| `accelerometer_m_s2[0]_f104` | Force pushing the drone forward/backward |
| `accelerometer_m_s2[1]_f104` | Force pushing the drone left/right |
| `accelerometer_m_s2[2]_f104` | Force pushing the drone up/down (includes gravity) |
| `indicated_airspeed_m_s_f5` | How fast air is flowing past the drone (raw Pitot reading) |
| `true_airspeed_m_s_f5` | Airspeed corrected for altitude and temperature |
| `q[0]_f119` | Orientation quaternion w (how the drone is tilted, part 1 of 4) |
| `q[1]_f119` | Orientation quaternion x (part 2 of 4) |
| `q[2]_f119` | Orientation quaternion y (part 3 of 4) |
| `q[3]_f119` | Orientation quaternion z (part 4 of 4) |
| `output[0]_f3` | Throttle command sent to the motor |
| `output[1]_f3` | Command sent to the ailerons (roll control) |
| `output[2]_f3` | Command sent to the elevator (pitch control) |
| `output[3]_f3` | Command sent to the rudder (yaw control) |

```python
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(predictor_cols):
    axes[i].hist(train[col], bins=50, edgecolor='none')
    axes[i].set_title(col.replace('_f104', '').replace('_f5', '').replace('_f119', '').replace('_f3', ''), fontsize=9)

plt.suptitle('Predictor Variable Distributions (Training Set)', y=1.01)
plt.tight_layout()
plt.show()
```

**Finding:** Key observations from the distributions:

- **Gyro rates:** All three centered near zero with roughly bell-shaped distributions. gyro_rad[0] (roll) has the widest spread, gyro_rad[1] (pitch) shows a moderate right skew, and gyro_rad[2] (yaw) is the tightest.
- **Accelerometer:** x and y axes are roughly normal around zero. The z-axis is centered around -10 m/s2 (gravity) with a slight left skew. All look clean.
- **Airspeed:** Both indicated and true airspeed show a slight left skew with a peak around 24-26 m/s. There's a long left tail down to ~13 m/s, which could be takeoff/landing or low-speed turns.
- **Quaternions:** q[0] (w) and q[3] (z) show bimodal or flat distributions -- the drone's heading takes on many different values as it flies the circuit. q[1] (x) and q[2] (y) are tightly peaked near zero -- limited roll and pitch, as expected with fixed wing aircraft.
- **Actuator outputs:** output[0] (throttle) is left-skewed with a peak between 1600-1700, meaning the drone spends most of its time at high throttle during cruise. output[1] (ailerons) is roughly normal around its midpoint. output[2] (elevator) is roughly normal with a tight range. output[3] (rudder) is bimodal with peaks near 1400 and 1600, possibly reflecting two distinct flight behaviors like straight cruise vs turning.

Nothing here requires transformation for tree-based models (XGBoost). Logistic Regression will need scaling due to the very different ranges across features (gyro in radians vs actuators in 1000s PWM).


## Predictor vs Label Relationship

```python
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(predictor_cols):
    efficient = train[train['inefficient'] == 0][col]
    inefficient = train[train['inefficient'] == 1][col]
    axes[i].hist(efficient, bins=50, alpha=0.5, label='Efficient', edgecolor='none')
    axes[i].hist(inefficient, bins=50, alpha=0.5, label='Inefficient', edgecolor='none')
    axes[i].set_title(col.replace('_f104', '').replace('_f5', '').replace('_f119', '').replace('_f3', ''), fontsize=9)

axes[0].legend(fontsize=8)
plt.suptitle('Predictor Distributions by Class (Training Set)', y=1.01)
plt.tight_layout()
plt.show()
```

**How to read these histograms:**

Each plot shows the same feature split by class: blue is efficient, orange is inefficient. A few things to keep in mind when interpreting:

- **Orange inside blue:** The inefficient moments look the same as efficient moments for that feature. On its own, this feature can't help you tell the two classes apart.
- **Orange shifted or outside blue:** The inefficient moments behave differently. There's a visible shift in position or spread. This feature could help a model separate the classes.
- **Same shape for both:** The feature follows the same pattern regardless of class. Even with a similar shape, a shift in where the distribution sits (different center) can still be useful.
- **Height difference:** Since inefficient rows make up only 40% of the data, the orange bars will always be shorter than the blue. This doesn't mean the feature is a separator -- it just means there are fewer orange data points. For this analysis, we focus on the difference of distribution shapes, not the difference in heights.

Based on these plots, `accelerometer_m_s2[0]`, `q[0]`, `output[0]`, and `output[2]` show the most visible indicators as possible separators -- their orange distributions are either shifted, wider, or shaped differently from the blue.

The remaining features show heavy overlap between classes, meaning they may not be strong separators individually. However, that doesn't make them useless -- combinations of weak features can still separate classes when used together, which is part of what the models (especially the GNN) will test.


## Environmental Predictor Variables

The paper notes that environmental variables such as temperature, pressure, and humidity can impact power demand. Let's check what's available in the dataset.

```python
env_cols = [c for c in train.columns if any(x in c.lower() for x in ['temperature', 'pressure', 'humidity', 'rho_'])]
for c in env_cols:
    nulls = train[c].isnull().sum()
    print(f'{c}: {nulls} nulls')
```

```python
env_predictor_cols = [
    'ambient_temperature_f117',
    'baro_pressure_pa_f117',
    'rho_f117',
    'differential_pressure_pa_f15',
]

predictor_cols = predictor_cols + env_predictor_cols
print(f'Updated predictor count: {len(predictor_cols)}')
print(f'\nNew columns stats:')
train[env_predictor_cols].describe().round(3)
```

**Finding:** Four environmental columns added to the predictor list.

| Column | What it means | Range |
|--------|--------------|-------|
| `ambient_temperature_f117` | Air temperature outside the drone (Celsius) | 30.9 - 43.4 |
| `baro_pressure_pa_f117` | Atmospheric pressure (Pascals) | 101,955 - 103,498 |
| `rho_f117` | Air density (kg/m3) -- how thick the air is, affects drag and lift | 1.126 - 1.183 |
| `differential_pressure_pa_f15` | Pressure difference measured by the Pitot tube (Pascals) -- directly related to airspeed | 93 - 830 |

Temperature varies between 31-43C, consistent with flights in eastern Spain across different days. Air density has a very tight range (std of 0.009), so it may not add much predictive value on its own. Differential pressure has a wide range and is physically tied to airspeed, so it may overlap with our existing airspeed columns -- worth checking during correlation analysis.

```python
wind_cols = [c for c in train.columns if 'wind' in c.lower()]
for c in wind_cols:
    nulls = train[c].isnull().sum()
    if nulls == 0:
        print(f'{c}: {nulls} nulls')
```

```python
wind_cols = [
    'windspeed_north_f7',
    'windspeed_east_f7',
]

predictor_cols = predictor_cols + wind_cols
print(f'Updated predictor count: {len(predictor_cols)}')
print(f'\nWind columns stats:')
train[wind_cols].describe().round(3)
```

**Finding:** Two wind columns added to the predictor list (now 22 total).

| Column | What it means | Range |
|--------|--------------|-------|
| `windspeed_north_f7` | Wind speed in the north/south direction (m/s). Positive = wind blowing north. | -3.8 to 6.7 |
| `windspeed_east_f7` | Wind speed in the east/west direction (m/s). Positive = wind blowing east. | -6.1 to 2.3 |

The east wind averages -2.5 m/s, meaning wind consistently blows from the east during these flights. Wind varies enough across the dataset to potentially matter for efficiency prediction -- headwinds and crosswinds directly affect how hard the drone works to maintain ground speed.


## Summary: Environmental Predictors Added

The paper states that environmental conditions affect power demand. We added 6 columns to capture this (22 predictors total):

- **Temperature, pressure, air density** -- the air the drone flies through affects drag and lift. Thinner, hotter air changes how hard the motor works.
- **Differential pressure** -- the raw Pitot tube reading that airspeed is derived from. Included in case it captures something the processed airspeed columns miss.
- **Wind speed (north and east)** -- headwinds and crosswinds directly affect how hard the drone works to maintain ground speed. A drone flying into a headwind burns more power for less progress.


## Environmental Predictor Distributions by Class

```python
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

env_all = env_predictor_cols + wind_cols

for i, col in enumerate(env_all):
    efficient = train[train['inefficient'] == 0][col]
    inefficient = train[train['inefficient'] == 1][col]
    axes[i].hist(efficient, bins=50, alpha=0.5, label='Efficient', edgecolor='none')
    axes[i].hist(inefficient, bins=50, alpha=0.5, label='Inefficient', edgecolor='none')
    axes[i].set_title(col.replace('_f117', '').replace('_f15', '').replace('_f7', ''), fontsize=9)

axes[0].legend(fontsize=8)
plt.suptitle('Environmental Predictor Distributions by Class (Training Set)', y=1.01)
plt.tight_layout()
plt.show()
```

**Finding:** How each environmental predictor separates efficient (blue) from inefficient (orange):

- **ambient_temperature:** Multimodal distribution with distinct spikes at ~34C and ~38C. Both classes follow the same pattern. These spikes likely represent different flight days with different weather. Not a separator -- temperature doesn't change within a flight, so it can't explain moment-to-moment efficiency changes.
- **baro_pressure_pa:** Similar story -- multimodal with peaks reflecting different flight days. Both classes overlap almost entirely. Not a separator on its own.
- **rho (air density):** Same flight-day clustering. Very tight range overall. Heavy overlap between classes. Unlikely to separate.
- **differential_pressure_pa:** Both classes follow a similar shape with heavy overlap. Slight left skew for inefficient moments. Weak separator at best, and likely correlated with the airspeed columns we already have.
- **windspeed_north:** Heavy overlap. Inefficient moments may shift slightly right (more northerly wind) but the difference is small.
- **windspeed_east:** Heavy overlap. Both classes follow the same distribution shape. Not a clear separator.

**Key takeaway:** The environmental features don't show strong individual class separation. The temperature, pressure, and air density features reflect flight-day conditions rather than moment-to-moment changes, which makes sense -- the weather doesn't change mid-flight.

However, these features may still be valuable in a different way: rather than as standalone predictors, they could be used to normalize flight dynamics features. For example, the same throttle setting produces different thrust in thin hot air vs thick cool air. Adjusting features like throttle or airspeed by air density would capture what the drone is actually doing rather than what the instruments read. This kind of normalization is a potential feature engineering step to explore after the baseline models are running.


## Predictor Correlations

```python
corr = train[predictor_cols + ['inefficient']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
masked_corr = corr.copy()
masked_corr[mask] = np.nan

labels = [c.replace('_f104', '').replace('_f5', '').replace('_f119', '').replace('_f3', '').replace('_f117', '').replace('_f15', '').replace('_f7', '') for c in predictor_cols + ['inefficient']]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(masked_corr, cmap='cividis', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
plt.colorbar(im, shrink=0.8)
plt.title('Correlation Matrix: Predictors + Target (Training Set)')
plt.tight_layout()
plt.show()
```

```python
# Correlation with the target variable
target_corr = corr['inefficient'].drop('inefficient').sort_values(ascending=False)
print('Correlation with inefficient label:\n')
for col, val in target_corr.items():
    name = col.replace('_f104', '').replace('_f5', '').replace('_f119', '').replace('_f3', '').replace('_f117', '').replace('_f15', '').replace('_f7', '')
    print(f'  {name:<30} {val:.3f}')
```

**Finding:** Correlation of each predictor with the inefficient label:

**Positively correlated (higher value = more likely inefficient):**
- **output[0] (throttle): 0.572** -- strongest correlation by far. Higher throttle = more likely inefficient.
- **accelerometer_m_s2[0] (forward force): 0.488** -- second strongest. Higher forward acceleration = more likely inefficient.
- **output[2] (elevator): 0.241** -- moderate. Higher elevator deflection = more likely inefficient. Could reflect climbing.

**Weakly positive (under 0.1):**
- ambient_temperature, windspeed_north, output[1] (ailerons), gyro_rad[1] (pitch rate) -- all under 0.09. Present but unlikely to carry much weight individually.

**Near zero (no linear relationship):**
- gyro_rad[2], q[3], output[3] (rudder), q[1], q[2], gyro_rad[0] -- essentially no linear correlation with efficiency.

**Negatively correlated (higher value = more likely efficient):**
- **baro_pressure_pa: -0.126** and **rho (air density): -0.116** -- modest. Higher pressure/denser air = slightly more efficient. Makes physical sense: denser air provides more lift.
- indicated_airspeed, differential_pressure, true_airspeed -- all around -0.06. Slightly higher airspeed = slightly more efficient.
- windspeed_east: -0.07 -- weak.

**Key takeaway:** Throttle and forward acceleration dominate the linear relationship with efficiency. Most other features have weak or no individual linear correlation. This sets expectations for the DOE: Logistic Regression will likely lean heavily on these two features. XGBoost and GNN may find non-linear combinations among the weaker features that Logistic Regression misses.

```python
# Find predictor pairs with high correlation (above 0.7)
predictor_corr = corr.loc[predictor_cols, predictor_cols]

pairs = []
for i in range(len(predictor_cols)):
    for j in range(i+1, len(predictor_cols)):
        val = predictor_corr.iloc[i, j]
        if abs(val) > 0.7:
            n1 = predictor_cols[i].replace('_f104', '').replace('_f5', '').replace('_f119', '').replace('_f3', '').replace('_f117', '').replace('_f15', '').replace('_f7', '')
            n2 = predictor_cols[j].replace('_f104', '').replace('_f5', '').replace('_f119', '').replace('_f3', '').replace('_f117', '').replace('_f15', '').replace('_f7', '')
            pairs.append((n1, n2, val))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print('Highly correlated predictor pairs (|r| > 0.7):\n')
for n1, n2, val in pairs:
    print(f'  {n1:<30} {n2:<30} {val:.3f}')
```

**Finding:** Six pairs of predictors are highly correlated (|r| > 0.7):

- **indicated_airspeed and true_airspeed (0.999):** Almost identical. We should drop one -- they're measuring the same thing.
- **true/indicated airspeed and differential_pressure (0.916):** Differential pressure is the raw Pitot reading that airspeed is derived from. Three features measuring the same physical quantity. We likely only need one of the three.
- **ambient_temperature and rho (-0.934):** Hotter air is less dense. Physically linked. Keeping both is redundant.
- **baro_pressure and rho (0.738):** Pressure and density are also physically linked. Combined with the temperature-rho pair, all three environmental features (temperature, pressure, density) are describing the same underlying condition.
- **gyro_rad[0] (roll rate) and output[1] (ailerons) (-0.762):** Ailerons directly cause rolling. The negative sign makes sense -- aileron deflection in one direction causes roll in the other. These are correlated but describe cause (aileron command) and effect (roll response), so both may still be valuable.

**Implications for modeling:**
- For Logistic Regression: highly correlated features cause instability. We should consider dropping redundant ones or using Ridge regularization.
- For XGBoost: handles correlated features fine, but redundant features add noise without adding information.
- For the GNN: correlated features between connected sensor nodes could actually help -- it validates that the graph edges represent real physical relationships.

**Team discussion point:** Which features to keep from each redundant group (e.g., keep indicated_airspeed and drop the other two? keep rho and drop temperature and pressure?).


## Time Series Behavior

How does efficiency change over the course of a flight? If transitions between efficient and inefficient are gradual, rolling averages will help the models. If they're sudden, frame stacking matters more. This directly informs the temporal engineering decisions in the DOE.

```python
# Pick 3 sample flights from training set to visualize
sample_ids = train['flight_id'].unique()[[10, 45, 80]]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, fid in enumerate(sample_ids):
    flight = train[train['flight_id'] == fid]
    axes[i].plot(flight['inefficient'].values, linewidth=0.8)
    axes[i].set_title(f'Flight {fid} -- Efficiency Label Over Time')
    axes[i].set_ylabel('0=Efficient, 1=Inefficient')
    axes[i].set_ylim(-0.1, 1.1)

axes[-1].set_xlabel('Row number')
plt.tight_layout()
plt.show()
```

**Finding:** The efficiency label switches between 0 and 1 sharply -- there are no gradual transitions. The drone is either efficient or inefficient at any given moment, with instant flips between the two states.

Across all three sample flights, there's a repeating pattern: blocks of inefficient moments (sustained at 1) followed by blocks of efficient moments (sustained at 0). This is consistent with the racetrack circuit -- the drone is likely inefficient during turns and climbs, then efficient during straight cruise segments. The blocks vary in length but the pattern is cyclical, reflecting the repeating laps.

**Implications for temporal engineering (DOE):**
- **Rolling averages:** Could smooth out the very short spikes (1-2 row blips) and help the model see the broader trend of "entering an inefficient segment."
- **Frame stacking:** More relevant here. Since transitions are sudden, the model needs to see the exact sequence of sensor readings leading up to a flip. A few rows of history could capture the moment the drone enters a turn or starts climbing.
- Both approaches are worth testing, as outlined in the DOE.
