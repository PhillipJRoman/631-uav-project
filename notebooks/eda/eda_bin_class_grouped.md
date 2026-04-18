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

Using the paper's methodology, we compute the efficiency metric for each row and apply the 0.1 m/W threshold to create the binary label.

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
print(f'\n% inefficient: {df["inefficient"].mean() * 100:.1f}%')
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

**Finding:** The inefficiency rate varies across flights, ranging from 18.9% to 63.0% with a standard deviation of 10%. No flight is entirely one class. 

Visually, earlier flights (roughly 1-30) tend to have higher inefficiency rates (often above the 40.2% average), while later flights (roughly 80-120) tend to be lower. This could reflect changing wind conditions across different flight days, pilot adjustments to the mission, or seasonal differences (the paper notes flights were conducted across different days). This pattern is worth keeping in mind when splitting train/test -- we should stratify or randomize which flights go into each set rather than splitting sequentially, to avoid training on mostly-inefficient flights and testing on mostly-efficient ones.

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
