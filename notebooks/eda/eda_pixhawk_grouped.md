---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: my-experiment
    language: python
    name: python3
---

# Pixhawk Grouped EDA

Exploring three Pixhawk grouped flights (11, 56, 101) to understand the data structure, quality, and viability for both regression (GPS-denied position) and classification (energy efficiency).

```python
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
```

## Load Three Flights (#11, #56, and #101)

```python
DATA_DIR = '../../data/processed/pixhawk_grouped_flights/'

flights = {
    11: pd.read_csv(f'{DATA_DIR}vuelo_11_sync.csv'),
    56: pd.read_csv(f'{DATA_DIR}vuelo_56_sync.csv'),
    101: pd.read_csv(f'{DATA_DIR}vuelo_101_sync.csv'),
}

for fid, df in flights.items():
    print(f'Flight {fid}: {df.shape[0]} rows, {df.shape[1]} columns')
```

## Column Overview

```python
cols_match = (flights[11].columns.tolist() == flights[56].columns.tolist() == flights[101].columns.tolist())
print(f'All three flights have identical columns: {cols_match}')
print(f'Total columns of each flight: {flights[11].shape[1]}')
```

```python
flights[11].dtypes.value_counts()
```

## Missing Values

```python
for fid, df in flights.items():
    null_pct = (df.isnull().sum() / len(df) * 100)
    cols_with_nulls = null_pct[null_pct > 0]
    print(f'Flight {fid}: {len(cols_with_nulls)} columns with nulls out of {df.shape[1]}')
    if len(cols_with_nulls) > 0:
        print(cols_with_nulls.describe())
    print()
```

## Timestamp Structure

```python
flights[11]['timestamp'].head()
```

```python
print(f'Estimation of flight durations, time segments, and frequency\n')

for fid, df in flights.items():
    assumed_duration = 360  # ~6 minutes per flight from paper
    t_per_row = assumed_duration / len(df)
    print(f'Flight {fid}:')
    print(f'  Rows: {len(df)}')
    print(f'  Assumed duration: {assumed_duration}s ({assumed_duration/60:.0f} min)')
    print(f'  Approx time per row: {t_per_row:.2f}s')
    print(f'  Approx frequency: {1/t_per_row:.2f} Hz')
    print()
```

## Key Columns: Regression Path (GPS-Denied Position)

For regression we need position (target) and sensor inputs (IMU, attitude, airspeed, actuators). Let's find the relevant columns in the grouped data.


The 'grouped' CSV squishes ~130 separate sensor files into one big spreadsheet. Since many of those files have columns with the same name (like x, y, z), each column gets a tag like `_f58` or `_f99` to show which file it originally came from.

We need the drone's position (x, y, z) as the thing we're trying to predict. There are multiple versions of x, y, z in this data because different onboard systems each calculate their own estimate of where the drone is. Below we look for all of them so we can figure out which one to use.

```python
[c for c in flights[11].columns if c.startswith('x_f') or c.startswith('y_f') or c.startswith('z_f')]
```

We have 3003 columns and no idea what most of them are. Each column ends with a tag like `_f9` or `_f58` that tells us which sensor file it originally came from. This groups the columns by these tags to build a table of contents for this dataset.

```python
# Group all columns by their _fXX suffix to see what sensor groups exist
df = flights[11]
suffix_groups = {}

for col in df.columns:
    match = re.search(r'_f(\d+)$', col)
    if match:
        suffix = f'f{match.group(1)}'
        if suffix not in suffix_groups:
            suffix_groups[suffix] = []
        suffix_groups[suffix].append(col)
    else:
        if 'no_suffix' not in suffix_groups:
            suffix_groups['no_suffix'] = []
        suffix_groups['no_suffix'].append(col)

print(f'Total sensor groups: {len(suffix_groups)}\n')
for suffix, cols in sorted(suffix_groups.items(), key=lambda x: len(x[1]), reverse=True):
    preview = ', '.join(cols[:3])
    print(f'{suffix} ({len(cols)} cols): {preview} ...')
```

```python
# Background check  - what columns are in the ungrouped vehicle_local_position file
vlp = pd.read_csv('/Users/phillipromanmacbook/Downloads/16992976/Holybro Pixhawk/processed/UnGrouped flights/lap_001/log_67_2025-8-21-14-24-48_vehicle_local_position_0.csv')
print('vehicle_local_position columns:')
print(vlp.columns.tolist()[:10])
```

```python
# Background check - ungrouped estimator_local_position file
vlp2 = pd.read_csv('/Users/phillipromanmacbook/Downloads/16992976/Holybro Pixhawk/processed/UnGrouped flights/lap_001/log_67_2025-8-21-14-24-48_estimator_local_position_0.csv')
print('estimator_local_position columns:')
print(vlp2.columns.tolist()[:10])
```

```python
# Regression target columns (f131 = vehicle_local_position)
df = flights[11]
target_cols = [c for c in df.columns if c.endswith('_f131')]
print('vehicle_local_position columns (f131):\n')
for c in target_cols:
    print(f'  {c}')
print(f'\n  ({len(target_cols)} total)')
```

**Finding:** Both f58 and f131 contain x, y, z position data with identical column structure (54 columns each). But they're different things:

- **f58** = `estimator_local_position_0` -- an intermediate position estimate from the onboard filter
- **f131** = `vehicle_local_position_0` -- the final position the drone actually used

The paper uses `vehicle_local_position_0` as the regression target. So for our GPS-denied position prediction, **f131 is the one we want, not f58**.


## Exploring Predictor Variables (Regression Path)

Now that we know our target (f131 = the drone's actual position), we need to find the inputs we'd feed into the model. The paper says we need IMU readings, attitude, airspeed, and actuator outputs. Let's see what's available in the grouped data, starting with the IMU.

```python
# Search for IMU / sensor_combined columns
imu_cols = [c for c in df.columns if 'accelerometer' in c.lower() or 'gyro_rad' in c.lower()]
print('IMU columns (sensor_combined):\n')
for c in imu_cols:
    print(f'  {c}')
```

**Finding:** 9 IMU columns, all from f104 (`sensor_combined_0`). Of these, 6 are actual sensor readings:
- 3 gyro values (rotation rates in rad/s): `gyro_rad[0-2]_f104`
- 3 accelerometer values (acceleration in m/s2): `accelerometer_m_s2[0-2]_f104`

The remaining 3 (timestamp_relative, integral_dt, clipping) are sensor metadata, not measurements.

The paper identifies `sensor_combined_0` as one of the required inputs for GPS-denied position estimation (Section: Position Estimation in GPS-Denied Scenarios).

```python
# Search for airspeed columns
airspeed_cols = [c for c in df.columns if 'airspeed' in c.lower()]
print('Airspeed columns:\n')
for c in airspeed_cols:
    print(f'  {c}')
print(f'    ({len(airspeed_cols)} total)')
```

**Finding:** 31 airspeed-related columns spread across multiple sensor groups (f5, f6, f49, f50, etc.). The ones that matter for our regression path:
- `indicated_airspeed_m_s_f5` and `true_airspeed_m_s_f5` -- raw Pitot tube readings from f5 (`airspeed_0`)

The rest are calibrated/validated versions (f6) or estimator innovation values (f49, f50) which we likely don't need as model inputs.

The paper lists `airspeed_0` as one of the required inputs for GPS-denied position estimation (Section: Position Estimation in GPS-Denied Scenarios).


## Regression Path - Initial EDA Summary

**What we're predicting:** The drone's position (x, y, z) in local coordinates.

**Target variables (f131 = `vehicle_local_position_0`):**
- `x_f131`, `y_f131`, `z_f131`

**Predictor variables identified so far:**

| Variable | Columns | Source |
|----------|---------|--------|
| Gyro (rotation rates) | `gyro_rad[0-2]_f104` | sensor_combined_0 |
| Accelerometer | `accelerometer_m_s2[0-2]_f104` | sensor_combined_0 |
| Airspeed | `indicated_airspeed_m_s_f5`, `true_airspeed_m_s_f5` | airspeed_0 |

**Still need to explore:**
- Attitude (drone orientation) -- the paper lists `vehicle_attitude_0` as a required input
- Actuator outputs (control surface positions) -- the paper lists `actuator_outputs_0` as a required input

These are identified in Section: Position Estimation in GPS-Denied Scenarios of the paper.

```python
# Attitude columns (vehicle_attitude)
att_cols = [c for c in df.columns if c.endswith('_f119')]
print('vehicle_attitude columns (f119):\n')
for c in att_cols:
    print(f'  {c}')
print(f'    ({len(att_cols)} total)')
```

**Finding:** 10 columns from f119 (`vehicle_attitude_0`). Of these, 4 are actual measurements:
- `q[0-3]_f119` -- quaternion values describing the drone's orientation (which way it's tilted and rotated)

The remaining 6 are reset tracking metadata, not measurements.

The paper lists `vehicle_attitude_0` as a required input for GPS-denied position estimation (Section: Position Estimation in GPS-Denied Scenarios).

```python
# Actuator outputs columns
act_cols = [c for c in df.columns if c.endswith('_f2')]
print('actuator_outputs_0 columns (f2):\n')
for c in act_cols:
    print(f'  {c}')
print(f'    ({len(act_cols)} total)')
```

```python
# Check which actuator output channels actually have varying data
for fid, df in flights.items():
    print(f'Flight {fid}:')
    for i in range(16):
        col = f'output[{i}]_f2'
        if col in df.columns:
            nunique = df[col].nunique()
            print(f'  {col}: {nunique} unique values, range [{df[col].min()}, {df[col].max()}]')
    print()
```

```python
for fid, df in flights.items():
    print(f'Flight {fid}:')
    for i in range(16):
        col = f'output[{i}]_f3'
        if col in df.columns:
            nunique = df[col].nunique()
            if nunique > 1:
                print(f'  {col}: {nunique} unique values, range [{df[col].min()}, {df[col].max()}]')
    print()
```

**Finding:** f2 (`actuator_outputs_0`) only has 1 active channel (throttle). The actual control surfaces are in f3 (`actuator_outputs_1`):

- `output[0]_f3` -- throttle
- `output[1]_f3` -- ailerons
- `output[2]_f3` -- elevator
- `output[3]_f3` -- rudder

All are PWM signals in the 1000-1800 range. Channels 4-15 are unused (all zeros). For the regression model, we'd use these 4 columns from f3 as predictor variables.


## Regression Path: Complete Variable Map

**Target:**

| Column | Description | Source file | Suffix |
|--------|------------|-------------|--------|
| `x_f131` | Position north (meters) | vehicle_local_position_0 | f131 |
| `y_f131` | Position east (meters) | vehicle_local_position_0 | f131 |
| `z_f131` | Position down (meters) | vehicle_local_position_0 | f131 |

**Predictors:**

| Column | Description | Source file | Suffix |
|--------|------------|-------------|--------|
| `gyro_rad[0]_f104` | Roll rotation rate (rad/s) | sensor_combined_0 | f104 |
| `gyro_rad[1]_f104` | Pitch rotation rate (rad/s) | sensor_combined_0 | f104 |
| `gyro_rad[2]_f104` | Yaw rotation rate (rad/s) | sensor_combined_0 | f104 |
| `accelerometer_m_s2[0]_f104` | Forward acceleration (m/s2) | sensor_combined_0 | f104 |
| `accelerometer_m_s2[1]_f104` | Lateral acceleration (m/s2) | sensor_combined_0 | f104 |
| `accelerometer_m_s2[2]_f104` | Vertical acceleration (m/s2) | sensor_combined_0 | f104 |
| `indicated_airspeed_m_s_f5` | Airspeed from Pitot tube (m/s) | airspeed_0 | f5 |
| `true_airspeed_m_s_f5` | Corrected airspeed (m/s) | airspeed_0 | f5 |
| `q[0]_f119` | Attitude quaternion w | vehicle_attitude_0 | f119 |
| `q[1]_f119` | Attitude quaternion x | vehicle_attitude_0 | f119 |
| `q[2]_f119` | Attitude quaternion y | vehicle_attitude_0 | f119 |
| `q[3]_f119` | Attitude quaternion z | vehicle_attitude_0 | f119 |
| `output[0]_f3` | Throttle (PWM) | actuator_outputs_1 | f3 |
| `output[1]_f3` | Ailerons (PWM) | actuator_outputs_1 | f3 |
| `output[2]_f3` | Elevator (PWM) | actuator_outputs_1 | f3 |
| `output[3]_f3` | Rudder (PWM) | actuator_outputs_1 | f3 |

**Totals:** 3 target columns, 16 predictor columns.

All inputs identified per the paper's Section: Position Estimation in GPS-Denied Scenarios.


Let's visualize what the regression target (position x, y, z) actually looks like across our three sample flights.

```python
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

pos_labels = ['x_f131', 'y_f131', 'z_f131']

for i, (fid, df) in enumerate(flights.items()):
    for j, col in enumerate(pos_labels):
        if col in df.columns:
            axes[i, j].plot(df[col].values, linewidth=0.8)
            axes[i, j].set_title(f'Flight {fid} - {col}')
        else:
            axes[i, j].set_title(f'Flight {fid} - {col} NOT FOUND')

for ax in axes[-1]:
    ax.set_xlabel('Row number')

for i, (fid, df) in enumerate(flights.items()):
    axes[i, 0].set_ylabel('meters')
plt.tight_layout()
plt.show()
```

**What these graphs show:** The drone's physical position during each flight, measured in meters from where it took off.

- **x** = how far north/south the drone has traveled
- **y** = how far east/west
- **z** = altitude (negative because the coordinate system points downward, so -120 means 120 meters above the start)

The repeating wave patterns in x and y are the drone flying the racetrack circuit -- each peak and valley is a turn. The z shows the drone climbing and descending throughout the flight, staying roughly 50-120 meters above the starting point.

All three flights show very similar patterns, which makes sense since they're flying the same circuit. This is what the regression model would try to predict using only sensor data (IMU, airspeed, attitude, actuators) without GPS.


## Key Columns: Classification Path (Energy Efficiency)

For classification we need battery voltage, current (to compute power), and ground velocity (to compute efficiency = speed / power).

```python
# Battery columns
batt_cols = [c for c in df.columns if 'voltage_v' in c.lower() or 'current_a' in c.lower()]
print('Battery columns:')
for c in batt_cols:
    print(f'  {c}')
```

```python
# Velocity columns for ground speed
vel_cols = [c for c in df.columns if c in ['vx_f131', 'vy_f131', 'vz_f131']]
print('Velocity columns (for ground speed):')
for c in vel_cols:
    print(f'  {c}')
```

## Basic Stats on Key Variables

```python
# Stats across all three flights for key regression and classification columns
key_cols = ['voltage_v_f9', 'current_a_f9']

for fid, df in flights.items():
    print(f'--- Flight {fid} ---')
    available = [c for c in key_cols if c in df.columns]
    if available:
        print(df[available].describe())
    else:
        print('Key columns not found -- check column names')
    print()
```

## Classification Path: Energy Efficiency

For classification, we need to compute an efficiency metric: ground speed divided by electrical power. The paper defines anything below 0.1 m/W as inefficient.

**Power** = voltage x current (from f9 = `battery_status_0`)
**Ground speed** = sqrt(vx^2 + vy^2 + vz^2) (from f131 = `vehicle_local_position_0`)

Battery data looks clean across all three flights: voltage steady around 15.2-16V, current ranges from near-zero up to ~20A. No missing values.


## Time Series: Battery Voltage and Current

```python
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

for i, (fid, df) in enumerate(flights.items()):
    axes[i, 0].plot(df['voltage_v_f9'].values, linewidth=0.8)
    axes[i, 0].set_title(f'Flight {fid} - Voltage')
    axes[i, 0].set_ylabel('Volts')

    axes[i, 1].plot(df['current_a_f9'].values, linewidth=0.8, color='orange')
    axes[i, 1].set_title(f'Flight {fid} - Current')
    axes[i, 1].set_ylabel('Amps')

for ax in axes[-1]:
    ax.set_xlabel('Row number')

plt.tight_layout()
plt.show()
```

The paper defines energy efficiency as ground speed divided by electrical power (watts). Any moment where that ratio drops below 0.1 m/W is labeled as "inefficient." These are things like tight turns, steep climbs, or fighting headwinds. Below we compute this for each flight and see what percentage of the data falls below that threshold.

```python
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

for i, (fid, df) in enumerate(flights.items()):
    power = df['voltage_v_f9'] * df['current_a_f9']
    ground_speed = np.sqrt(df['vx_f131']**2 + df['vy_f131']**2 + df['vz_f131']**2)
    efficiency = ground_speed / power
    
    inefficient = efficiency < 0.1
    pct_inefficient = inefficient.sum() / len(df) * 100
    
    axes[i].plot(efficiency.values, linewidth=0.8)
    axes[i].axhline(y=0.1, color='red', linestyle='--', label='0.1 m/W threshold')
    axes[i].set_title(f'Flight {fid} - Efficiency ({pct_inefficient:.1f}% below threshold)')
    axes[i].set_ylabel('m/W')
    axes[i].legend()

axes[-1].set_xlabel('Row number')
plt.tight_layout()
plt.show()
```

**Finding:** The 0.1 m/W threshold from the paper produces a roughly 50/50 split between efficient and inefficient across all three flights (47-53% inefficient). That's a naturally balanced dataset for classification, which is a good sign.

A few things to note:
- The big spikes (3-6 m/W) are likely gliding moments where the drone covers distance on very little power
- Flight 11 shows a negative efficiency value, which shouldn't be possible -- worth investigating (possibly negative current or a data quirk)
- Most of the data hugs close to the threshold line, meaning the model will need to distinguish between values that are close together


## Classification Path Summary

**What we're predicting:** Whether the drone is flying efficiently or inefficiently at any given moment.

**How we create the label:**
- Calculate power: `voltage_v_f9` x `current_a_f9` (watts)
- Calculate ground speed: sqrt(`vx_f131`^2 + `vy_f131`^2 + `vz_f131`^2) (m/s)
- Calculate efficiency: ground speed / power (m/W)
- Label: below 0.1 m/W = inefficient (1), above = efficient (0)

**Variables needed:**

| Column | Description | Source file | Suffix |
|--------|------------|-------------|--------|
| `voltage_v_f9` | Battery voltage (V) | battery_status_0 | f9 |
| `current_a_f9` | Battery current (A) | battery_status_0 | f9 |
| `vx_f131` | Velocity north (m/s) | vehicle_local_position_0 | f131 |
| `vy_f131` | Velocity east (m/s) | vehicle_local_position_0 | f131 |
| `vz_f131` | Velocity down (m/s) | vehicle_local_position_0 | f131 |

**Key findings from Initial EDA:**
- Battery data is clean across all three flights, no nulls
- Voltage steady around 15.2-16V, current ranges 0-20A (consistent with paper's stated 18A cruise)
- The 0.1 m/W threshold produces a roughly 50/50 split (47-53% inefficient), so no class imbalance issue
- Flight 11 has a negative efficiency value worth investigating

**Predictor variables:** Still to be determined -- the label columns above are used to create the target, but the model inputs would likely be similar to the regression path (IMU, attitude, actuators) plus possibly battery features. Further exploration is needed. 
