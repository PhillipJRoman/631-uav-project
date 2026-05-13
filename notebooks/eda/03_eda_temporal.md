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

# EDA: Temporal Structure of UAV Energy Efficiency (Pixhawk Grouped)

Building on notebook 02, this notebook analyzes the temporal structure of the training data to inform feature engineering and model design decisions for the four-tier model comparison.

## Phase 1: Flight Integrity and Split Validation

Verify the training data loads cleanly, row order within each flight is preserved, the train/test split is representative across flight eras, and engineer the `time_since_start_of_flight` feature.

### Load Training Set and Verify Row Order

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 631

plt.style.use('tableau-colorblind10')

train = pd.read_csv('../../data/splits/train.csv')

print(f'Training rows: {len(train)}')
print(f'Training flights: {train["flight_id"].nunique()}')

# Verify row order within each flight is monotonic
# We use the original row index within each flight as the time axis
order_ok = True
for fid, group in train.groupby('flight_id'):
    if not group.index.is_monotonic_increasing:
        print(f'Flight {fid}: row index not monotonic')
        order_ok = False

print(f'\nRow order preserved within all flights: {order_ok}')
```

### Split Representativeness Check

Verify the stratified train/test split is balanced across flight eras (early vs late flights), not just across inefficiency rate. Notebook 02 found earlier flights trend more inefficient and later flights trend less inefficient, so we want to confirm both train and test contain a mix of eras.

```python
test = pd.read_csv('../../data/splits/test.csv')

train_ids = sorted(train['flight_id'].unique())
test_ids = sorted(test['flight_id'].unique())

print(f'Train flight ID range: {min(train_ids)} to {max(train_ids)}')
print(f'Test flight ID range: {min(test_ids)} to {max(test_ids)}')
print(f'\nTrain mean flight ID: {np.mean(train_ids):.1f}')
print(f'Test mean flight ID: {np.mean(test_ids):.1f}')

fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(train_ids, bins=30, alpha=0.6, label=f'Train (n={len(train_ids)})')
ax.hist(test_ids, bins=30, alpha=0.6, label=f'Test (n={len(test_ids)})')
ax.set_xlabel('Flight ID')
ax.set_ylabel('Count')
ax.set_title('Flight ID Distribution: Train vs Test')
ax.legend()
plt.tight_layout()
plt.show()
```

**Finding:** Train and test sets are well-balanced across flight eras. Train mean flight ID is 59.6, test mean is 60.8 (difference of 1.2 out of 116). Test flights span from flight 3 to flight 112, covering early, middle, and late eras of the dataset. The stratified split by inefficiency rate did not introduce a flight-era skew, so any temporal trend in the data (e.g. earlier flights trending more inefficient) is represented in both train and test.


### Engineer `time_since_start_of_flight`

Add a positional feature that captures how far into a flight each row is. This serves two purposes:

1. Gives flat models (Logistic Regression, XGBoost) a sense of flight progression without requiring full temporal context
2. Acts as a continuous proxy for flight phase (early rows = takeoff/climb, middle rows = cruise, late rows = descent/land) without requiring manual phase inference

The feature is the row number within each flight, starting at 0. At ~1.4 Hz sampling, row 100 is approximately 71 seconds into the flight.

```python
# Compute row index within each flight (0-based)
train['time_since_start_of_flight'] = train.groupby('flight_id').cumcount()

# check
print('Sample of new feature:')
print(train.groupby('flight_id')['time_since_start_of_flight'].agg(['min', 'max', 'count']).head())

print(f'\nOverall range: {train["time_since_start_of_flight"].min()} to {train["time_since_start_of_flight"].max()}')
```

**Finding:** Each flight now has a row counter starting at 0. Most flights run ~500 rows, with the longest at 707 rows. At 1.4 Hz sampling, that's roughly 6 minutes for a typical flight.


### Phase 1 Summary

- Training set loaded: 94 flights, 47,957 rows. Row order preserved within each flight.
- Train/test split is balanced across flight eras (train mean flight ID 59.6, test mean 60.8). No era skew from the stratified split.
- New feature `time_since_start_of_flight` added: row counter per flight, starting at 0.


## Phase 2: How Each Sensor Changes Over Time (One at a Time)

In Phase 1 we confirmed the data is in order. Now we look at how individual sensors behave over time, one sensor at a time. The goal is to answer: does a sensor's current value depend on its recent past values, or is each reading independent? This tells us whether temporal feature engineering (lags, rolling stats) will help our models.

We focus on three signals:
- **Throttle (`output[0]_f3`)**: strongest correlate with the target from notebook 02
- **Forward acceleration (`accelerometer_m_s2[0]_f104`)**: second strongest correlate
- **The target itself (`inefficient`)**: tells us if the label is "sticky" (current state predicts next state)

We use two tools:
- **Autocorrelation Function (ACF)**: measures how similar a signal is to its own past, at different time lags. Includes both direct and indirect effects.
- **Partial Autocorrelation Function (PACF)**: measures the *direct* relationship at each lag, after removing the influence of shorter lags.

In plain terms: ACF tells you "does the past matter at all?" PACF tells you "exactly which past timesteps matter directly?"


### ACF and PACF on the Target (column: `inefficient`)

We compute ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) on the target variable `inefficient`. Since the target switches sharply between 0 and 1 (from notebook 02), we expect high autocorrelation at short lags (the label is "sticky" within a segment) and a drop-off as lags get longer (we eventually cross into a different segment).

We compute ACF and PACF *per flight*, then average across flights. This avoids artificially inflating correlations by treating the entire training set as one long time series (it isn't, since flights are separate).

```python
from statsmodels.tsa.stattools import acf, pacf

MAX_LAG = 20

# Compute ACF and PACF per flight, then average
acf_per_flight = []
pacf_per_flight = []

for fid, group in train.groupby('flight_id'):
    y = group['inefficient'].values
    if len(y) > MAX_LAG + 1 and y.std() > 0:  # skip flights with no variance
        acf_per_flight.append(acf(y, nlags=MAX_LAG, fft=False))
        pacf_per_flight.append(pacf(y, nlags=MAX_LAG, method='ywm'))

acf_mean = np.mean(acf_per_flight, axis=0)
pacf_mean = np.mean(pacf_per_flight, axis=0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].stem(range(MAX_LAG + 1), acf_mean)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('ACF (Autocorrelation Function) -- Target')
axes[0].set_xlabel('Lag (rows)')
axes[0].set_ylabel('Correlation')

axes[1].stem(range(MAX_LAG + 1), pacf_mean)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('PACF (Partial Autocorrelation Function) -- Target')
axes[1].set_xlabel('Lag (rows)')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.show()

print(f'ACF at lag 1: {acf_mean[1]:.3f}')
print(f'ACF at lag 5: {acf_mean[5]:.3f}')
print(f'ACF at lag 10: {acf_mean[10]:.3f}')
print(f'PACF at lag 1: {pacf_mean[1]:.3f}')
print(f'PACF at lag 2: {pacf_mean[2]:.3f}')
```

**Finding:** The target is highly autocorrelated at lag 1 (0.838) but the PACF (Partial Autocorrelation Function) drops to near zero starting at lag 2. This means the target behaves as an AR(1) process: each moment depends mostly on the immediately previous moment, with no hidden longer-range structure in the label itself.

Practical implications:
- The persistence baseline (predict `target(t) = target(t-1)`) will be a strong floor, around 84% accuracy. Models must beat this.
- The label is "blocky" with runs of efficient and inefficient moments lasting roughly 10-20 rows on average.
- This says nothing about sensor lags. Sensors may carry predictive signal at longer lags even though the target itself does not. Phase 4 will check this.


### ACF and PACF on Throttle (column: `output[0]_f3`)

Now we run the same analysis on throttle (column `output[0]_f3`), the strongest predictor from notebook 02 (correlation with target = 0.572). This tells us how much memory throttle has on its own, and how far back throttle history might carry useful signal for the models.

```python
# Compute ACF and PACF per flight for throttle
acf_per_flight = []
pacf_per_flight = []

for fid, group in train.groupby('flight_id'):
    y = group['output[0]_f3'].values
    if len(y) > MAX_LAG + 1 and y.std() > 0:
        acf_per_flight.append(acf(y, nlags=MAX_LAG, fft=False))
        pacf_per_flight.append(pacf(y, nlags=MAX_LAG, method='ywm'))

acf_mean = np.mean(acf_per_flight, axis=0)
pacf_mean = np.mean(pacf_per_flight, axis=0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].stem(range(MAX_LAG + 1), acf_mean)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('ACF (Autocorrelation Function) -- Throttle')
axes[0].set_xlabel('Lag (rows)')
axes[0].set_ylabel('Correlation')

axes[1].stem(range(MAX_LAG + 1), pacf_mean)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('PACF (Partial Autocorrelation Function) -- Throttle')
axes[1].set_xlabel('Lag (rows)')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.show()

print(f'ACF at lag 1: {acf_mean[1]:.3f}')
print(f'ACF at lag 5: {acf_mean[5]:.3f}')
print(f'ACF at lag 10: {acf_mean[10]:.3f}')
print(f'PACF at lag 1: {pacf_mean[1]:.3f}')
print(f'PACF at lag 2: {pacf_mean[2]:.3f}')
print(f'PACF at lag 3: {pacf_mean[3]:.3f}')
```

**Finding:** Throttle is an AR(2) process. The previous two rows directly carry information; lag 3 and beyond add nothing new directly. The negative PACF at lag 2 (-0.287) reflects autopilot correction behavior: throttle is pushed up, then pulled back as the drone stabilizes.

Practical implications:
- `throttle_lag_1` and `throttle_lag_2` are both worth adding as features.
- First differences on throttle (`throttle(t) - throttle(t-1)`) directly capture the autopilot's correction signal and are high-priority features per Professor Wang's recommendation.
- Rolling statistics over short windows (3-5 rows) will capture local dynamics. Rolling standard deviation captures volatility during transitions.
- Throttle has more useful temporal structure than the target itself, meaning sensor lags will likely outperform target lags as features.


### ACF and PACF on Forward Acceleration (column: `accelerometer_m_s2[0]_f104`)

Now we run the same analysis on forward acceleration (column `accelerometer_m_s2[0]_f104`), the second strongest predictor from notebook 02 (correlation with target = 0.488). Acceleration is expected to be noisier than throttle, so the temporal structure may look different.

```python
# Compute ACF and PACF per flight for forward acceleration
acf_per_flight = []
pacf_per_flight = []

for fid, group in train.groupby('flight_id'):
    y = group['accelerometer_m_s2[0]_f104'].values
    if len(y) > MAX_LAG + 1 and y.std() > 0:
        acf_per_flight.append(acf(y, nlags=MAX_LAG, fft=False))
        pacf_per_flight.append(pacf(y, nlags=MAX_LAG, method='ywm'))

acf_mean = np.mean(acf_per_flight, axis=0)
pacf_mean = np.mean(pacf_per_flight, axis=0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].stem(range(MAX_LAG + 1), acf_mean)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('ACF (Autocorrelation Function) -- Forward Acceleration')
axes[0].set_xlabel('Lag (rows)')
axes[0].set_ylabel('Correlation')

axes[1].stem(range(MAX_LAG + 1), pacf_mean)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('PACF (Partial Autocorrelation Function) -- Forward Acceleration')
axes[1].set_xlabel('Lag (rows)')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.show()

print(f'ACF at lag 1: {acf_mean[1]:.3f}')
print(f'ACF at lag 5: {acf_mean[5]:.3f}')
print(f'ACF at lag 10: {acf_mean[10]:.3f}')
print(f'PACF at lag 1: {pacf_mean[1]:.3f}')
print(f'PACF at lag 2: {pacf_mean[2]:.3f}')
print(f'PACF at lag 3: {pacf_mean[3]:.3f}')
```

**Finding:** Forward acceleration is an AR(3) process. The previous three rows directly carry information, with lag 1 (0.590), lag 2 (0.271), and lag 3 (0.153). No negative correction term like throttle, because acceleration is a response to forces rather than a control command.

Practical implications:
- Lag features at lag 1, 2, and 3 are all justified for forward acceleration.
- First differences will pick up real signal but will be noisier than throttle's first differences.
- Second differences will be very noisy on this signal due to the lower lag-1 ACF. Include them in uniform engineering per Approach B, but expect lower usefulness than for throttle.
- Rolling statistics (mean and std over 3-5 rows) are likely to be particularly valuable here for smoothing out the noise.
- Combined with throttle and the target, a uniform lag depth of 2 or 3 covers the direct temporal signal across these sensors.


### ACF and PACF on Yaw Rate (column: `gyro_rad[2]_f104`)

Now we run the same analysis on yaw rate (column `gyro_rad[2]_f104`), the dominant rotation for a fixed-wing UAV flying a racetrack circuit. Yaw rate captures how fast the drone is turning left/right, which should be highest during corners and near zero during straight cruise segments. This gives us a look at how IMU-class sensors behave temporally.

```python
# Compute ACF and PACF per flight for yaw rate
acf_per_flight = []
pacf_per_flight = []

for fid, group in train.groupby('flight_id'):
    y = group['gyro_rad[2]_f104'].values
    if len(y) > MAX_LAG + 1 and y.std() > 0:
        acf_per_flight.append(acf(y, nlags=MAX_LAG, fft=False))
        pacf_per_flight.append(pacf(y, nlags=MAX_LAG, method='ywm'))

acf_mean = np.mean(acf_per_flight, axis=0)
pacf_mean = np.mean(pacf_per_flight, axis=0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].stem(range(MAX_LAG + 1), acf_mean)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('ACF (Autocorrelation Function) -- Yaw Rate')
axes[0].set_xlabel('Lag (rows)')
axes[0].set_ylabel('Correlation')

axes[1].stem(range(MAX_LAG + 1), pacf_mean)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('PACF (Partial Autocorrelation Function) -- Yaw Rate')
axes[1].set_xlabel('Lag (rows)')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.show()

print(f'ACF at lag 1: {acf_mean[1]:.3f}')
print(f'ACF at lag 5: {acf_mean[5]:.3f}')
print(f'ACF at lag 10: {acf_mean[10]:.3f}')
print(f'PACF at lag 1: {pacf_mean[1]:.3f}')
print(f'PACF at lag 2: {pacf_mean[2]:.3f}')
print(f'PACF at lag 3: {pacf_mean[3]:.3f}')
```

**Finding:** Yaw rate is an AR(1) process with fast memory decay. The previous row directly carries information (PACF lag 1 = 0.558), but lag 2 and beyond add nothing. The ACF (Autocorrelation Function) drops to zero by lag 5, meaning yaw rate has no memory beyond ~3 seconds.

Practical implications:
- Only lag 1 has direct statistical justification for yaw rate. Lag 2 and 3 features under uniform engineering (Approach B) will be statistical noise that the models can downweight.
- First differences on yaw rate are likely valuable: they capture turn initiation and termination.
- Second differences are likely noise on this signal.
- Rolling standard deviation over 3-5 rows will spike during turns and stay low during straight cruise, potentially useful for distinguishing flight phases.
- Across the four signals examined so far (target, throttle, forward accel, yaw rate), AR order ranges from 1 to 3. A uniform lag depth of 2 or 3 covers most direct signal, with the models filtering irrelevant lags per sensor.


### ACF and PACF on Quaternion W (column: `q[0]_f119`)

Now we run the same analysis on the quaternion w component (column `q[0]_f119`), the orientation component with the widest distribution from notebook 02. Quaternions describe overall drone orientation, and q[0] changes continuously as the drone flies the circuit. This gives us a look at how the attitude sensor group behaves temporally.

```python
# Compute ACF and PACF per flight for quaternion w
acf_per_flight = []
pacf_per_flight = []

for fid, group in train.groupby('flight_id'):
    y = group['q[0]_f119'].values
    if len(y) > MAX_LAG + 1 and y.std() > 0:
        acf_per_flight.append(acf(y, nlags=MAX_LAG, fft=False))
        pacf_per_flight.append(pacf(y, nlags=MAX_LAG, method='ywm'))

acf_mean = np.mean(acf_per_flight, axis=0)
pacf_mean = np.mean(pacf_per_flight, axis=0)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].stem(range(MAX_LAG + 1), acf_mean)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('ACF (Autocorrelation Function) -- Quaternion W')
axes[0].set_xlabel('Lag (rows)')
axes[0].set_ylabel('Correlation')

axes[1].stem(range(MAX_LAG + 1), pacf_mean)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('PACF (Partial Autocorrelation Function) -- Quaternion W')
axes[1].set_xlabel('Lag (rows)')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.show()

print(f'ACF at lag 1: {acf_mean[1]:.3f}')
print(f'ACF at lag 5: {acf_mean[5]:.3f}')
print(f'ACF at lag 10: {acf_mean[10]:.3f}')
print(f'PACF at lag 1: {pacf_mean[1]:.3f}')
print(f'PACF at lag 2: {pacf_mean[2]:.3f}')
print(f'PACF at lag 3: {pacf_mean[3]:.3f}')
```

**Finding:** Quaternion w is an AR(2) process with a negative correction term (PACF lag 2 = -0.288), the same pattern as throttle. The very slow ACF decay (still 0.55 at lag 20) confirms quaternion w is the smoothest signal examined.

Practical implications:
- Lag 1 and lag 2 features are well justified for quaternion w.
- First differences capture rotation rate. Second differences capture the autopilot's heading correction behavior, parallel to the throttle autopilot's behavior.
- Rolling statistics will be less informative here due to the very high ACF -- raw values and rolling means will be nearly identical.
- The four quaternion components likely follow similar AR(1) or AR(2) patterns; the established pattern is sufficient to set uniform engineering depth.

**Phase 2 conclusion:** Across the four representative sensors examined, AR order ranges from 1 to 3. A uniform lag depth of 3 (per Approach B) covers the maximum direct memory observed. For sensors with shorter direct memory, extra lag features become statistical noise that models will downweight through their native feature selection mechanisms.


### Phase 2 Summary

Examined ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) on four representative signals:

| Signal | Column | AR Order |
|---|---|---|
| Target | `inefficient` | AR(1) |
| Yaw rate | `gyro_rad[2]_f104` | AR(1) |
| Throttle | `output[0]_f3` | AR(2) with correction |
| Quaternion w | `q[0]_f119` | AR(2) with correction |
| Forward acceleration | `accelerometer_m_s2[0]_f104` | AR(3) |

**Key takeaways for feature engineering:**

- **Uniform lag depth up to 3** is statistically defensible across all sensor classes per Approach B. Sensors with shorter memory will have noise lags that models downweight.
- **First differences are justified for all sensors.** They capture rate of change, which is physically meaningful (rotation rate, acceleration change, throttle change).
- **Second differences are justified for smooth AR(2) signals** (throttle, quaternions). They directly measure autopilot correction behavior. Less useful for noisy AR(1) signals (yaw rate) butcould be considered under uniform engineering.
- **Rolling statistics hold value for noisy signals** (forward acceleration). Less informative for very smooth signals (quaternions) where rolling means closely track raw values.
- **The persistence baseline (target lag 1 = 0.838) will be a strong floor.** Models must beat this to be useful.



## Phase 3: How Sensors Relate to Each Other (Multivariate Analysis)

In Phase 2 we looked at sensors one at a time. Now we look at how sensors relate to each other. This phase exists to inform the GAT (Graph Attention Network) model in the DOE: GAT needs a graph structure where edges connect sensors that have meaningful relationships. If sensors don't relate to each other in a structured way, GAT has no advantage over flat models.

We answer two questions:
1. **Which sensors are correlated overall?** A static correlation heatmap shows baseline relationships.
2. **Do sensor relationships change between efficient and inefficient moments?** If relationships shift during inefficient periods, GAT can detect that through attention weights. If relationships are constant, GAT has less to work with.


### Correlation Heatmap Across All 16 Sensors

We compute the Pearson correlation matrix across the 16 sensor predictors on the training set. This shows which sensors move together, regardless of efficiency state. Strong correlations indicate sensor pairs that should have edges in the GAT graph.

```python
predictor_cols = [
    'gyro_rad[0]_f104', 'gyro_rad[1]_f104', 'gyro_rad[2]_f104',
    'accelerometer_m_s2[0]_f104', 'accelerometer_m_s2[1]_f104', 'accelerometer_m_s2[2]_f104',
    'indicated_airspeed_m_s_f5', 'true_airspeed_m_s_f5',
    'q[0]_f119', 'q[1]_f119', 'q[2]_f119', 'q[3]_f119',
    'output[0]_f3', 'output[1]_f3', 'output[2]_f3', 'output[3]_f3',
]

# Short labels for plotting
short_labels = [
    'gyro_roll', 'gyro_pitch', 'gyro_yaw',
    'accel_fwd', 'accel_side', 'accel_up',
    'airspeed_ind', 'airspeed_true',
    'q_w', 'q_x', 'q_y', 'q_z',
    'throttle', 'aileron', 'elevator', 'rudder',
]

corr = train[predictor_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='cividis', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=45, ha='right')
ax.set_yticklabels(short_labels)

# Annotate cells with correlation values
for i in range(len(short_labels)):
    for j in range(len(short_labels)):
        val = corr.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=7)

ax.set_title('Sensor Correlation Heatmap (Training Set)')
plt.colorbar(im, ax=ax, label='Pearson correlation')
plt.tight_layout()
plt.show()
```

**Finding:** The 16 sensors show a structured correlation pattern grounded in flight physics:

- **Strong control-response pairs:** aileron ↔ gyro_roll (-0.76), throttle ↔ forward acceleration (0.63), elevator ↔ gyro_pitch (0.56), rudder ↔ side acceleration (-0.50)
- **Coupled dynamics:** gyro_pitch ↔ vertical acceleration (-0.51), throttle ↔ elevator (0.40)
- **Redundant sensors:** indicated airspeed and true airspeed are identical (1.00)
- **Quaternion isolation:** the four quaternion components show very weak correlations (most below 0.15) with all other sensors, including each other

Practical implications for the GAT (Graph Attention Network) model:

- The natural graph structure follows flight physics: control surfaces connect to their response sensors. This gives GAT physically meaningful edges to attend over.
- Quaternions appear disconnected from instantaneous sensor relationships because they integrate rates over time. Three options for GAT design: drop quaternions, connect them to gyros based on the mathematical relationship, or use a fully connected graph and let attention learn edge weights. Decision deferred to modeling phase.
- Pearson correlation captures only linear relationships. The class-conditional comparison in the next block addresses whether relationships shift during inefficient moments, which is the deeper question for GAT justification.


### Class-Conditional Correlation Comparison - between Efficient and Inefficient

The overall heatmap shows baseline sensor relationships. Now we ask the deeper question for GAT: do these relationships *change* between efficient and inefficient moments? If correlations shift meaningfully across classes, GAT can exploit that through attention weights. If they stay constant, the graph structure carries less information.

We compute the correlation matrix separately on efficient rows (target = 0) and inefficient rows (target = 1), then subtract to find the differences. Large differences flag sensor pairs whose relationship depends on flight efficiency state.

```python
efficient = train[train['inefficient'] == 0]
inefficient = train[train['inefficient'] == 1]

corr_eff = efficient[predictor_cols].corr()
corr_ineff = inefficient[predictor_cols].corr()
corr_diff = corr_ineff - corr_eff

# Plot the difference heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_diff, cmap='cividis', vmin=-0.5, vmax=0.5, aspect='auto')

ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=45, ha='right')
ax.set_yticklabels(short_labels)

# Annotate cells with the difference values
for i in range(len(short_labels)):
    for j in range(len(short_labels)):
        val = corr_diff.iloc[i, j]
        color = 'white' if abs(val) > 0.25 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=7)

ax.set_title('Correlation Difference: Inefficient minus Efficient\n(positive = stronger correlation during inefficient moments)')
plt.colorbar(im, ax=ax, label='Correlation difference')
plt.tight_layout()
plt.show()

# Print the largest shifts
print('\nLargest correlation shifts (|diff| > 0.15):\n')
pairs = []
for i in range(len(predictor_cols)):
    for j in range(i + 1, len(predictor_cols)):
        diff = corr_diff.iloc[i, j]
        if abs(diff) > 0.15:
            pairs.append((short_labels[i], short_labels[j], corr_eff.iloc[i, j], corr_ineff.iloc[i, j], diff))

pairs.sort(key=lambda x: abs(x[4]), reverse=True)
for n1, n2, eff_r, ineff_r, diff in pairs:
    print(f'  {n1:<15} <-> {n2:<15}  efficient: {eff_r:+.2f}  inefficient: {ineff_r:+.2f}  diff: {diff:+.2f}')
```

**Finding:** Most sensor relationships are stable across classes, but three regions shift meaningfully:

1. **Quaternions couple during inefficient moments.** All six quaternion pairs shift strongly (e.g., q_x ↔ q_y: -0.13 to +0.41). During efficient cruise they vary independently; during inefficient moments they move together.
2. **Throttle decouples from other sensors.** Throttle ↔ forward acceleration drops from +0.58 to +0.23. The clean throttle-response relationship breaks down during inefficient moments.
3. **Pitch-airspeed coupling emerges.** Airspeed ↔ elevator goes from ~0.00 to -0.27. Nearly absent during cruise, emerges during pitch-driven segments.

**GAT implication:** The graph has both static edges (baseline physics) and conditional edges (target-dependent). Quaternions are not disconnected, they have *conditional* coupling. This justifies GAT over flat models: attention can amplify or suppress edges based on the target state.


### Phase 3 Summary

**Static relationships (overall heatmap):** Sensor pairs follow flight physics. Strong control-response edges (aileron↔roll, throttle↔forward accel, elevator↔pitch, rudder↔side accel). Airspeed sensors are redundant. Quaternions appear weakly connected.

**Conditional relationships (class-conditional comparison):** Quaternions couple during inefficient moments, throttle decouples from forward acceleration, and a pitch-airspeed-elevator triangle emerges.

**Implications for feature engineering:**

- **Interaction features may help flat models** (Logistic Regression, XGBoost). Specifically: throttle × forward acceleration, airspeed × elevator, and quaternion product or sum features. These capture relationships that shift with the target.
- **GAT is justified** by the conditional edge structure. No new feature engineering needed for GAT, the graph itself carries the information.
- **Redundant airspeed features should be reduced to one** before modeling (already flagged in notebook 02).
- **Quaternions stay in the feature set.** They looked disconnected in the overall heatmap but carry strong class-dependent information.


## Phase 4: When Do Sensors Carry Signal Relative to the Target?

In Phase 2 we measured how sensors relate to themselves over time. In Phase 3 we measured how sensors relate to each other. Now we measure how sensors relate to the target at different time lags. This phase sets the lookback window for uniform lag features.

For each sensor, we compute the correlation between the sensor at time t-k and the target at time t, for k from 0 to 15. This tells us how far back in time a sensor's value still carries useful information about the current target. The lag where correlation drops to noise sets a defensible upper bound for k.


### Cross-Correlation Between All 16 Sensors and the Target

For each of the 16 sensors, we compute the correlation between the sensor at time t-k and the target at time t, for k from 0 to 5. Computed per flight, then averaged across flights. The lag where correlation drops to noise sets the upper bound for uniform lag features.

```python
MAX_K = 5

xcorr_per_sensor = {col: [] for col in predictor_cols}

for fid, group in train.groupby('flight_id'):
    group = group.reset_index(drop=True)
    target = group['inefficient'].values
    for col in predictor_cols:
        sensor = group[col].values
        flight_xcorr = []
        for k in range(MAX_K + 1):
            if k == 0:
                # Correlation at t=0 (no lag)
                r = np.corrcoef(sensor, target)[0, 1]
            else:
                # Sensor at t-k vs target at t
                r = np.corrcoef(sensor[:-k], target[k:])[0, 1]
            flight_xcorr.append(r)
        xcorr_per_sensor[col].append(flight_xcorr)

# Average across flights
xcorr_mean = {col: np.nanmean(xcorr_per_sensor[col], axis=0) for col in predictor_cols}

# Plot all sensors on one figure
fig, ax = plt.subplots(figsize=(8, 4))

for col, label in zip(predictor_cols, short_labels):
    ax.plot(range(MAX_K + 1), xcorr_mean[col], marker='o', label=label, linewidth=1)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Lag k (sensor at t-k vs target at t)')
ax.set_ylabel('Correlation with target')
ax.set_title('Cross-Correlation: All 16 Sensors vs Target')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print table of values
print('\nCross-correlation values (averaged across flights):\n')
print(f'{"Sensor":<18} {"k=0":>8} {"k=1":>8} {"k=2":>8} {"k=3":>8} {"k=4":>8} {"k=5":>8}')
for col, label in zip(predictor_cols, short_labels):
    vals = xcorr_mean[col]
    print(f'{label:<18} {vals[0]:+.3f}  {vals[1]:+.3f}  {vals[2]:+.3f}  {vals[3]:+.3f}  {vals[4]:+.3f}  {vals[5]:+.3f}')
```

**Finding:** Three tiers of sensors emerge:

- **Strong, slow-decay:** throttle (0.580 → 0.461) and forward acceleration (0.500 → 0.366) carry signal at all lags k=0 to k=5 without sharp drop-off.
- **Moderate, faster-decay:** elevator (0.342 → 0.197) loses most of its signal by k=3. Side acceleration (0.115 → 0.084) is modest but stable.
- **Noise:** gyros, quaternions, airspeed, aileron, rudder, vertical accel all stay within ±0.10 at every lag. Limited direct linear relationship with the target.

**Decision: uniform lag depth k=2 for feature engineering.** Justification: matches the direct PACF memory of the top AR(2) sensor (throttle) and captures most of the elevator drop-off. Forward acceleration (AR(3)) carries some signal at k=3, but the slow decay across all 5 lags means any k within 2-5 is defensible. k=2 keeps the feature space manageable.


**Finding:** Three tiers of sensors emerge:

- **Strong, slow-decay:** throttle (0.580 → 0.461) and forward acceleration (0.500 → 0.366) carry signal at all lags k=0 to k=5 without sharp drop-off.
- **Moderate, faster-decay:** elevator (0.342 → 0.197) loses most of its signal by k=3. Side acceleration (0.115 → 0.084) is modest but stable.
- **Noise:** gyros, quaternions, airspeed, aileron, rudder, vertical accel all stay within ±0.10 at every lag. Limited direct linear relationship with the target.

**Decision: uniform lag depth k=2 for feature engineering.** Justification: matches the direct PACF memory of the top AR(2) sensor (throttle) and captures most of the elevator drop-off. Forward acceleration (AR(3)) carries some signal at k=3, but the slow decay across all 5 lags means any k within 2-5 is defensible. k=2 keeps the feature space manageable.


### Phase 4 Summary

**Cross-correlation findings:**
- Throttle and forward acceleration are the dominant signals at all lags (k=0 to k=5).
- Elevator shows the clearest lag-based drop-off, losing most of its signal by k=3.
- The remaining sensors stay within ±0.10 at every lag, indicating limited direct linear relationship with the target.

**Decision for feature engineering:**
- Uniform lag depth: **k=2** across all 16 sensors.
- This locks the lag features as: lag 1 and lag 2 for each sensor (32 lag features total).
- Linear correlation does not capture the full picture: low-correlation sensors may still contribute through non-linear interactions, conditional relationships (Phase 3), or temporal patterns the models can learn directly. We keep all 16 sensors in the feature set.


## Phase 5: Build and Validate Engineered Features

Phases 1 through 4 set the analytical groundwork. Now we build the engineered features and test whether they actually separate the classes better than raw values. Features that pass the test go into the modeling phase. Features that don't, get dropped or flagged.

Based on the prior phases, the feature set under consideration:

- **Lag features at k=1 and k=2** for each of the 16 sensors (32 features)
- **First differences** for each sensor (16 features)
- **Second differences** for each sensor (16 features, may or may not be useful)
- **Rolling mean, standard deviation, and range** at window sizes 3 and 5 for each sensor (96 features)
- **Interaction features:** `throttle × forward acceleration` and `airspeed × elevator` (2 features)
- **`time_since_start_of_flight`** (already built in Phase 1, will also be tested here)

Total candidate features: ~163. We build them all with uniform engineering, then test class separation per feature family. Features that don't separate classes get dropped or flagged.

**Class separation metric:** Cohen's d, the difference between class means divided by pooled standard deviation. A single number per feature lets us rank and compare. Rough interpretation: |d| < 0.2 is negligible separation, 0.2-0.5 is small, 0.5-0.8 is medium, > 0.8 is large.

**Strict rule on leakage:** Every rolling statistic and difference uses `.shift(1)` so the current row's value cannot leak into its own feature. All features are computed *within* each flight (groupby flight_id), never across flight boundaries.


### Build Lag Features and Test Class Separation

We build lag-1 and lag-2 features for each of the 16 sensors. Lag features are computed within each flight using `.shift(k)` after grouping by `flight_id`, which prevents any cross-flight leakage. The first k rows of each flight are NaN (no prior value exists) and are excluded from the separation test.

We also test `time_since_start_of_flight` here as a baseline standalone feature.

For each feature, we compute Cohen's d to measure class separation. Cohen's d above 0.2 indicates at least small separation; above 0.5 is medium; above 0.8 is large.

```python
def cohens_d(series, target):
    """Compute Cohen's d between two classes for a single feature."""
    s = series.dropna()
    t = target.loc[s.index]
    g0 = s[t == 0]
    g1 = s[t == 1]
    if len(g0) < 2 or len(g1) < 2:
        return np.nan
    pooled_std = np.sqrt(((g0.std() ** 2) * (len(g0) - 1) + (g1.std() ** 2) * (len(g1) - 1)) / (len(g0) + len(g1) - 2))
    if pooled_std == 0:
        return np.nan
    return (g1.mean() - g0.mean()) / pooled_std

# Build lag features within each flight
for col in predictor_cols:
    train[f'{col}_lag1'] = train.groupby('flight_id')[col].shift(1)
    train[f'{col}_lag2'] = train.groupby('flight_id')[col].shift(2)

# Compute Cohen's d for each lag feature and time_since_start_of_flight
results = []

# Time since start of flight
d = cohens_d(train['time_since_start_of_flight'], train['inefficient'])
results.append(('time_since_start_of_flight', 'baseline', d))

# Raw values (for comparison)
for col, label in zip(predictor_cols, short_labels):
    d = cohens_d(train[col], train['inefficient'])
    results.append((label, 'raw', d))

# Lag 1
for col, label in zip(predictor_cols, short_labels):
    d = cohens_d(train[f'{col}_lag1'], train['inefficient'])
    results.append((f'{label}_lag1', 'lag1', d))

# Lag 2
for col, label in zip(predictor_cols, short_labels):
    d = cohens_d(train[f'{col}_lag2'], train['inefficient'])
    results.append((f'{label}_lag2', 'lag2', d))

results_df = pd.DataFrame(results, columns=['feature', 'family', 'cohens_d'])
results_df['abs_d'] = results_df['cohens_d'].abs()
results_df = results_df.sort_values('abs_d', ascending=False).reset_index(drop=True)

print('Top 20 features by |Cohen\'s d|:\n')
print(results_df.head(20).to_string(index=False))
print(f'\nFeatures with |d| > 0.2 (at least small separation): {(results_df["abs_d"] > 0.2).sum()} of {len(results_df)}')
print(f'Features with |d| > 0.5 (medium separation): {(results_df["abs_d"] > 0.5).sum()} of {len(results_df)}')
```

**Finding:** Three sensors dominate raw class separation:
- **Throttle:** d = 1.42 (large)
- **Forward acceleration:** d = 1.14 (large)
- **Elevator:** d = 0.51 (medium)

**Lag features are individually weaker than their raw counterparts** for every sensor (e.g., throttle: 1.42 raw vs 1.26 at lag 2). This is expected: Cohen's d measures marginal separation, and lag features carry the same information as raw with one or two steps of additional noise. Their value emerges through interactions with current values during modeling, not through marginal separation. **Conclusion: do not drop lag features based on Cohen's d. Re-evaluate using model-based feature importance during the modeling phase.**

**`time_since_start_of_flight`** shows d = -0.307 (small, negative). Inefficient moments occur earlier in flights on average, consistent with climb-out and acceleration phases. Worth keeping.

10 of 49 features show at least small separation (|d| > 0.2). 

Only 7 of 49 show medium or large separation, all from raw throttle, raw forward accel, raw elevator, and their lag variants.


### Build First Differences and Test Class Separation

A first difference is the change from one row to the next: `feature(t) - feature(t-1)`. It captures velocity in feature space (how fast a sensor is changing). Computed within each flight using `.diff(1)` after grouping by `flight_id`. The first row of each flight is NaN.

Cohen's d is computed for each first-difference feature and compared against raw values from the previous block.

```python
# Build first differences within each flight
for col in predictor_cols:
    train[f'{col}_diff1'] = train.groupby('flight_id')[col].diff(1)

# Compute Cohen's d for each first difference
diff_results = []
for col, label in zip(predictor_cols, short_labels):
    d = cohens_d(train[f'{col}_diff1'], train['inefficient'])
    diff_results.append((f'{label}_diff1', 'diff1', d))

diff_df = pd.DataFrame(diff_results, columns=['feature', 'family', 'cohens_d'])
diff_df['abs_d'] = diff_df['cohens_d'].abs()
diff_df = diff_df.sort_values('abs_d', ascending=False).reset_index(drop=True)

print('First differences ranked by |Cohen\'s d|:\n')
print(diff_df.to_string(index=False))

# Compare each first difference to its raw counterpart
print('\n\nFirst difference vs raw value (separation comparison):\n')
print(f'{"Sensor":<18} {"raw |d|":>10} {"diff1 |d|":>12} {"verdict":>20}')
for col, label in zip(predictor_cols, short_labels):
    raw_d = abs(cohens_d(train[col], train['inefficient']))
    diff_d = abs(cohens_d(train[f'{col}_diff1'], train['inefficient']))
    if diff_d > raw_d + 0.05:
        verdict = 'diff1 stronger'
    elif diff_d < raw_d - 0.05:
        verdict = 'raw stronger'
    else:
        verdict = 'similar'
    print(f'{label:<18} {raw_d:>10.3f} {diff_d:>12.3f} {verdict:>20}')
```

```python
from sklearn.feature_selection import mutual_info_classif

def mutual_info_single(series, target):
    """Compute mutual information between a single feature and the target."""
    s = series.dropna()
    t = target.loc[s.index]
    mi = mutual_info_classif(s.values.reshape(-1, 1), t.values, random_state=RANDOM_SEED)
    return mi[0]

# Compute mutual information for raw values and first differences
mi_results = []
for col, label in zip(predictor_cols, short_labels):
    mi_raw = mutual_info_single(train[col], train['inefficient'])
    mi_diff = mutual_info_single(train[f'{col}_diff1'], train['inefficient'])
    mi_results.append((label, mi_raw, mi_diff))

mi_df = pd.DataFrame(mi_results, columns=['sensor', 'mi_raw', 'mi_diff1'])
mi_df['diff_minus_raw'] = mi_df['mi_diff1'] - mi_df['mi_raw']
mi_df = mi_df.sort_values('mi_diff1', ascending=False).reset_index(drop=True)

print('Mutual information: raw vs first difference\n')
print(f'{"Sensor":<18} {"MI raw":>10} {"MI diff1":>12} {"diff - raw":>14}')
for _, row in mi_df.iterrows():
    print(f'{row["sensor"]:<18} {row["mi_raw"]:>10.4f} {row["mi_diff1"]:>12.4f} {row["diff_minus_raw"]:>+14.4f}')

print(f'\nFirst differences with MI > 0.01: {(mi_df["mi_diff1"] > 0.01).sum()} of 16')
print(f'First differences with MI > 0.05: {(mi_df["mi_diff1"] > 0.05).sum()} of 16')
```

**Finding:** Mutual information confirms first differences carry minimal information about the target. Only `throttle_diff1` exceeds MI = 0.01 (at 0.040), and no first difference exceeds 0.05. For every meaningful sensor, the raw value carries substantially more information than its first difference (e.g., throttle: 0.258 raw vs 0.040 diff1; forward acceleration: 0.138 raw vs 0.0005 diff1).

**Decision: drop first differences.** Mutual information confirms what Cohen's d showed: the classes are characterized by absolute sensor levels, not by rates of change at single timesteps. Lag features already give models access to the information needed to compute differences implicitly if useful.


### Build Second Differences and Test Class Separation

A second difference is the change in the first difference: `(feature(t) - feature(t-1)) - (feature(t-1) - feature(t-2))`. It captures acceleration in feature space (how the rate of change is itself changing). Computed within each flight using `.diff().diff()` after grouping by `flight_id`. The first two rows of each flight are NaN.

Cohen's d is computed for each second-difference feature.

```python
# Build second differences within each flight
for col in predictor_cols:
    train[f'{col}_diff2'] = train.groupby('flight_id')[col].diff().diff()

# Compute Cohen's d for each second difference
diff2_results = []
for col, label in zip(predictor_cols, short_labels):
    d = cohens_d(train[f'{col}_diff2'], train['inefficient'])
    diff2_results.append((f'{label}_diff2', 'diff2', d))

diff2_df = pd.DataFrame(diff2_results, columns=['feature', 'family', 'cohens_d'])
diff2_df['abs_d'] = diff2_df['cohens_d'].abs()
diff2_df = diff2_df.sort_values('abs_d', ascending=False).reset_index(drop=True)

print('Second differences ranked by |Cohen\'s d|:\n')
print(diff2_df.to_string(index=False))

print(f'\nSecond differences with |d| > 0.2: {(diff2_df["abs_d"] > 0.2).sum()} of 16')
print(f'Second differences with |d| > 0.5: {(diff2_df["abs_d"] > 0.5).sum()} of 16')
```

**Finding:** Second differences show no class separation. Strongest is `elevator_diff2` at |d| = 0.072, well below the 0.2 small-separation threshold. None of the 16 features reach even small separation. This is weaker than first differences, consistent with noise amplification through repeated differencing.

**Decision: drop second differences.** No verification with mutual information needed since no feature shows meaningful separation under Cohen's d.


### Build Rolling Statistics (Mean, Std, Range) at Windows 3, 5, and 7

We build three rolling statistics at three window sizes for each of the 16 sensors. Total: 16 sensors × 3 stats × 3 windows = 144 features.

Each rolling stat uses `.shift(1)` before the rolling window so the current row's value cannot leak into its own feature. The rolling window then covers rows t-1 through t-window, computed within each flight only.

Window 7 is included to test the upper limit. Phase 2 ACF suggested label segments average 10-20 rows, so window 7 may begin to blur across transitions. If window 7 underperforms window 5, it confirms our upper bound.

Cohen's d is computed for each feature. We then summarize separation by window size and statistic type.

```python
train = train.loc[:, ~train.columns.duplicated()]

roll_cols = [c for c in train.columns if '_rmean' in c or '_rstd' in c or '_rrange' in c]
train = train.drop(columns=roll_cols)

print(f'Columns after cleanup: {len(train.columns)}')
```

```python
duplicates = train.columns[train.columns.duplicated()].tolist()
print(f'Duplicate columns: {len(duplicates)}')
print(duplicates[:10])
```

```python
WINDOWS = [3, 5, 7, 10]
STATS = ['mean', 'std', 'range']

# Build rolling features
for col in predictor_cols:
    for w in WINDOWS:
        # Shift by 1 first to prevent current row from leaking into its own feature
        shifted = train.groupby('flight_id')[col].shift(1)
        train[f'{col}_rmean{w}'] = shifted.groupby(train['flight_id']).rolling(window=w, min_periods=w).mean().reset_index(level=0, drop=True)
        train[f'{col}_rstd{w}'] = shifted.groupby(train['flight_id']).rolling(window=w, min_periods=w).std().reset_index(level=0, drop=True)
        rmax = shifted.groupby(train['flight_id']).rolling(window=w, min_periods=w).max().reset_index(level=0, drop=True)
        rmin = shifted.groupby(train['flight_id']).rolling(window=w, min_periods=w).min().reset_index(level=0, drop=True)
        train[f'{col}_rrange{w}'] = rmax - rmin

# Compute Cohen's d for each rolling feature
roll_results = []
for col, label in zip(predictor_cols, short_labels):
    for w in WINDOWS:
        for stat in STATS:
            feat = f'{col}_r{stat}{w}'
            d = cohens_d(train[feat], train['inefficient'])
            roll_results.append((f'{label}_r{stat}{w}', stat, w, d))

roll_df = pd.DataFrame(roll_results, columns=['feature', 'stat', 'window', 'cohens_d'])
roll_df['abs_d'] = roll_df['cohens_d'].abs()

# Top features overall
print('Top 20 rolling features by |Cohen\'s d|:\n')
print(roll_df.sort_values('abs_d', ascending=False).head(20).to_string(index=False))

# Summary by stat type and window size
print('\n\nMean |Cohen\'s d| by stat type and window:\n')
summary = roll_df.groupby(['stat', 'window'])['abs_d'].agg(['mean', 'max']).round(3)
print(summary)

# Count features with meaningful separation by stat type and window
print('\n\nCount of features with |d| > 0.2 by stat type and window:\n')
counts = roll_df[roll_df['abs_d'] > 0.2].groupby(['stat', 'window']).size().unstack(fill_value=0)
print(counts)
```

**Finding:** Rolling statistics show three patterns:

1. **Rolling mean tracks raw values.** Throttle rmean (1.22-1.30) is slightly weaker than raw throttle (1.42). Forward acceleration rmean (1.18-1.20) is comparable to raw (1.14). Rolling mean does not add new information beyond what raw and lag features provide.

2. **Rolling std and range capture volatility, which is a separate signal.** Throttle volatility at window 7 reaches |d| = 0.74-0.79, independent of throttle level. More importantly, airspeed std and range show meaningful separation (0.37-0.43) at window 7 even though raw airspeed is weak (~0.13). Airspeed volatility is a new discovery from EDA: raw airspeed alone would have been dropped.

3. **Window 7 wins for volatility metrics.** Mean |d| for std and range grows monotonically with window size (w3: 0.12, w5: 0.17, w7: 0.21). Larger windows extract more signal without blurring class boundaries. Rolling mean is flat across windows.

**Decisions:**
- Drop rolling mean (verify with mutual information first).
- Keep rolling std and range at window 7 only.
- Drop windows 3 and 5 for std and range.


### Verify Rolling Statistics With Mutual Information

Cohen's d ranked rolling mean as redundant with raw values, while rolling std and range showed independent signal (especially at window 7). We re-run the same check using mutual information to confirm the rankings hold under a metric that captures non-linear and shape-based relationships. Same three statistics (mean, std, range) and same three windows (3, 5, 7) as the Cohen's d check.

```python
WINDOWS = [3, 5, 7, 10]
STATS = ['mean', 'std', 'range']

mi_roll_results = []
for col, label in zip(predictor_cols, short_labels):
    for w in WINDOWS:
        for stat in STATS:
            feat = f'{col}_r{stat}{w}'
            mi = mutual_info_single(train[feat], train['inefficient'])
            mi_roll_results.append((f'{label}_r{stat}{w}', stat, w, mi))

mi_roll_df = pd.DataFrame(mi_roll_results, columns=['feature', 'stat', 'window', 'mi'])

print('Top 20 rolling features by mutual information:\n')
print(mi_roll_df.sort_values('mi', ascending=False).head(20).to_string(index=False))

print('\n\nMean mutual information by stat type and window:\n')
mi_summary = mi_roll_df.groupby(['stat', 'window'])['mi'].agg(['mean', 'max']).round(4)
print(mi_summary)

print('\n\nCount of features with MI > 0.05 by stat type and window:\n')
mi_counts = mi_roll_df[mi_roll_df['mi'] > 0.05].groupby(['stat', 'window']).size().unstack(fill_value=0)
print(mi_counts)
```

### Test Relative Volatility (Short-Window Std / Long-Window Std)

The autocorrelation contamination problem: rolling range and rolling std grow mechanically with window size, partly because longer windows are more likely to span class transitions, not because longer windows capture more volatility signal.

We test a relative volatility feature: rolling standard deviation over a short window divided by rolling standard deviation over a long window. The denominator absorbs the "near a transition" effect because long windows almost always span transitions in this data; the ratio cancels it out. What survives is genuine local volatility bursts above the recent baseline.

Configuration:
- Short window: 5 rows
- Long window: 20 rows
- Test on throttle (`output[0]_f3`) and forward acceleration (`accelerometer_m_s2[0]_f104`) first

If this works, we expand to all 16 sensors. If not, we go with window size 5 for rolling std on its own and move on.

```python
SHORT_W = 5
LONG_W = 20
EPS = 1e-6  # floor for denominator to avoid div-by-near-zero spikes

test_sensors = ['output[0]_f3', 'accelerometer_m_s2[0]_f104']
test_labels = ['throttle', 'accel_fwd']

rel_vol_results = []

for col, label in zip(test_sensors, test_labels):
    shifted = train.groupby('flight_id')[col].shift(1)
    grouped = shifted.groupby(train['flight_id'])

    short_std = grouped.rolling(window=SHORT_W, min_periods=SHORT_W).std().reset_index(level=0, drop=True).sort_index()
    long_std = grouped.rolling(window=LONG_W, min_periods=LONG_W).std().reset_index(level=0, drop=True).sort_index()

    rel_vol = short_std / (long_std + EPS)
    train[f'{col}_relvol'] = rel_vol.values

    d = cohens_d(train[f'{col}_relvol'], train['inefficient'])
    mi = mutual_info_single(train[f'{col}_relvol'], train['inefficient'])

    raw_d = abs(cohens_d(train[col], train['inefficient']))
    raw_mi = mutual_info_single(train[col], train['inefficient'])

    rel_vol_results.append((label, raw_d, abs(d), raw_mi, mi))

print(f'{"Sensor":<12} {"raw |d|":>10} {"relvol |d|":>12} {"raw MI":>10} {"relvol MI":>12}')
for label, raw_d, rv_d, raw_mi, rv_mi in rel_vol_results:
    print(f'{label:<12} {raw_d:>10.3f} {rv_d:>12.3f} {raw_mi:>10.4f} {rv_mi:>12.4f}')
```

**Finding:** Tested relative volatility (short-window std / long-window std) to control for autocorrelation contamination in rolling stats. Both metrics collapsed on both sensors (throttle |d| 1.42 → 0.03, accel_fwd |d| 1.14 → 0.19; MI similarly near zero). The ratio removed the autocorrelation effect but also removed the signal. The rolling std/range increases we saw at larger windows were largely autocorrelation contamination, not genuine local volatility above baseline.

**Decisions for final rolling features:**
- Drop rolling mean (redundant with raw and lag features).
- Drop rolling range (mathematically too similar to std for this data, same contamination issue).
- Keep rolling std at window 5 only (16 features). Defensible: short enough to fit within typical class segments (avg 10-20 rows), shows meaningful but not inflated separation under both Cohen's d (max 0.61) and MI.


### Build Interaction Features and Test Class Separation

Two interaction features from the Phase 3 finding that sensor relationships shift between classes:

- **throttle × forward acceleration:** captures the relationship that *decouples* during inefficient moments (correlation drops from +0.58 to +0.23).
- **airspeed × elevator:** captures the relationship that *emerges* during inefficient moments (correlation goes from ~0 to -0.27).

Each interaction is the row-wise product of the two sensors. Tested with Cohen's d, verified with mutual information.

```python
# Build the two interaction features
train['throttle_x_accel_fwd'] = train['output[0]_f3'] * train['accelerometer_m_s2[0]_f104']
train['airspeed_x_elevator'] = train['indicated_airspeed_m_s_f5'] * train['output[2]_f3']

# Test both interactions
interactions = ['throttle_x_accel_fwd', 'airspeed_x_elevator']

print(f'{"Feature":<25} {"Cohen\'s d":>12} {"|d|":>8} {"MI":>10}')
for feat in interactions:
    d = cohens_d(train[feat], train['inefficient'])
    mi = mutual_info_single(train[feat], train['inefficient'])
    print(f'{feat:<25} {d:>+12.3f} {abs(d):>8.3f} {mi:>10.4f}')

# Compare against the raw components for context
print('\nFor reference, raw components:')
print(f'{"Sensor":<25} {"Cohen\'s d":>12} {"|d|":>8} {"MI":>10}')
for col, label in [('output[0]_f3', 'throttle'), 
                    ('accelerometer_m_s2[0]_f104', 'accel_fwd'),
                    ('indicated_airspeed_m_s_f5', 'airspeed_ind'),
                    ('output[2]_f3', 'elevator')]:
    d = cohens_d(train[col], train['inefficient'])
    mi = mutual_info_single(train[col], train['inefficient'])
    print(f'{label:<25} {d:>+12.3f} {abs(d):>8.3f} {mi:>10.4f}')
```

**Finding:** Simple product interactions did not capture the conditional correlation structure from Phase 3:
- `throttle × accel_fwd`: |d| = 1.15, MI = 0.14. Strong, but no better than raw accel_fwd alone.
- `airspeed × elevator`: |d| = 0.007, MI = 0.009. No signal.

A row-wise product captures magnitude, not relationship shift. The Phase 3 finding was about how correlations between sensors change with class, which requires a different feature construction (e.g., residual interactions). Out of scope for this notebook.

**Decision: drop both interaction features.** Neither earns its keep.


### Phase 5 Summary

Tested seven feature families with Cohen's d, verified key drops with mutual information.

**Final feature set for modeling:**
- Raw sensors (16) plus `time_since_start_of_flight` (1)
- Lag features at k=1 and k=2 for each sensor (32)
- Rolling std at window 5 for each sensor (16)

Total: 65 features.

**Dropped families and why:**
- First differences: Cohen's d max 0.11, MI max 0.04. Classes are characterized by sensor levels, not single-timestep change rates.
- Second differences: Cohen's d max 0.07. Noise amplification.
- Rolling mean: tracks raw and lag features without adding information.
- Rolling range, rolling std at windows 7 and 10: signal grew mechanically with window size. Relative volatility test confirmed the larger-window signal was autocorrelation contamination from the sticky target, not genuine local volatility.
- Interaction products (throttle × accel_fwd, airspeed × elevator): the first was redundant with raw components, the second had no signal. Simple products don't capture the conditional correlation shifts seen in Phase 3.

**Lag features caveat:** Cohen's d ranked lag features below raw values because both metrics are univariate. Lag features earn their value through interactions with raw values inside the model. Re-evaluate using model-based feature importance during the modeling phase before dropping any.

**Key analytical lesson:** When a metric grows monotonically with a parameter (window size), the metric may be measuring the parameter, not the signal. Using both Cohen's d and mutual information caught patterns either metric alone would have missed.


### Final Cleanup: Drop Features Not in the Final Set

We built many features during Phase 5 to test class separation. Now we drop everything that didn't make the cut. Final engineered feature set: 65 features (1 baseline + 32 lag + 16 rolling std at window 5 + 16 raw sensors already in dataset).

```python
# Identify columns to drop
cols_to_drop = []

# Drop all first differences
cols_to_drop.extend([c for c in train.columns if c.endswith('_diff1')])

# Drop all second differences
cols_to_drop.extend([c for c in train.columns if c.endswith('_diff2')])

# Drop all rolling means
cols_to_drop.extend([c for c in train.columns if '_rmean' in c])

# Drop all rolling ranges
cols_to_drop.extend([c for c in train.columns if '_rrange' in c])

# Drop rolling std at windows other than 5
cols_to_drop.extend([c for c in train.columns if '_rstd3' in c or '_rstd7' in c or '_rstd10' in c])

# Drop relative volatility test features
cols_to_drop.extend([c for c in train.columns if c.endswith('_relvol')])

# Drop interaction features
cols_to_drop.extend(['throttle_x_accel_fwd', 'airspeed_x_elevator'])

# Keep only existing columns (in case any are missing)
cols_to_drop = [c for c in cols_to_drop if c in train.columns]

print(f'Columns to drop: {len(cols_to_drop)}')

train = train.drop(columns=cols_to_drop)
train = train.copy()  # defragment

print(f'Columns remaining: {len(train.columns)}')

# Verify the engineered features we kept
engineered = [c for c in train.columns if any(s in c for s in ['_lag1', '_lag2', '_rstd5', 'time_since_start_of_flight'])]
print(f'\nEngineered features kept: {len(engineered)}')
```

### Drop Unused Columns

The original training set has thousands of raw sensor columns we never used. We keep only what's needed for modeling: flight ID, target, 16 predictors, and 49 engineered features.


```python
keep_cols = (
    ['flight_id', 'inefficient'] +
    predictor_cols +
    [f'{c}_lag1' for c in predictor_cols] +
    [f'{c}_lag2' for c in predictor_cols] +
    [f'{c}_rstd5' for c in predictor_cols] +
    ['time_since_start_of_flight']
)

print(f'Columns to keep: {len(keep_cols)}')

train = train[keep_cols].copy()

print(f'Final column count: {len(train.columns)}')
print(f'Final row count: {len(train)}')
```

### Feature Correlation With Target

We compute the Pearson correlation between every feature and the target. Engineered features should ideally show correlation comparable to or stronger than their raw counterparts.

```python
# All features (excluding flight_id and target)
all_features = [c for c in train.columns if c not in ['flight_id', 'inefficient']]

# Correlation with target
target_corr = train[all_features].corrwith(train['inefficient']).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(7, 14))
colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(target_corr)))
ax.barh(range(len(target_corr)), target_corr.values, color=colors)
ax.set_yticks(range(len(target_corr)))
ax.set_yticklabels(target_corr.index, fontsize=7)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('Pearson correlation with target')
ax.set_title('Feature Correlation with `inefficient` (sorted by |r|)')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

print(f'\nTop 15 features by |correlation with target|:\n')
print(target_corr.head(15).to_string())
print(f'\nFeatures with |r| > 0.2: {(target_corr.abs() > 0.2).sum()} of {len(target_corr)}')
```

**Finding:** The final feature set is dominated by throttle and forward acceleration (raw and lags), consistent with the rankings from notebook 02 and the Cohen's d analysis.

Three engineered features stand out beyond the dominant raw signals:
- `output[0]_f3_rstd5` (throttle volatility, window 5): negative correlation. Throttle variation decreases during inefficient moments, consistent with sustained high-throttle climbs.
- `indicated_airspeed_m_s_f5_rstd5` and `true_airspeed_m_s_f5_rstd5` (airspeed volatility): negative correlation. Airspeed level alone showed near-zero correlation in notebook 02; volatility is a discovery from EDA.
- `time_since_start_of_flight`: negative correlation, confirming inefficient moments occur earlier in flights on average.

Lag features sit just below their raw counterparts (e.g., throttle: raw > lag1 > lag2). This is expected: lag features add value through interaction with raw values in the model, not through marginal correlation.


### Engineered Features Correlation Heatmap

Now we check correlations *between* the engineered features (lags and rolling std) to see if they duplicate each other. High inter-correlation means redundancy and potential multicollinearity for Logistic Regression.

```python
# Engineered features only (no raw sensors, no time_since_start_of_flight)
engineered = [c for c in train.columns if '_lag1' in c or '_lag2' in c or '_rstd5' in c]

corr = train[engineered].corr()

# Short labels for plotting
def short_name(col):
    name = col
    name = name.replace('gyro_rad[0]_f104', 'gyro_roll')
    name = name.replace('gyro_rad[1]_f104', 'gyro_pitch')
    name = name.replace('gyro_rad[2]_f104', 'gyro_yaw')
    name = name.replace('accelerometer_m_s2[0]_f104', 'accel_fwd')
    name = name.replace('accelerometer_m_s2[1]_f104', 'accel_side')
    name = name.replace('accelerometer_m_s2[2]_f104', 'accel_up')
    name = name.replace('indicated_airspeed_m_s_f5', 'airspeed_ind')
    name = name.replace('true_airspeed_m_s_f5', 'airspeed_true')
    name = name.replace('q[0]_f119', 'q_w')
    name = name.replace('q[1]_f119', 'q_x')
    name = name.replace('q[2]_f119', 'q_y')
    name = name.replace('q[3]_f119', 'q_z')
    name = name.replace('output[0]_f3', 'throttle')
    name = name.replace('output[1]_f3', 'aileron')
    name = name.replace('output[2]_f3', 'elevator')
    name = name.replace('output[3]_f3', 'rudder')
    return name

short = [short_name(c) for c in engineered]

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap='cividis', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(short)))
ax.set_yticks(range(len(short)))
ax.set_xticklabels(short, rotation=90, fontsize=6)
ax.set_yticklabels(short, fontsize=6)

ax.set_title('Engineered Features Correlation Heatmap')
plt.colorbar(im, ax=ax, label='Pearson correlation')
plt.tight_layout()
plt.show()

# Identify highly correlated engineered feature pairs
print('\nEngineered feature pairs with |r| > 0.9 (likely redundant):\n')
high_corr_pairs = []
for i in range(len(engineered)):
    for j in range(i + 1, len(engineered)):
        r = corr.iloc[i, j]
        if abs(r) > 0.9:
            high_corr_pairs.append((short[i], short[j], r))

high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for n1, n2, r in high_corr_pairs[:30]:
    print(f'  {n1:<30} <-> {n2:<30} {r:+.3f}')

print(f'\nTotal pairs with |r| > 0.9: {len(high_corr_pairs)}')

```

**Finding:** 13 engineered feature pairs show |r| > 0.9. Three patterns:

1. **Airspeed duplicates (4 pairs):** `indicated_airspeed` and `true_airspeed` are nearly identical at every lag and rolling stat (r up to 1.000). Same physical quantity from two readings.
2. **Lag1 vs lag2 of smooth signals (8 pairs):** quaternions, throttle, rudder, and airspeed show lag1 ↔ lag2 correlation above 0.9. Expected from Phase 2 ACF analysis: very smooth signals barely change row to row, so consecutive lags carry near-identical information.
3. **Cross-lag airspeed pairs:** subset of the airspeed duplication.

**Decision: keep all features as-is.** Tree-based models handle redundancy through feature importance, Logistic Regression handles it through regularization, and GAT may benefit from the redundant connections as edge structure. Pre-pruning at EDA removes optionality the models may exploit. Redundancy is documented here for the modeling phase to address through model-specific feature selection.


### Apply Feature Engineering Function to Train and Test

Define one function that takes a raw dataframe and produces the final 67-column engineered dataset. Apply it to both train and test so they have identical structure. The function does the same operations we did manually throughout the notebook, just packaged for reuse.

```python
def engineer_features(df, predictor_cols):
    """
    Apply the temporal feature engineering pipeline to a raw dataframe.
    
    Adds: time_since_start_of_flight, lag1, lag2, and rolling std at window 5
    for each predictor. All transformations are computed within flight_id groups
    to prevent leakage across flights.
    
    Returns a new dataframe with only the final 67 columns.
    """
    df = df.copy()
    
    # time_since_start_of_flight
    df['time_since_start_of_flight'] = df.groupby('flight_id').cumcount()
    
    # Build lag and rolling std features in a dict to avoid fragmentation
    new_features = {}
    
    for col in predictor_cols:
        # Lag features
        new_features[f'{col}_lag1'] = df.groupby('flight_id')[col].shift(1).values
        new_features[f'{col}_lag2'] = df.groupby('flight_id')[col].shift(2).values
        
        # Rolling std at window 5 (shift first to prevent current-row leakage)
        shifted = df.groupby('flight_id')[col].shift(1)
        rolling = shifted.groupby(df['flight_id']).rolling(window=5, min_periods=5).std()
        new_features[f'{col}_rstd5'] = rolling.reset_index(level=0, drop=True).sort_index().values
    
    # Add all new features in one operation
    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    
    # Keep only the final 67 columns
    keep_cols = (
        ['flight_id', 'inefficient'] +
        predictor_cols +
        [f'{c}_lag1' for c in predictor_cols] +
        [f'{c}_lag2' for c in predictor_cols] +
        [f'{c}_rstd5' for c in predictor_cols] +
        ['time_since_start_of_flight']
    )
    
    return df[keep_cols].copy()


# Apply to train (rebuild from source to confirm function works end-to-end)
train_raw = pd.read_csv('../../data/splits/train.csv')
train = engineer_features(train_raw, predictor_cols)
print(f'Train: {len(train)} rows, {len(train.columns)} columns')

# Apply to test
test_raw = pd.read_csv('../../data/splits/test.csv')
test = engineer_features(test_raw, predictor_cols)
print(f'Test: {len(test)} rows, {len(test.columns)} columns')

# Verify train and test have identical column structure
assert list(train.columns) == list(test.columns), 'Column mismatch between train and test'
print('Train and test column structure match')

# Save both
train.to_csv('../../data/splits/train.csv', index=False)
test.to_csv('../../data/splits/test.csv', index=False)
print('\nSaved engineered train and test to data/splits/')
```

## Notebook Summary

**Goal:** Build and validate temporal features for the UAV efficiency classification task. Produce a clean train/test dataset ready for modeling.

**Final feature set: 67 columns**
- `flight_id`, `inefficient` (target)
- 16 raw sensors
- 32 lag features (lag 1 and lag 2 per sensor)
- 16 rolling standard deviation features (window 5 per sensor)
- `time_since_start_of_flight`

**Key analytical decisions:**

- **Autoregressive (AR) order across sensors ranges from 1 to 3.** Phase 2 Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) analysis on representative signals supported a uniform lag depth of k=2, confirmed by Phase 4 cross-correlation. AR order describes how many previous timesteps a sensor's current value directly depends on.
- **First and second differences dropped.** Both metrics (Cohen's d and mutual information) showed minimal class separation. Classes are characterized by absolute sensor levels, not single-timestep change rates.
- **Rolling mean and rolling range dropped.** Mean tracked raw values without adding information. Range and std grew mechanically with window size; relative volatility test confirmed the larger-window signal was autocorrelation contamination from the sticky target.
- **Rolling std at window 5 retained.** Small but real signal across multiple sensors. Window 5 stays within typical class segment length to avoid contamination.
- **Interaction features dropped.** Simple products did not capture the conditional correlation shifts seen in Phase 3.
- **Redundancy kept.** 13 engineered feature pairs show |r| > 0.9 (airspeed duplicates and smooth-signal lag pairs). Documented for the modeling phase to handle through model-specific feature selection.

**For modeling:**
- Persistence baseline (predict `target(t) = target(t-1)`) is the floor. Phase 2 ACF lag 1 = 0.838 on the target.
- Lag features ranked lower than raw values under marginal metrics. Re-evaluate using model-based feature importance before dropping.
- Cross-validation must be at the flight level. Splits and folds defined in notebook 02.
