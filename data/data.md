# Data

Flight telemetry data from the IDF-DS dataset (Garcia-Gascon et al., 2026). Publicly available on [Zenodo](https://doi.org/10.5281/zenodo.16992975).

## Hardware Configurations

The dataset includes two hardware configurations on the same airframe (Volantex Ranger 2400):

- **Holybro Pixhawk** — heavier setup (2,160 g) with higher-end sensors and a Jetson companion computer. Contains both grouped and ungrouped flight data.
  - **Grouped**: 120 flights downsampled to approximately 1.4 Hz, synchronized across sensors into one CSV per flight.
  - **Ungrouped**: native sensor rates (up to 400 Hz for IMU, 10 Hz for GNSS), one CSV per sensor per flight. Not yet assembled.
- **SpeedyBee** — lighter setup (1,590 g) with lower-end sensors.

**Currently in use:** Holybro Pixhawk, grouped flights only.

## Directory Layout

- `raw/` — original zip archives (both configurations)
- `processed/pixhawk_grouped_flights/` — 120 per-flight grouped CSVs
- `splits/` — stratified 85/15 train/test splits (94/17 flights after exclusions)

## Excluded Flights

Nine flights were excluded during EDA:

| Flights | Reason |
|---|---|
| 7, 8, 9 | Missing key sensor groups (different column structure) |
| 16 | Only 14 rows (likely a startup or abort log) |
| 113 | Only 2 rows |
| 117, 118, 119, 120 | Missing sensor groups and uncalibrated Pitot data (flagged in dataset README) |

Usable flights: 111 (approximately 56,766 rows total).
