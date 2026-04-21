# DSCI 631 Term Project: UAV Energy Efficiency Classification

Binary classification of UAV flight efficiency from onboard sensor telemetry. Term project for DSCI 631 Applied Machine Learning at Drexel University.

## Team

- Phillip Roman (PhillipJRoman)
- Ryan Quinlan (rqhq)

## Dataset

Holybro Pixhawk portion of the IDF-DS dataset (Garcia-Gascon et al., 2026), publicly available on [Zenodo](https://doi.org/10.5281/zenodo.16992975). 120 autonomous fixed-wing UAV flights over a repeatable waypoint circuit in Spain.

## Setup

Requires [uv](https://astral.sh/uv/) and [DVC](https://dvc.org/).

```console
$ uv sync
$ . ./.venv/bin/activate
$ dvc pull
```

Data is stored in Google Cloud Storage via DVC. Pulling requires GCP credentials for the `dsci631-uav-data` bucket. Raw data is publicly available at the Zenodo link above; for DVC-pull access to preprocessed artifacts, contact Phillip.

## Directory Layout

- `data/` — raw and processed flight data, tracked with DVC
  - `raw/` — original Pixhawk and SpeedyBee zip archives
  - `processed/pixhawk_grouped_flights/` — per-flight grouped CSVs (120 flights)
  - `splits/` — stratified train and test splits
- `notebooks/eda/` — exploratory analysis. Notebooks are stored as `.md` via jupytext for clean diffs; `.ipynb` versions are kept in sync.
- `src/` — project source code
- `pyproject.toml` — dependencies and project metadata
