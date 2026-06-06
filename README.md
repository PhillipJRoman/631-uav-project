# DSCI 631 Term Project: UAV Energy Efficiency Classification

Binary classification of UAV flight efficiency from onboard sensor telemetry. Term project for DSCI 631: Applied Machine Learning, Spring 2026, at Drexel University.

## Team

- Phillip Roman (PhillipJRoman)
- Ryan Quinlan (rqhq)

### Contributions

**Both team members:**
- Exploratory data analysis
- `04_eda_environmental` notebook
- `09_umap_embeddings` notebook

**Phillip Roman:**
- `01_eda_pixhawk_grouped` and `02_eda_bin_class_grouped` notebooks
- `07_gat` (Graph Attention Network) notebook
- `08_hybrid` (GAT + XGBoost) notebook

**Ryan Quinlan:**
- `03_eda_temporal` notebook and temporal feature engineering
- `05_model_baselines` notebook (persistence, logistic regression, LDA)
- `06_xgboost` notebook

## Dataset and Target Definition

The Holybro Pixhawk portion of the IDF-DS dataset (García-Gascón et al., 2026), publicly available on [Zenodo](https://doi.org/10.5281/zenodo.16992975). 120 autonomous fixed-wing UAV flights over a repeatable waypoint circuit in Spain. After pruning problematic flights, 111 flights were used for modeling.

The target label and the 0.1 m/W efficiency threshold come directly from the dataset paper. Instantaneous efficiency is defined as:

```
eta(t) = v_ground(t) / P(t)
```

where `v_ground(t)` is the magnitude of the ground velocity vector and `P(t) = V(t) x I(t)` is instantaneous power from voltage and current. Per-row labels: inefficient (1) if `eta(t) < 0.1 m/W`, efficient (0) otherwise.

## Approach

Five models were compared on a flight-level train/test split (94 train, 17 test):

1. Persistence (naive baseline)
2. Logistic Regression (Lasso and Elastic Net) and Linear Discriminant Analysis
3. XGBoost
4. Graph Attention Network (GATv2)
5. GAT embeddings + XGBoost hybrid

Primary metric: F1 score. Secondary: ROC AUC.

## Results

| Model | F1 | ROC AUC |
|---|---|---|
| Persistence (naive) | 0.9105 | 0.9261 |
| XGBoost | 0.8692 | 0.9634 |
| Hybrid (GAT + XGBoost) | 0.8666 | 0.9642 |
| Lasso Logistic Regression | 0.8182 | 0.9239 |
| Elastic Net Logistic Regression | 0.8155 | 0.9229 |
| Linear Discriminant Analysis | 0.7956 | 0.9097 |
| Graph Attention Network | 0.7241 | 0.8556 |

The persistence F1 is inflated by target autocorrelation (lag-1 = 0.838); ROC AUC is the more honest comparison. XGBoost on the original engineered features sets the strongest benchmark. The GAT learned physically meaningful sensor relationships but did not translate them into better predictions on a small 16-node graph. The hybrid effectively tied XGBoost alone, indicating that the GAT embeddings were redundant with information already extractable from the raw features.

## Setup

Requires [uv](https://astral.sh/uv/) and [DVC](https://dvc.org/).

```console
$ uv sync
$ . ./.venv/bin/activate
$ dvc pull
```

Data is stored in Google Cloud Storage via DVC. Pulling requires GCP credentials for the `dsci631-uav-data` bucket. Raw data is publicly available at the Zenodo link above.

## Directory Layout

- `data/` — raw and processed flight data, tracked with DVC
  - `raw/` — original Pixhawk and SpeedyBee zip archives
  - `processed/pixhawk_grouped_flights/` — per-flight grouped CSVs
  - `splits/` — stratified train and test splits
- `notebooks/eda/` — exploratory analysis (notebooks 01–04)
- `notebooks/models/` — modeling notebooks (05–08)
- `notebooks/analysis/` — UMAP embedding analysis (09)
- `models/` — saved model artifacts
- `results/` — metrics, plots, and per-notebook output files
- `src/` — project source code
- `pyproject.toml` — dependencies and project metadata

Notebooks are stored as `.md` via jupytext for clean diffs; `.ipynb` versions are kept in sync.

## References

- **Original GAT paper, 2018.** Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y. "Graph Attention Networks." ICLR 2018. arXiv:1710.10903
- **GATv2 paper, 2022.** Brody, S., Alon, U., Yahav, E. "How Attentive are Graph Attention Networks?" ICLR 2022. arXiv:2105.14491
- **Hands-On Graph Neural Networks Using Python, 2023.** Labonne, M. Packt Publishing.
- **DyGAT-FTNet paper, 2025.** Duan, H., Chen, G., Yu, Y., Du, C., Bao, Z., Ma, D. "DyGAT-FTNet: A Dynamic Graph Attention Network for Multi-Sensor Fault Diagnosis and Time-Frequency Data Fusion." Sensors 25(3), 810.
- **Dataset paper.** García-Gascón, C., Bas-Bolufer, J., Castelló-Pedrero, P., García-Manrique, J. A. "An open benchmark dataset for machine learning and intelligent trajectory optimization in fixed-wing unmanned aerial systems." Scientific Data, 2026.

## License

This project was developed for educational purposes as part of DSCI 631 coursework at Drexel University.

Data sourced from García-Gascón et al. (2026), distributed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License via [Zenodo](https://doi.org/10.5281/zenodo.16992975).
