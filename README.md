# MLOps Project

A structured MLOps pipeline for risk flag classification, covering:

- Data processing
- Model training
- Model serving
- Batch prediction
- Monitoring
- Retraining

---

# Project Structure

```text
MLOps/
├── data/
│   ├── raw/                         # Raw, immutable data (CSV waves)
│   └── processed/                   # Cleaned and transformed data
│
├── notebooks/
│   └── 01_eda.ipynb                 # Exploratory Data Analysis
│
├── src/
│   ├── training/
│   │   ├── train.py                 # Model training logic
│   │   └── retrain.py               # Retraining logic
│   │
│   └── inference/
│       ├── serve.py                 # FastAPI serving logic
│       └── predict_batch.py         # Batch prediction logic
│
├── reports/
│   ├── model_card.py                # Model card generation
│   ├── monitoring_retrain/          # Monitoring plots & retraining reports
│   └── docs/                        # PDF reports and feature importance
│
├── logs/
│   └── inference_log.csv            # Inference log
│
├── mlruns/                          # MLflow experiment tracking
├── mlflow.db                        # MLflow backend store
├── monitor.py                       # Monitoring entry point
└── pyproject.toml                   # Project dependencies
```

---

# Setup

## Create and activate the environment

Using `uv`:

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

---

# Usage

## Train baseline model

```bash
python src/training/train.py
```

## Serve model via FastAPI

```bash
python src/inference/serve.py
```

## Run batch predictions

```bash
python src/inference/predict_batch.py
```

## Monitor model performance

Includes:
- PSI monitoring
- Data drift detection
- Metric tracking

```bash
python monitor.py -- wave
```

## Retraining is triggered automatically by monitor pipeline

```bash
python src/training/retrain.py --wave
```

---

# MLflow Tracking

Experiments and model runs are tracked with MLflow.

Link to UI:
- http://46.225.163.17:5000 

---

# Tech Stack

## Machine Learning
- scikit-learn
- LightGBM
- XGBoost

## Experiment Tracking
- MLflow

## Model Serving
- FastAPI
- Uvicorn

## Data Processing
- pandas
- NumPy

## Analysis & Visualization
- matplotlib
- seaborn
- statsmodels
