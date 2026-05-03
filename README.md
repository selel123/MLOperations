# MLOps Project
A structured MLOps pipeline for risk flag classification, covering data processing, model training, serving, batch prediction, monitoring, and retraining.

# Project Structure

MLOps/
├── data/
│   ├── raw/                    # Raw, immutable data (CSV waves)
│   └── processed/              # Cleaned and transformed data
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── training/
│   │   ├── train.py            # Model training logic
│   │   └── retrain.py          # Retraining logic
│   └── inference/
│       ├── serve.py            # FastAPI serving logic
│       └── predict_batch.py    # Batch prediction logic
├── reports/
│   ├── model_card.py           # Model card generation
│   ├── monitoring_retrain/     # Monitoring plots & retraining reports
│   └── docs/                   # PDF reports and feature importance
├── logs/
│   └── inference_log.csv       # Inference log
├── mlruns/                     # MLflow experiment tracking
├── mlflow.db                   # MLflow backend store
├── monitor.py                  # Entry point: monitor model performance
└── pyproject.toml              # Project dependencies
Setup

# Create and activate virtual environment
uv sync

# Or with pip
pip install -e .
Usage

# Train baseline model
python src/training/train.py

# Serve model via FastAPI
python src/inference/serve.py

# Run batch predictions
python src/inference/predict_batch.py

# Monitor model performance (PSI, drift, metrics)
python monitor.py

# Trigger retraining
python src/training/retrain.py
MLflow Tracking
Experiments and model runs are tracked with MLflow. To launch the UI locally:


mlflow ui --backend-store-uri sqlite:///mlflow.db

# Tech Stack
ML: scikit-learn, LightGBM, XGBoost
Tracking: MLflow
Serving: FastAPI, Uvicorn
Data: pandas, NumPy
Analysis: matplotlib, seaborn, statsmodels
