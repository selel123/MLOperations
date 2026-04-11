# MLOps Project

A structured MLOps project covering data processing, model training, serving, batch prediction, monitoring, and retraining.

## Project Structure

```
mlops-project/
├── data/
│   ├── raw/          # Raw, immutable data
│   └── processed/    # Cleaned and transformed data
├── notebooks/
│   └── 01_eda.ipynb  # Exploratory Data Analysis
├── src/
│   ├── data/         # Data loading and preprocessing
│   ├── features/     # Feature engineering
│   ├── training/     # Model training logic
│   └── inference/    # Inference/prediction logic
├── reports/          # Generated reports and metrics
├── logs/             # Application and training logs
├── train.py          # Entry point: train a model
├── serve.py          # Entry point: serve model via API
├── predict_batch.py  # Entry point: batch predictions
├── monitor.py        # Entry point: monitor model performance
└── retrain.py        # Entry point: trigger retraining
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train
python train.py

# Serve
python serve.py

# Batch prediction
python predict_batch.py

# Monitor
python monitor.py

# Retrain
python retrain.py
```
