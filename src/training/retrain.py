"""
Retraining Pipeline – LGBM-RiskClassification

Ablauf:
    1. Trainingsdaten + neue Production Waves zusammenführen
    2. LGBM neu trainieren
    3. Als neuen Run in MLflow loggen
    4. In Model Registry als @staging registrieren
    Die Promotion zu @production erfolgt manuell (Human in the Loop)

Usage:
    python src/training/retrain.py --waves data/raw/production_wave_2.csv data/raw/production_wave_3.csv
"""

import argparse
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
TRACKING_URI    = "http://46.225.163.17:5000"
EXPERIMENT_NAME = "classification-ml"
MODEL_NAME      = "LGBM-RiskClassification"
TRAIN_PATH      = Path("data/raw/train_full.csv")

LOG_FEATURES        = ["transaction_volume", "processing_time_hours",
                       "historical_incidents_90d", "change_requests_30d"]
PLAIN_FEATURES      = ["open_cases_count", "customer_tenure_months",
                       "missing_docs_flag", "high_priority_source_flag"]
CATEGORICAL_FEATURES = ["region", "channel", "customer_segment", "product_line"]
ALL_FEATURES        = LOG_FEATURES + PLAIN_FEATURES + CATEGORICAL_FEATURES
TARGET              = "risk_flag"


# ---------------------------------------------------------------------------
# Preprocessing Pipeline (identisch zu train.py)
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    log_pipe   = Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("log", FunctionTransformer(np.log1p)),
                           ("scl", StandardScaler())])
    plain_pipe  = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("scl", StandardScaler())])
    cat_pipe    = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([
        ("log",   log_pipe,   LOG_FEATURES),
        ("plain", plain_pipe, PLAIN_FEATURES),
        ("cat",   cat_pipe,   CATEGORICAL_FEATURES),
    ])


# ---------------------------------------------------------------------------
# Hauptpipeline
# ---------------------------------------------------------------------------
def run_retraining(wave_paths: list[Path]):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Daten zusammenführen: train_full + alle neuen Waves
    dfs = [pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])]
    for p in wave_paths:
        dfs.append(pd.read_csv(p, parse_dates=["timestamp"]))
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")

    # Zeitbasierter Split: letzte 15% als Validation
    split_idx = int(len(df) * 0.85)
    train_df  = df.iloc[:split_idx]
    val_df    = df.iloc[split_idx:]

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
    X_val,   y_val   = val_df[ALL_FEATURES],   val_df[TARGET]

    print(f"Train: {len(X_train):,} Zeilen | Validation: {len(X_val):,} Zeilen")

    # 2. Modell trainieren
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   LGBMClassifier(is_unbalance=True, random_state=42,
                                        n_jobs=1, verbose=-1)),
    ])
    model.fit(X_train, y_train)

    # 3. Metriken berechnen
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    metrics = {
        "retrain_f1":      round(f1_score(y_val, y_pred, zero_division=0), 4),
        "retrain_roc_auc": round(roc_auc_score(y_val, y_prob), 4),
        "train_rows":      len(X_train),
        "val_rows":        len(X_val),
        "wave_count":      len(wave_paths),
    }
    print(f"F1={metrics['retrain_f1']}  AUC={metrics['retrain_roc_auc']}")

    # 4. In MLflow loggen & als @staging registrieren
    with mlflow.start_run(run_name="LGBM_retrained") as run:
        mlflow.log_param("waves", [str(p) for p in wave_paths])
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:3],
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    # Alias @staging setzen
    client  = mlflow.MlflowClient()
    version = client.get_model_version_by_run_id(MODEL_NAME, run_id)
    client.set_registered_model_alias(MODEL_NAME, "staging", version.version)

    print(f"\nModell registriert: {MODEL_NAME} v{version.version} @staging")
    print("Promotion zu @production erfolgt manuell im MLflow UI.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining – LGBM-RiskClassification")
    parser.add_argument("--waves", nargs="+", type=Path, required=True,
                        help="Neue Production-Wave CSV-Dateien")
    args = parser.parse_args()
    run_retraining(args.waves)
