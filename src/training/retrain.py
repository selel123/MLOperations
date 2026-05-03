"""
Retraining Pipeline – LGBM-RiskClassification

Ablauf:
    1. Trainingsdaten + neue Production Waves laden und zusammenführen
    2. Champion-Metriken des aktuellen Produktionsmodells aus MLflow abrufen
    3. LGBM neu trainieren und als Run in MLflow loggen
    4. Promotion-Entscheidung anhand PROMOTION_POLICY (automatisch)
    5. Bei Erfolg: Challenger → @staging → @production, altes Modell → @archived
    6. Retraining-Report als JSON in reports/ speichern

Verwendung:
    python src/training/retrain.py --waves data/raw/production_wave_2.csv data/raw/production_wave_3.csv
    python src/training/retrain.py --waves data/raw/production_wave_2.csv --no-promote
"""

import argparse
import logging
import json
import sys
import re
import warnings
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Pfade
TRAIN_PATH = Path("data/raw/train_full.csv")
REPORTS_DIR = Path("reports")

TRACKING_URI    = "http://localhost:5000"
EXPERIMENT_NAME = "classification-ml"
MODEL_NAME      = "LGBM-RiskClassification"

# Feature-Definitionen (konsistent mit train.py)
TARGET = "risk_flag"

LOG_FEATURES = [
    "transaction_volume",
    "processing_time_hours",
    "historical_incidents_90d",
    "change_requests_30d",
]
PLAIN_FEATURES = [
    "open_cases_count",
    "customer_tenure_months",
    "missing_docs_flag",
    "high_priority_source_flag",
]
CATEGORICAL_FEATURES = [
    "region", 
    "channel", 
    "customer_segment", 
    "product_line"
]

ALL_FEATURES = LOG_FEATURES + PLAIN_FEATURES + CATEGORICAL_FEATURES

# Capping auf 99. Perzentil (echte Ausreißer in diesen Features, EDA-basiert)
CAPPING_FEATURES = ["historical_incidents_90d", "processing_time_hours", "transaction_volume"]

# Promotion Policy – Schwellenwerte
PROMOTION_POLICY = {
    "min_f1_improvement": -0.01,   # Neues Modell darf max. 1% schlechter sein
    "min_precision": 0.50,         # Mindest-Precision
    "min_recall": 0.55,            # Mindest-Recall
    "min_roc_auc": 0.68,           # Mindest-AUC
}


# ---------------------------------------------------------------------------
# Preprocessing Pipeline (identisch zu train.py)
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Baut den ColumnTransformer – konsistent mit train.py."""

    log_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p",   FunctionTransformer(np.log1p)),
        ("scaler",  StandardScaler()),
    ])
    plain_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("log",   log_transformer,         LOG_FEATURES),
        ("plain", plain_transformer,       PLAIN_FEATURES),
        ("cat",   categorical_transformer, CATEGORICAL_FEATURES),
        
    ], remainder="drop")

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Berechnet alle relevanten Klassifikationsmetriken."""
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

def get_production_model_metrics(client: MlflowClient) -> dict | None:
    """Holt die Metriken des aktuellen Produktionsmodells aus der MLflow Registry."""
    try:
        prod_versions = client.get_model_version_by_alias(MODEL_NAME, "production")
        run_id = prod_versions.run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics
        logger.info(
            f"Produktionsmodell gefunden: Version {prod_versions.version}, "
            f"Run-ID {run_id}"
        )
        return {
            "version": prod_versions.version,
            "run_id": run_id,
            "f1": metrics.get("test_f1"),
            "precision": metrics.get("test_precision"),
            "recall": metrics.get("test_recall"),
            "roc_auc": metrics.get("test_roc_auc"),
        }
    except Exception as e:
        logger.warning(f"Kein Produktionsmodell gefunden: {e}")
        return None


def evaluate_promotion(
    challenger_metrics: dict,
    champion_metrics: dict | None,
) -> tuple[bool, str]:
    """
    Entscheidet anhand der Promotion Policy, ob der Challenger promoted wird.

    Returns:
        (should_promote, reason)
    """
    reasons = []

    # Absolute Mindestanforderungen prüfen
    if challenger_metrics["precision"] < PROMOTION_POLICY["min_precision"]:
        reasons.append(
            f"Precision {challenger_metrics['precision']:.3f} < "
            f"Minimum {PROMOTION_POLICY['min_precision']}"
        )
    if challenger_metrics["recall"] < PROMOTION_POLICY["min_recall"]:
        reasons.append(
            f"Recall {challenger_metrics['recall']:.3f} < "
            f"Minimum {PROMOTION_POLICY['min_recall']}"
        )
    if challenger_metrics["roc_auc"] < PROMOTION_POLICY["min_roc_auc"]:
        reasons.append(
            f"ROC-AUC {challenger_metrics['roc_auc']:.3f} < "
            f"Minimum {PROMOTION_POLICY['min_roc_auc']}"
        )

    if reasons:
        return False, "Mindestanforderungen nicht erfüllt: " + " | ".join(reasons)

    # Vergleich mit Champion (falls vorhanden)
    if champion_metrics and champion_metrics.get("f1") is not None:
        f1_diff = challenger_metrics["f1"] - champion_metrics["f1"]
        if f1_diff < PROMOTION_POLICY["min_f1_improvement"]:
            return (
                False,
                f"F1 Challenger ({challenger_metrics['f1']:.3f}) vs. "
                f"Champion ({champion_metrics['f1']:.3f}): "
                f"Δ={f1_diff:+.3f} unterschreitet Schwellenwert "
                f"({PROMOTION_POLICY['min_f1_improvement']:+.3f})",
            )
        return (
            True,
            f"Challenger übertrifft Champion: "
            f"F1 {champion_metrics['f1']:.3f} → {challenger_metrics['f1']:.3f} "
            f"(Δ={f1_diff:+.3f})",
        )

    return True, "Kein bestehendes Produktionsmodell – Challenger wird promoted."

def save_report(report: dict):
    """Speichert den Retraining-Report als JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"retraining_report_{ts}.json"
    report["timestamp"] = ts
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Retraining-Report gespeichert: {report_path}")
    return report_path

# ---------------------------------------------------------------------------
# Hauptpipeline
# ---------------------------------------------------------------------------
def run_retraining(
    waves: list[str],
    no_promote: bool,
):
    """Führt die gesamte Retraining-Pipeline aus."""

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    # ------------------------------------------------------------------
    # 1. Daten laden und zusammenführen
    # ------------------------------------------------------------------
    
    logger.info("SCHRITT 1: Daten laden")
    
    dfs = [pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])]
    for p in waves:
        df_wave = pd.read_csv(p, parse_dates=["timestamp"])
        dfs.append(df_wave[df_wave[TARGET].notna()])
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")

    report["data_sources"] = waves
    report["total_samples"] = len(df)

    # Zeitbasierter Split: letzte 15% als Validation
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET]

    # Capping auf 99. Perzentil — berechnet auf Trainingsdaten, auf beide angewendet
    caps = X_train[CAPPING_FEATURES].quantile(0.99)
    for col, cap in caps.items():
        X_train[col] = X_train[col].clip(upper=cap)
        X_val[col] = X_val[col].clip(upper=cap)

    logger.info(
        f"Train: {len(X_train)} Zeilen | Validation: {len(X_val)} Zeilen"
    )

    # ------------------------------------------------------------------
    # 2. Bestehendes Produktionsmodell abrufen
    # ------------------------------------------------------------------

    logger.info("SCHRITT 2: Champion-Metriken abrufen")

    champion_metrics = get_production_model_metrics(client)
    if champion_metrics:
        logger.info(
            f"Champion: Version {champion_metrics['version']} | "
            f"F1={champion_metrics['f1']:.3f} | "
            f"Precision={champion_metrics['precision']:.3f} | "
            f"Recall={champion_metrics['recall']:.3f}"
        )
    else:
        logger.info("Kein Produktionsmodell gefunden – erster Deployment.")

    report["champion"] = champion_metrics

    # ------------------------------------------------------------------
    # 3. Modell Training
    # ------------------------------------------------------------------

    logger.info("SCHRITT 3: Training")

    clf = LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        n_jobs=1, 
        verbose=-1,
        #Parameter, dei beim Hyperparameter-Tuning in train.py als beste identifiert wurden
        n_estimators=100,
        learning_rate=0.05,
        min_child_samples=10,
        num_leaves=15,
    )
    
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", clf),
    ])

    wave_label = "_".join([
        m.group(1) if (m := re.search(r"wave_(\d+)", p)) else Path(p).stem
        for p in waves
    ])
    
    with mlflow.start_run(run_name=f"LGBM_retrained_w{wave_label}") as run:

        mlflow.set_tags({
            "model_type": "lgbm",
            "waves_included": str(waves),
            "retrain_timestamp": datetime.now().isoformat(),
        })

        mlflow.log_param("model_type", "lgbm")
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("waves_included", str(waves))
        mlflow.log_params(clf.get_params())

        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        y_pred_val = pipeline.predict(X_val)
        y_prob_val = pipeline.predict_proba(X_val)[:, 1]

        train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train)
        val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)

        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        if champion_metrics and champion_metrics.get("f1") is not None:
            mlflow.log_metric(
                "f1_delta_vs_champion",
                round(val_metrics["f1"] - champion_metrics["f1"], 4),
            )

        logger.info(
            f"Challenger: F1={val_metrics['f1']:.3f} | "
            f"Precision={val_metrics['precision']:.3f} | "
            f"Recall={val_metrics['recall']:.3f} | "
            f"AUC={val_metrics['roc_auc']:.3f}"
        )

        # Classification Report
        report_str = classification_report(y_val, y_pred_val, target_names=["no risk", "risk"])
        logger.info(f"\n{report_str}")

        clf_path = REPORTS_DIR / f"classification_report_{wave_label}.txt"
        clf_path.write_text(report_str)
        mlflow.log_artifact(str(clf_path))
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred_val,
            display_labels=["no risk", "risk"],
            ax=ax,
            colorbar=False,
        )
        ax.set_title("Confusion Matrix – LGBM_retrained")
        plt.tight_layout()
        cm_path = REPORTS_DIR / f"confusion_matrix_{wave_label}.png"
        fig.savefig(cm_path)
        mlflow.log_artifact(str(cm_path))
        plt.close(fig)

        sample_input = X_val.iloc[:5]
        signature = infer_signature(sample_input, pipeline.predict(sample_input))

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id

    report["challenger"] = {
        "run_id": run_id,
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }

    # ------------------------------------------------------------------
    # 4. Promotion-Entscheidung
    # ------------------------------------------------------------------

    logger.info("SCHRITT 4: Promotion-Entscheidung")

    should_promote, promotion_reason = evaluate_promotion(
        val_metrics, champion_metrics
    )
    report["promotion_decision"] = should_promote
    report["promotion_reason"] = promotion_reason

    logger.info(f"Promotion: {'JA' if should_promote else 'NEIN'}")
    logger.info(f"Begründung: {promotion_reason}")

    if no_promote:
        logger.info("--no-promote gesetzt: Promotion wird nicht ausgeführt.")
        should_promote = False

    
    # ------------------------------------------------------------------
    # 5. Modell in Registry registrieren und promoten
    # ------------------------------------------------------------------

    if should_promote:
        logger.info("SCHRITT 5: Registrierung und Promotion")

        versions = client.search_model_versions(
                f"name='{MODEL_NAME}' and run_id='{run_id}'"
        )
        if not versions:
            logger.error(
                    "Modellversion nicht gefunden – bitte manuell über MLflow UI promoten."
            )
        else:
            new_version = versions[0].version

            # Altes Produktionsmodell archivieren
            try:
                old_prod = client.get_model_version_by_alias(MODEL_NAME, "production")
                client.delete_registered_model_alias(MODEL_NAME, "production")
                client.set_registered_model_alias(MODEL_NAME, "archived", old_prod.version)
                
                logger.info(f"Alte Produktionsversion {old_prod.version} → Archived")
            except Exception:
                pass # Kein bestehendes Produktionsmodell

            # Staging → Production
            #client.set_registered_model_alias(MODEL_NAME, "staging", new_version)
            #logger.info(f"Version {new_version} → Staging")

            client.set_registered_model_alias(MODEL_NAME, "production", new_version)
            logger.info(f"Version {new_version} → Production ✓")

            client.update_model_version(
                name=MODEL_NAME,
                version=new_version,
                description=(
                    f"Retrained: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                    f"Waves: {waves} | "
                    f"F1: {val_metrics['f1']:.3f} | "
                    f"{promotion_reason}"
                ),
            )

            report["promoted_version"] = new_version
    else:
        logger.info(
            "Challenger wird NICHT promoted. Aktuelles Produktionsmodell bleibt aktiv."
        )
        report["promoted_version"] = None


    # ------------------------------------------------------------------
    # 6. Report speichern
    # ------------------------------------------------------------------
    logger.info("SCHRITT 6: Report speichern")

    report_path = save_report(report)

    logger.info("Retraining abgeschlossen.")
    logger.info(f"Report: {report_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Retraining – LGBM-RiskClassification"
    )
    parser.add_argument(
        "--waves",
        nargs="+",
        type=str,
        required=True,
        metavar="PATH",
        help="Pfade zu den Production Waves, z.B. data/raw/production_wave_1.csv",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Nur trainieren und vergleichen, aber nicht in Production promoten.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_retraining(
        waves=args.waves,
        no_promote=args.no_promote,
    )
