"""Training pipeline – Logistische Regression, Random Forest, Gradient Boosting mit MLflow Tracking.

Datensplit:
    Training:  Jan–Mrz 2025 (timestamp < 2025-04-01)
    Test:      Apr 2025     (timestamp >= 2025-04-01)

    Zeitbasierter Split (kein zufälliger Split), da der Datensatz eine
    zeitliche Struktur aufweist. Ein zufälliger Split würde Data Leakage
    erzeugen, weil zukünftige Muster ins Training einfließen würden.

Metriken:
    Primär:    F1 (Klasse 1) – Hauptmetrik bei moderater Klassenimbalance (65/35)
    Sekundär:  Recall (Klasse 1) – Risikofälle dürfen nicht übersehen werden
               Precision (Klasse 1) – Kontrolle gegen zu viele False Positives
               ROC-AUC – schwellenwertunabhängige Gesamtperformance

Run:
    python train.py                   → alle Runs (Baseline + optimierte Varianten)
    python train.py --model lr        → nur LR mit Standardparametern
    python train.py --model rf --n_estimators 200 --max_depth 10
    python train.py --model gb --n_estimators 200 --max_depth 5 --learning_rate 0.05
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "classification-ml"
DATA_PATH = Path("data/raw/train_full.csv")
TARGET = "risk_flag"
SPLIT_DATE = "2025-04-01"

# log1p-Transformation bei Skewness > 0.75 (außer binäre Flags)
LOG_FEATURES = [
    "transaction_volume",
    "processing_time_hours",
    "historical_incidents_90d",
    "change_requests_30d",
]
# Kein Handlungsbedarf oder binäre Features
PLAIN_FEATURES = [
    "open_cases_count",
    "customer_tenure_months",
    "missing_docs_flag",
    "high_priority_source_flag",
]
NUMERIC_FEATURES = LOG_FEATURES + PLAIN_FEATURES
CATEGORICAL_FEATURES = ["region", "channel", "customer_segment", "product_line"]

# Capping auf 99. Perzentil (echte Ausreißer in diesen Features)
CAPPING_FEATURES = ["historical_incidents_90d", "processing_time_hours", "transaction_volume"]

# ---------------------------------------------------------------------------
# Alle Runs: Baseline + optimierte Varianten
# ---------------------------------------------------------------------------
ALL_RUNS = [
    # Baseline: LR mit Standardparametern, kein Tuning
    {
        "run_name": "LR_baseline",
        "model":    "lr",
        "params":   {"C": 1.0, "max_iter": 1000},
    },
    # LR – verschiedene Regularisierungsstärken
    {
        "run_name": "LR_C0.01",
        "model":    "lr",
        "params":   {"C": 0.01, "max_iter": 1000},
    },
    {
        "run_name": "LR_C0.1",
        "model":    "lr",
        "params":   {"C": 0.1, "max_iter": 1000},
    },
    {
        "run_name": "LR_C10",
        "model":    "lr",
        "params":   {"C": 10.0, "max_iter": 1000},
    },
    # RF – Tiefe und Baumanzahl variiert
    {
        "run_name": "RF_d5_n100",
        "model":    "rf",
        "params":   {"n_estimators": 100, "max_depth": 5},
    },
    {
        "run_name": "RF_d10_n100",
        "model":    "rf",
        "params":   {"n_estimators": 100, "max_depth": 10},
    },
    {
        "run_name": "RF_d10_n200",
        "model":    "rf",
        "params":   {"n_estimators": 200, "max_depth": 10},
    },
    {
        "run_name": "RF_d15_n200",
        "model":    "rf",
        "params":   {"n_estimators": 200, "max_depth": 15},
    },
    # GB – Lernrate und Tiefe variiert
    {
        "run_name": "GB_lr0.1_d3_n100",
        "model":    "gb",
        "params":   {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    },
    {
        "run_name": "GB_lr0.05_d3_n200",
        "model":    "gb",
        "params":   {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    },
    {
        "run_name": "GB_lr0.1_d5_n200",
        "model":    "gb",
        "params":   {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    },
    {
        "run_name": "GB_lr0.05_d5_n300",
        "model":    "gb",
        "params":   {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    },
]


# ---------------------------------------------------------------------------
# Daten laden & zeitbasierter Split
# ---------------------------------------------------------------------------
def load_and_split(path: Path):
    df = pd.read_csv(path, parse_dates=["timestamp"])

    train_df = df[df["timestamp"] < SPLIT_DATE].copy()
    test_df  = df[df["timestamp"] >= SPLIT_DATE].copy()

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train_df[features], train_df[TARGET]
    X_test,  y_test  = test_df[features],  test_df[TARGET]

    # Capping auf 99. Perzentil — berechnet auf Trainingsdaten, auf beide angewendet
    caps = X_train[CAPPING_FEATURES].quantile(0.99)
    for col, cap in caps.items():
        X_train[col] = X_train[col].clip(upper=cap)
        X_test[col]  = X_test[col].clip(upper=cap)

    print(f"Trainingszeitraum:  {train_df['timestamp'].min().date()} – {train_df['timestamp'].max().date()}")
    print(f"Testzeitraum:       {test_df['timestamp'].min().date()}  – {test_df['timestamp'].max().date()}")
    print(f"Train: {len(X_train):,} Zeilen | Test: {len(X_test):,} Zeilen")
    print(f"Klassenverteilung Train: {dict(y_train.value_counts().sort_index())}")
    print(f"Klassenverteilung Test:  {dict(y_test.value_counts().sort_index())}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(model: str, params: dict) -> Pipeline:
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
    preprocessor = ColumnTransformer([
        ("log",   log_transformer,         LOG_FEATURES),
        ("plain", plain_transformer,       PLAIN_FEATURES),
        ("cat",   categorical_transformer, CATEGORICAL_FEATURES),
    ])

    if model == "lr":
        classifier = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
    elif model == "rf":
        classifier = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model == "gb":
        classifier = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
        )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Metriken & Visualisierung
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob, prefix: str) -> dict:
    return {
        f"{prefix}_f1":        f1_score(y_true, y_pred),
        f"{prefix}_recall":    recall_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc":   roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, run_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["no risk", "risk"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix – {run_name}")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Einzelner MLflow Run
# ---------------------------------------------------------------------------
def run_single(run_name: str, model: str, params: dict, X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}")
    print(f"Starte: {run_name}  |  params: {params}")
    print("=" * 60)

    pipeline = build_pipeline(model, params)

    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test  = pipeline.predict(X_test)
        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        y_prob_test  = pipeline.predict_proba(X_test)[:, 1]

        train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train, prefix="train")
        test_metrics  = compute_metrics(y_test,  y_pred_test,  y_prob_test,  prefix="test")

        mlflow.log_params({"model": model, "split_date": SPLIT_DATE,
                           "train_size": len(X_train), "test_size": len(X_test), **params})
        mlflow.log_metrics({**train_metrics, **test_metrics})

        fig = plot_confusion_matrix(y_test, y_pred_test, run_name)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, name="model", signature=signature,
                                 input_example=X_train.iloc[:3])

        print(f"\n{'─'*30} Train {'─'*30}")
        for k, v in train_metrics.items():
            print(f"  {k:<25} {v:.4f}")
        print(f"\n{'─'*30} Test  {'─'*30}")
        for k, v in test_metrics.items():
            print(f"  {k:<25} {v:.4f}")
        print(f"\n{classification_report(y_test, y_pred_test, target_names=['no risk', 'risk'])}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------
def train_all():
    """Startet alle Runs sequentiell: Baseline LR + optimierte Varianten."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)

    print(f"\n{len(ALL_RUNS)} Runs werden gestartet ...")
    for run_cfg in ALL_RUNS:
        run_single(run_cfg["run_name"], run_cfg["model"], run_cfg["params"],
                   X_train, X_test, y_train, y_test)

    print(f"\n{'='*60}")
    print(f"Alle {len(ALL_RUNS)} Runs abgeschlossen.")
    print(f"Experiment: {EXPERIMENT_NAME}  |  Tracking: {TRACKING_URI}")


def train_single(args):
    """Startet einen einzelnen Run basierend auf CLI-Argumenten."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)

    model_labels = {"lr": "LogisticRegression", "rf": "RandomForest", "gb": "GradientBoosting"}
    if args.model == "lr":
        params = {"C": args.C, "max_iter": args.max_iter}
    elif args.model == "rf":
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    elif args.model == "gb":
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                  "learning_rate": args.learning_rate}

    run_single(model_labels[args.model], args.model, params, X_train, X_test, y_train, y_test)
    print(f"\nExperiment: {EXPERIMENT_NAME}  |  Tracking: {TRACKING_URI}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Risk Flag Classification Training mit MLflow")
    parser.add_argument("--model", type=str, default=None, choices=["lr", "rf", "gb"],
                        help="Einzelnes Modell trainieren. Ohne Angabe: alle Runs.")
    # LR
    parser.add_argument("--C",             type=float, default=1.0,  help="LR: Regularisierungsstärke")
    parser.add_argument("--max_iter",      type=int,   default=1000, help="LR: Max. Iterationen")
    # RF + GB
    parser.add_argument("--n_estimators",  type=int,   default=100,  help="RF/GB: Anzahl Bäume")
    parser.add_argument("--max_depth",     type=int,   default=None, help="RF/GB: Max. Tiefe")
    # GB
    parser.add_argument("--learning_rate", type=float, default=0.1,  help="GB: Lernrate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model is None:
        train_all()
    else:
        train_single(args)


"""
Analyse Baseline-Modell – Logistische Regression (C=1.0)
                                                                                                                                     
  Kein Overfitting: Train- und Test-Metriken liegen sehr nah beieinander (train_f1=0.577 vs. test_f1=0.560, train_roc_auc=0.730 vs.
  test_roc_auc=0.719). Das Modell generalisiert stabil auf den April-Testdaten.                                                      
                                          
  Recall vor Precision: Der Recall von 0.636 bedeutet, dass das Modell ~64% aller echten Risikofälle korrekt erkennt. Gemäß der      
  Problemstellung (False Negatives sind kritischer als False Positives) ist das die relevante Größe. Die Precision von 0.50 zeigt,
  dass jeder zweite Alarm ein False Positive ist — bei einem Screening-System vor manueller Prüfung akzeptabel.                      
                                          
  ROC-AUC 0.72: Das Modell ist deutlich besser als Random (0.5), aber weit von perfekter Trennung (1.0) entfernt. Das entspricht dem 
  EDA-Befund: kein Feature hat eine starke Einzelkorrelation mit dem Target (Maximum r=0.355 bei transaction_volume). Die schwache
  Trennbarkeit ist datensatzbedingt, nicht modellbedingt.                                                                            
                                          
  Klassenimbalance kompensiert: Durch class_weight="balanced" werden beide Klassen gleich gewichtet. Ohne diesen Parameter würde das 
  Modell die Mehrheitsklasse (65% kein Risiko) bevorzugen und Recall deutlich schlechter.
                                                                                                                                     
  Fazit: Das Baseline-Modell ist solide und gut kalibriert — die Performance ist durch die Datenlage begrenzt, nicht durch den       
  Algorithmus. Kein anderes getestetes Modell konnte es übertreffen.
                                                                          
"""