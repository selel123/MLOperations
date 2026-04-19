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
    python src/training/train.py --all          → alle Modelle mit GridSearchCV tunen
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
from sklearn.svm import SVC
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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://46.225.163.17:5000"
EXPERIMENT_NAME = "classification-ml"
DATA_PATH = Path("data/raw/train_full.csv")
TARGET = "risk_flag"
SPLIT_DATE = "2025-04-01"

# log1p nur für stark skewed Features (Skewness > 0.75, EDA-basiert)
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

# Capping auf 99. Perzentil (echte Ausreißer in diesen Features, EDA-basiert)
CAPPING_FEATURES = ["historical_incidents_90d", "processing_time_hours", "transaction_volume"]

# ---------------------------------------------------------------------------
# Modell-Registry — neue Modelle hier ergänzen!!
# ---------------------------------------------------------------------------
MODELS = {
    "lr": LogisticRegression,
    "rf": RandomForestClassifier,
    "gb": GradientBoostingClassifier,
    "lgbm": LGBMClassifier,
    "xgb": XGBClassifier,
    "svm": SVC,
}

# Feste Modell Argumente 
MODEL_FIXED_KWARGS = {
    "lr": {"class_weight": "balanced", "random_state": 42, "solver": "lbfgs", "max_iter": 1000},
    "rf": {"class_weight": "balanced", "random_state": 42, "n_jobs": -1},
    "gb": {"random_state": 42},
    "lgbm": {"is_unbalance": True, "random_state": 42, "n_jobs": 1, "verbose": -1},
    "xgb": {"random_state": 42, "n_jobs": 1, "eval_metric": "logloss"},
    "svm":  {"class_weight": "balanced", "probability": True, "kernel": "rbf"},
}


# ---------------------------------------------------------------------------
# 1. load_data
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH):
    """Daten laden, zeitbasiert splitten, Capping auf Trainingsdaten berechnen."""
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
# 2. build_preprocessor
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Preprocessing-Pipeline getrennt vom Modellcode.

    log1p: nur stark skewed Features (EDA-basiert, Skewness > 0.75)
    StandardScaler: alle numerischen Features
    OneHotEncoder: kategoriale Features
    """
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
    ])


# ---------------------------------------------------------------------------
# 3. get_model
# ---------------------------------------------------------------------------
def get_model(model_name: str) -> Pipeline:
    """Gibt eine vollständige sklearn Pipeline (Preprocessor + Classifier) zurück."""
    if model_name not in MODELS:
        raise ValueError(f"Unbekanntes Modell: '{model_name}'. Verfügbar: {list(MODELS)}")

    classifier = MODELS[model_name](**MODEL_FIXED_KWARGS[model_name])
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   classifier),
    ])


# ---------------------------------------------------------------------------
# 4. get_param_grid
# ---------------------------------------------------------------------------
def get_param_grid(model_name: str) -> dict:
    """Hyperparameter-Suchraum pro Modell für GridSearchCV."""
    param_grids = {
        "lr": {
            "classifier__C": [0.01, 0.1, 1, 10],
        },
        "rf": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth":    [5, 10, 15],
        },
        "gb": {
            "classifier__n_estimators":  [100, 200],
            "classifier__max_depth":     [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        },
        "lgbm": {
            "classifier__num_leaves":        [15, 31, 63],
            "classifier__n_estimators":      [100, 200],
            "classifier__learning_rate":     [0.05, 0.1],
            "classifier__min_child_samples": [10, 20, 50],
        },
        "xgb": {
            "classifier__n_estimators":  [100, 200],
            "classifier__max_depth":     [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        },
        "svm": {
            "classifier__C":     [0.1, 1, 10],
            "classifier__gamma": ["scale", "auto"],
        },
    }
    return param_grids[model_name]


# ---------------------------------------------------------------------------
# 5. tune_model
# ---------------------------------------------------------------------------
def tune_model(model_name: str, X_train, y_train) -> GridSearchCV:
    """GridSearchCV auf Trainingsdaten — gibt das fertig gefittete Objekt zurück."""
    pipeline   = get_model(model_name)
    param_grid = get_param_grid(model_name)

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=TimeSeriesSplit(3),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"  Beste Params: {search.best_params_}")
    print(f"  Bestes CV-F1: {search.best_score_:.4f}")

    return search


# ---------------------------------------------------------------------------
# 6. evaluate_model
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name: str):
    """Testset-Evaluation — gibt Metriken-Dict und Confusion-Matrix-Figure zurück."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_f1":        f1_score(y_test, y_pred),
        "test_recall":    recall_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_roc_auc":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'─'*30} Test  {'─'*30}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['no risk', 'risk'])}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["no risk", "risk"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()

    return metrics, fig


# ---------------------------------------------------------------------------
# 7. compute_feature_importance
# ---------------------------------------------------------------------------
def compute_feature_importance(model, X_test, y_test, model_name: str):
    """Permutation Importance auf dem Testset — nach GridSearch.

    Feature-Namen werden direkt aus der Pipeline (ColumnTransformer) extrahiert.
    Gibt einen sortierten DataFrame zurück und speichert ihn als CSV.
    """
    feature_names = X_test.columns
    # Präfixe (log__, plain__, cat__) entfernen für bessere Lesbarkeit
    feature_names = [n.split("__", 1)[-1] for n in feature_names]

    result = permutation_importance(
        model, X_test, y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature":          feature_names,
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    csv_path = Path(f"feature_importance_{model_name}.csv")
    importance_df.to_csv(csv_path, index=False)

    print(f"\n{'─'*30} Feature Importance ({model_name}) {'─'*10}")
    print(importance_df.head(10).to_string(index=False))

    return importance_df, csv_path


# ---------------------------------------------------------------------------
# 8. run_exists
# ---------------------------------------------------------------------------
def run_exists(model_name: str, best_params: dict) -> bool:
    """Prüft ob ein identischer Run (Modell + Parameter) bereits in MLflow existiert."""
    try:
        runs = mlflow.search_runs(search_all_experiments=False)
    except Exception:
        return False

    if runs.empty:
        return False

    for _, run in runs.iterrows():
        if run.get("params.model") != model_name:
            continue
        if all(
            str(run.get(f"params.{k}")) == str(v)
            for k, v in best_params.items()
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# 9. log_to_mlflow
# ---------------------------------------------------------------------------
def log_to_mlflow(model_name: str, best_params: dict, best_model,
                  metrics: dict, confusion_fig, importance_csv: Path, X_train):
    """Loggt einen einzigen Run pro Modell — keine CV-Folds, keine Einzelkombinationen."""
    run_name = f"{model_name.upper()}_tuned"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("split_date", SPLIT_DATE)
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        mlflow.log_figure(confusion_fig, "confusion_matrix.png")
        mlflow.log_artifact(str(importance_csv), artifact_path="feature_importance")

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:3],
        )

        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    plt.close(confusion_fig)


# ---------------------------------------------------------------------------
# 10. main
# ---------------------------------------------------------------------------
def main(model_names: list):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Modell: {model_name.upper()}  |  GridSearchCV startet ...")
        print("=" * 60)

        search      = tune_model(model_name, X_train, y_train)
        best_model  = search.best_estimator_
        best_params = {
            k.replace("classifier__", ""): v
            for k, v in search.best_params_.items()
        }

        if run_exists(model_name, best_params):
            print(f"  Run existiert bereits (Modell={model_name}, Params={best_params}) → skip")
            continue

        metrics, confusion_fig           = evaluate_model(best_model, X_test, y_test, model_name)
        _, importance_csv                 = compute_feature_importance(best_model, X_test, y_test, model_name)
        log_to_mlflow(model_name, best_params, best_model, metrics, confusion_fig, importance_csv, X_train)

    print(f"\n{'='*60}")
    print(f"Fertig. Experiment: {EXPERIMENT_NAME}  |  Tracking: {TRACKING_URI}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Risk Flag Classification Training mit MLflow")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", type=str, choices=list(MODELS),
        help="Einzelnes Modell tunen und loggen.",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Alle Modelle tunen und loggen.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models_to_run = list(MODELS) if args.all else [args.model]
    main(models_to_run)
