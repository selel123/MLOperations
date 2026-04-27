"""
Batch-Scoring – LGBM-RiskClassification

Verwendung:
    python predict_batch.py --input data/raw/new_customers.csv
    python predict_batch.py --input data/raw/new_customers.csv --output results/scored.csv
"""

import argparse
from datetime import datetime
from pathlib import Path

import mlflow.sklearn
import pandas as pd

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
TRACKING_URI = "http://46.225.163.17:5000"
MODEL_NAME   = "LGBM-RiskClassification"

EXPECTED_FEATURES = [
    "transaction_volume", "processing_time_hours",
    "historical_incidents_90d", "change_requests_30d",
    "open_cases_count", "customer_tenure_months",
    "missing_docs_flag", "high_priority_source_flag",
    "region", "channel", "customer_segment", "product_line",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(input_path: Path, output_path: Path):
    # Modell laden
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    print(f"Modell geladen: {MODEL_NAME}@production")

    # Input einlesen
    df = pd.read_csv(input_path)
    print(f"{len(df):,} Zeilen eingelesen")

    # Fehlende Spalten prüfen
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")

    # Scoring
    X = df[EXPECTED_FEATURES]
    df["risk_flag"]        = model.predict(X).astype(int)
    df["risk_probability"] = model.predict_proba(X)[:, 1].round(4)

    # Ergebnis speichern
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Ergebnis gespeichert: {output_path}")
    print(f"Churn-Risiko erkannt: {df['risk_flag'].sum():,} von {len(df):,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-Scoring fuer LGBM-RiskClassification")
    parser.add_argument("--input",  type=Path, required=True, help="Input-CSV mit den 12 Feature-Spalten")
    parser.add_argument("--output", type=Path, default=None,  help="Output-CSV (Standard: results/scored_<timestamp>.csv)")
    args = parser.parse_args()

    output = args.output or Path(f"results/scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    main(args.input, output)