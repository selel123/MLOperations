"""
Monitoring Pipeline – Drift-Erkennung & Retraining-Trigger

PSI < 0.10    → stabil
PSI 0.10–0.25 → Warnung
PSI > 0.25    → kritischer Drift → retrain.py wird aufgerufen
F1-Abfall >= 3% → retrain.py wird aufgerufen

F1_REF wird dynamisch aus dem aktuellen Production-Modell in MLflow geladen.
Die Plots bauen sich kumulativ auf: wave1 zeigt nur wave1, wave2 zeigt wave1+2, usw.

Usage:
    python monitor.py --waves data/raw/production_wave_2.csv data/raw/production_wave_3.csv
    python monitor.py --waves data/raw/production_wave_2.csv --no-retrain
"""

import argparse
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
TRACKING_URI  = "http://46.225.163.17:5000"
MODEL_NAME    = "LGBM-RiskClassification"
TRAIN_PATH    = Path("data/raw/train_full.csv")
REPORTS_DIR   = Path("reports")
INFERENCE_LOG = Path("logs/inference_log.csv")

PSI_WARN      = 0.10
PSI_CRITICAL  = 0.25
F1_DROP       = 0.03 # Trigger bei F1-Abfall von 3 Prozentpunkten, sonst zu viele verlorerene Kunden


NUMERIC_FEATURES = [
    "transaction_volume", "processing_time_hours",
    "historical_incidents_90d", "open_cases_count",
    "customer_tenure_months", "change_requests_30d",
]
ALL_FEATURES = NUMERIC_FEATURES + [
    "missing_docs_flag", "high_priority_source_flag",
    "region", "channel", "customer_segment", "product_line",
]


# ---------------------------------------------------------------------------
# PSI – misst wie stark sich eine Feature-Verteilung verändert hat
# Vorgehen:
#   1. Referenz (Training) in 10 Bins aufteilen
#   2. Anteil der Werte pro Bin berechnen
# ---------------------------------------------------------------------------
def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    breakpoints = np.unique(np.percentile(reference.dropna(), np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0
    ref_pct = np.histogram(reference.dropna(), bins=breakpoints)[0] / len(reference.dropna())
    cur_pct = np.histogram(current.dropna(),   bins=breakpoints)[0] / len(current.dropna())
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ---------------------------------------------------------------------------
# Performance – Modell macht Predictions auf Wave, vergleicht mit echten Labels
# ---------------------------------------------------------------------------
def evaluate_performance(model, df: pd.DataFrame, wave_name: str) -> dict:
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
    X, y   = df[ALL_FEATURES], df["risk_flag"]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "wave":      wave_name,
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Inference Logging – speichert Predictions mit Zeitstempel & Modellversion
# ---------------------------------------------------------------------------
def log_inference(model_version: str, wave_name: str, df: pd.DataFrame, model):
    X      = df[ALL_FEATURES]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    log_df = df[ALL_FEATURES].copy()
    log_df["timestamp"]     = datetime.now().isoformat()
    log_df["model_version"] = model_version
    log_df["wave"]          = wave_name
    log_df["prediction"]    = y_pred
    log_df["score"]         = y_prob.round(4)

    INFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    header = not INFERENCE_LOG.exists()
    log_df.to_csv(INFERENCE_LOG, mode="a", header=header, index=False)
    print(f"Inference Log aktualisiert: {INFERENCE_LOG} ({len(log_df)} Zeilen)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_psi(psi_df: pd.DataFrame, suffix: str = ""):
    """Balkendiagramm: PSI pro Feature, eine Spalte pro Wave."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    waves    = psi_df["wave"].unique()
    n        = len(waves)
    features = psi_df.groupby("feature")["psi"].max().sort_values().index

    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(3, len(features) * 0.45)), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, wave in zip(axes, waves):
        df     = psi_df[psi_df["wave"] == wave].set_index("feature")["psi"].reindex(features)
        colors = ["#E53935" if v >= PSI_CRITICAL else "#FB8C00" if v >= PSI_WARN else "#43A047"
                  for v in df.values]
        ax.barh(df.index, df.values, color=colors, height=0.6)
        ax.axvline(PSI_WARN,     linestyle="--", color="#FB8C00", linewidth=1.2, label=f"Warn ({PSI_WARN})")
        ax.axvline(PSI_CRITICAL, linestyle="--", color="#E53935", linewidth=1.2, label=f"Kritisch ({PSI_CRITICAL})")
        ax.set_xlabel("PSI")
        ax.set_title(wave, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("PSI pro Feature", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = REPORTS_DIR / f"psi{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot gespeichert: {path}")


def plot_performance(perf_rows: list, f1_ref: float, suffix: str = ""):
    """Liniendiagramm: F1, Recall, Precision, AUC über alle Waves."""
    if not perf_rows:
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(perf_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, color in [("f1", "#2196F3"), ("recall", "#4CAF50"),
                          ("precision", "#FF9800"), ("roc_auc", "#9C27B0")]:
        ax.plot(df["wave"], df[metric], "-o", color=color, linewidth=2, label=metric.upper())
    ax.axhline(f1_ref,           linestyle="--", color="#E53935", linewidth=1.2, label=f"F1-Ref ({f1_ref:.4f})")
    ax.axhline(f1_ref - F1_DROP, linestyle=":",  color="#E53935", linewidth=1.0, label=f"Trigger ({f1_ref - F1_DROP:.4f})")
    ax.set_ylim(0, 1)
    ax.set_title("Modell-Performance im Zeitverlauf", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = REPORTS_DIR / f"performance{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot gespeichert: {path}")


# ---------------------------------------------------------------------------
# Hauptpipeline
# ---------------------------------------------------------------------------
def run_monitoring(wave_paths: list[Path], auto_retrain: bool = True):
    print("=" * 60)
    print("MLOps Monitoring Pipeline")
    print("=" * 60)

    # 1. Trainingsdaten als Referenz laden
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])
    train_df = train_df[train_df["timestamp"] < "2025-04-01"]
    print(f"\nReferenz (Training): {len(train_df):,} Zeilen")

    # 2. Production Waves laden
    waves = [{"name": p.stem.replace("_", " ").title(),
              "stem": p.stem,
              "df":   pd.read_csv(p, parse_dates=["timestamp"])}
             for p in wave_paths]
    for w in waves:
        print(f"{w['name']}: {len(w['df']):,} Zeilen")

    # 3. Modell aus MLflow laden + F1_REF aus Trainings-Run-Metriken
    model         = None
    model_version = "unbekannt"
    f1_ref        = None
    try:
        import mlflow.sklearn
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        client        = mlflow.MlflowClient()
        version       = client.get_model_version_by_alias(MODEL_NAME, "production")
        model_version = f"v{version.version}"
        model         = mlflow.sklearn.load_model(f"runs:/{version.run_id}/model")
        run           = client.get_run(version.run_id)
        f1_ref        = run.data.metrics.get("test_f1")
        if f1_ref is None:
            print("WARNUNG: F1-Metrik nicht im MLflow-Run gefunden, verwende Fallback 0.564")
            f1_ref = 0.564
        print(f"\nModell geladen: {MODEL_NAME}@production ({model_version})")
        print(f"F1-Referenz (aus Production-Run): {f1_ref:.4f}")
    except Exception as e:
        print(f"\nWARNUNG: Modell nicht geladen ({e})")
        f1_ref = 0.564

    # 4. PSI pro Feature & Wave berechnen
    psi_rows = []
    for wave in waves:
        for feat in NUMERIC_FEATURES:
            psi_rows.append({
                "feature": feat,
                "wave":    wave["name"],
                "psi":     round(compute_psi(train_df[feat], wave["df"][feat]), 4),
            })
    psi_df = pd.DataFrame(psi_rows)

    # 5. Performance & Inference Logging pro Wave (nur wenn Modell geladen)
    perf_rows = []
    if model:
        for wave in waves:
            if "risk_flag" in wave["df"].columns:
                perf_rows.append(evaluate_performance(model, wave["df"], wave["name"]))
            log_inference(model_version, wave["name"], wave["df"], model)

    # 6. Trigger-Logik -- Schwellwerte Definition
    # 
    reasons = []
    max_psi = psi_df["psi"].max()
    if max_psi >= PSI_CRITICAL:
        worst = psi_df.loc[psi_df["psi"].idxmax()]
        reasons.append(f"PSI={max_psi:.4f} für '{worst['feature']}' in '{worst['wave']}' > {PSI_CRITICAL}")
    if perf_rows:
        last_f1     = perf_rows[-1]["f1"]
        f1_drop_val = f1_ref - last_f1
        if f1_drop_val >= F1_DROP:
            reasons.append(f"F1={last_f1:.4f} (Abfall {f1_drop_val:.4f} >= Schwelle {F1_DROP})")
    triggered = bool(reasons)

    # 7. Ergebnisse ausgeben
    print(f"\n{'─'*50}")
    print("PSI-Ergebnisse (max über alle Waves):")
    for feat, val in psi_df.groupby("feature")["psi"].max().sort_values(ascending=False).items():
        status = "KRITISCH" if val >= PSI_CRITICAL else "WARNUNG" if val >= PSI_WARN else "OK"
        print(f"  {feat:<35} {val:.4f}  [{status}]")

    if perf_rows:
        print("\nModell-Performance:")
        for r in perf_rows:
            print(f"  {r['wave']:<30} F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}  "
                  f"Recall={r['recall']:.4f}  Precision={r['precision']:.4f}")

    print(f"\nRetraining-Trigger: {'JA' if triggered else 'NEIN'}")
    for r in reasons:
        print(f"  → {r}")
    print(f"{'─'*50}")

    # 8. Kumulative Plots – wave1 zeigt nur wave1, wave2 zeigt wave1+2, usw.
    wave_names = [w["name"] for w in waves]
    for i, wave in enumerate(waves):
        suffix       = f"_{wave['stem']}"
        waves_so_far = wave_names[: i + 1]
        plot_psi(psi_df[psi_df["wave"].isin(waves_so_far)], suffix=suffix)
        plot_performance([r for r in perf_rows if r["wave"] in waves_so_far], f1_ref, suffix=suffix)

    # 9. PSI-Ergebnisse als CSV speichern
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    psi_df.to_csv(REPORTS_DIR / "psi_summary.csv", index=False)

    # 10. Retraining starten
    if triggered and auto_retrain:
        print("\nStarte automatisches Retraining ...")
        result = subprocess.run(
            [sys.executable, "src/training/retrain.py", "--waves"] + [str(p) for p in wave_paths]
        )
        print("Retraining abgeschlossen." if result.returncode == 0
              else f"Retraining fehlgeschlagen (Exit {result.returncode}).")
    elif triggered:
        print("\nTrigger aktiv, aber --no-retrain gesetzt.")
        print("Manuell: python src/training/retrain.py --waves <wave_paths>")

    return {"triggered": triggered, "reasons": reasons, "max_psi": max_psi}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Monitoring – Drift-Erkennung")
    parser.add_argument("--waves", nargs="+", type=Path, required=True,
                        help="Production-Wave CSV-Dateien")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Trigger melden, retrain.py aber NICHT aufrufen")
    args = parser.parse_args()
    run_monitoring(args.waves, auto_retrain=not args.no_retrain)
