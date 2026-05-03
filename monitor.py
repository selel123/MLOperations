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
INFERENCE_LOG    = Path("logs/inference_log.csv")
PERFORMANCE_LOG  = Path("logs/performance_log.csv")
WAVE_HISTORY_LOG = Path("logs/processed_waves.txt")

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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    waves    = psi_df["wave"].unique()
    features = psi_df.groupby("feature")["psi"].max().sort_values().index
    fig, axes = plt.subplots(1, len(waves), figsize=(5 * len(waves), max(3, len(features) * 0.45)), sharey=True)
    if len(waves) == 1:
        axes = [axes]
    for ax, wave in zip(axes, waves):
        vals = psi_df[psi_df["wave"] == wave].set_index("feature")["psi"].reindex(features)
        ax.barh(vals.index, vals.values, color="steelblue")
        ax.axvline(PSI_WARN,     linestyle="--", color="orange", label=f"Warn ({PSI_WARN})")
        ax.axvline(PSI_CRITICAL, linestyle="--", color="red",    label=f"Kritisch ({PSI_CRITICAL})")
        ax.set_xlabel("PSI")
        ax.set_title(wave)
        ax.legend(fontsize=8)
    fig.suptitle("PSI pro Feature")
    fig.tight_layout()
    path = REPORTS_DIR / f"psi{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot gespeichert: {path}")


def plot_performance(perf_rows: list, f1_ref: float, suffix: str = ""):
    if not perf_rows:
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(perf_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ["f1", "recall", "precision", "roc_auc"]:
        ax.plot(df["wave"], df[metric], "-o", label=metric.upper())
    ax.axhline(f1_ref,           linestyle="--", color="red", label=f"F1-Ref ({f1_ref:.4f})")
    ax.axhline(f1_ref - F1_DROP, linestyle=":",  color="red", label=f"Trigger ({f1_ref - F1_DROP:.4f})")
    ax.set_ylim(0, 1)
    ax.set_title("Modell-Performance im Zeitverlauf")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = REPORTS_DIR / f"performance{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot gespeichert: {path}")


# ---------------------------------------------------------------------------
# Einzelne Wave überwachen (intern genutzt)
# ---------------------------------------------------------------------------
def _monitor_single_wave(wave: dict, train_df: pd.DataFrame, model, model_version: str,
                          f1_ref: float, all_perf_rows: list) -> dict:
    """Prüft eine einzelne Wave auf PSI-Drift und F1-Abfall. Gibt Trigger-Ergebnis zurück."""
    wave_name = wave["name"]
    wave_stem = wave["stem"]

    # PSI berechnen
    psi_rows = []
    for feat in NUMERIC_FEATURES:
        psi_rows.append({
            "feature": feat,
            "wave":    wave_name,
            "psi":     round(compute_psi(train_df[feat], wave["df"][feat]), 4),
        })
    psi_df = pd.DataFrame(psi_rows)

    # Performance & Inference Logging
    perf_row = None
    if model:
        if "risk_flag" in wave["df"].columns:
            perf_row = evaluate_performance(model, wave["df"], wave_name)
            PERFORMANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([perf_row]).to_csv(
                PERFORMANCE_LOG, mode="a",
                header=not PERFORMANCE_LOG.exists(), index=False
            )
            all_perf_rows.append(perf_row)
        log_inference(model_version, wave_name, wave["df"], model)

    # Trigger-Logik
    reasons = []
    max_psi = psi_df["psi"].max()
    if max_psi >= PSI_CRITICAL:
        worst = psi_df.loc[psi_df["psi"].idxmax()]
        reasons.append(f"PSI={max_psi:.4f} für '{worst['feature']}' in '{worst['wave']}' > {PSI_CRITICAL}")
    if perf_row:
        f1_drop_val = f1_ref - perf_row["f1"]
        if f1_drop_val >= F1_DROP:
            reasons.append(f"F1={perf_row['f1']:.4f} (Abfall {f1_drop_val:.4f} >= Schwelle {F1_DROP})")
    triggered = bool(reasons)

    # Ausgabe
    print(f"\n{'─'*50}")
    print(f"Wave: {wave_name}")
    print("PSI-Ergebnisse:")
    for feat, val in psi_df.set_index("feature")["psi"].sort_values(ascending=False).items():
        status = "KRITISCH" if val >= PSI_CRITICAL else "WARNUNG" if val >= PSI_WARN else "OK"
        print(f"  {feat:<35} {val:.4f}  [{status}]")
    if perf_row:
        print(f"  Performance: F1={perf_row['f1']:.4f}  AUC={perf_row['roc_auc']:.4f}  "
              f"Recall={perf_row['recall']:.4f}  Precision={perf_row['precision']:.4f}")
    print(f"Retraining-Trigger: {'JA' if triggered else 'NEIN'}")
    for r in reasons:
        print(f"  → {r}")
    print(f"{'─'*50}")

    # Plots
    plot_psi(psi_df, suffix=f"_{wave_stem}")
    plot_performance(all_perf_rows, f1_ref, suffix=f"_{wave_stem}")

    # PSI als CSV speichern
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    psi_df.to_csv(REPORTS_DIR / f"psi_summary_{wave_stem}.csv", index=False)

    return {"triggered": triggered, "reasons": reasons, "max_psi": max_psi}


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

    # 4. Sequenzielles Wave-by-Wave Monitoring mit kumulativem Retraining
    # Wave-History laden (für sequenziellen CLI-Modus)
    WAVE_HISTORY_LOG.parent.mkdir(parents=True, exist_ok=True)
    if WAVE_HISTORY_LOG.exists():
        historical_waves = [Path(p.strip()) for p in WAVE_HISTORY_LOG.read_text().splitlines() if p.strip()]
    else:
        historical_waves = []

    all_perf_rows = []
    last_result   = {"triggered": False, "reasons": [], "max_psi": 0.0}

    for i, wave in enumerate(waves):
        # Bisherige Waves aus History + alle bereits in diesem Lauf verarbeiteten + aktuelle
        waves_so_far = historical_waves + list(wave_paths[: i + 1])

        print(f"\n{'='*60}")
        print(f"Schritt {i+1}/{len(waves)}: {wave['name']}")
        print(f"{'='*60}")

        result = _monitor_single_wave(wave, train_df, model, model_version, f1_ref, all_perf_rows)
        last_result = result

        # 5. Retraining mit allen bisherigen Waves, dann weiter zur nächsten Wave
        if result["triggered"] and auto_retrain:
            print(f"\nStarte Retraining mit {len(waves_so_far)} Wave(s): "
                  f"{[p.name for p in waves_so_far]}")
            subprocess_result = subprocess.run(
                [sys.executable, "src/training/retrain.py", "--waves"]
                + [str(p) for p in waves_so_far]
            )
            if subprocess_result.returncode == 0:
                print("Retraining abgeschlossen. Lade aktualisiertes Modell ...")
                # Modell & F1-Referenz nach Retraining neu laden
                try:
                    import mlflow
                    mlflow.set_tracking_uri(TRACKING_URI)
                    client        = mlflow.MlflowClient()
                    version       = client.get_model_version_by_alias(MODEL_NAME, "production")
                    model_version = f"v{version.version}"
                    model         = mlflow.sklearn.load_model(f"runs:/{version.run_id}/model")
                    run           = client.get_run(version.run_id)
                    f1_ref        = run.data.metrics.get("test_f1") or f1_ref
                    print(f"Neues Modell geladen: {model_version}, F1-Ref: {f1_ref:.4f}")
                except Exception as e:
                    print(f"WARNUNG: Modell-Reload fehlgeschlagen ({e}), fahre mit altem Modell fort.")
            else:
                print(f"Retraining fehlgeschlagen (Exit {subprocess_result.returncode}).")
        elif result["triggered"]:
            print(f"\nTrigger aktiv, aber --no-retrain gesetzt.")
            print(f"Manuell: python src/training/retrain.py --waves "
                  f"{' '.join(str(p) for p in waves_so_far)}")

        # Wave in History schreiben (nur wenn noch nicht vorhanden)
        wave_path = wave_paths[i]
        if wave_path not in historical_waves:
            with open(WAVE_HISTORY_LOG, "a") as f:
                f.write(str(wave_path) + "\n")
            historical_waves.append(wave_path)

    return last_result


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
