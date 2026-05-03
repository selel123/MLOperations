"""
Setzt die Model Card als Description für LGBM-RiskClassification in der MLflow Registry.

Verwendung:
    python set_model_card.py
"""

from mlflow import MlflowClient

TRACKING_URI = "http://46.225.163.17:5000"
MODEL_NAME   = "LGBM-RiskClassification"

MODEL_CARD = """## LGBM-RiskClassification
**Autoren:** Elezovic, Schmid — Sommersemester 2026

---

### Was macht dieses Modell?
Erkennt Versicherungskunden mit erhöhtem Kündigungsrisiko innerhalb der nächsten 90 Tage.
Auf Basis von Vertragsdaten wird jeden Monat automatisch eine priorisierte Liste für das Retention-Team der Allianz VersicherungsAG erstellt — sodass gefährdete Kunden gezielt angesprochen werden können, bevor sie kündigen.

---

### Fehlklassifikationsrisiken

| Typ | Konsequenz | Bewertung |
|-----|-----------|-----------|
| **False Negative** | Churn-Kunde nicht erkannt → kein Retention-Angebot → Vertrag verloren | Kritisch |
| **False Positive** | Nicht-Churn-Kunde fälschlich markiert → unnötiger Kontakt, Rabattangebot | Vertretbar |

---

### Bekannte Einschränkungen
- **Feature-Dominanz:** Starke Abhängigkeit von `transaction_volume` — das Modell reagiert sensibel auf strukturelle Veränderungen dieses Wertes (PSI-Monitoring erforderlich)
- **Schwache Einzelkorrelationen:** Max. Pearson r = 0,355 — die Datenbasis begrenzt die erreichbare Performance unabhängig vom Algorithmus
- **Zeitlicher Gültigkeitsbereich:** Trainiert auf Jan–Mrz 2025 — saisonale Effekte nicht abgedeckt, regelmäßiges Retraining erforderlich
"""

def main():
    client = MlflowClient(TRACKING_URI)

    client.update_registered_model(
        name=MODEL_NAME,
        description=MODEL_CARD,
    )

    print(f"Model Card erfolgreich gesetzt für: {MODEL_NAME}")
    print(f"MLflow: {TRACKING_URI}")

if __name__ == "__main__":
    main()