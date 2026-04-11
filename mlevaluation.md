# ML Evaluation Report

**Experiment:** classification-ml (ID: 1)
**Datenschnitt:** split_date = 2025-04-01 | Quelle: train.py

---

## Metriken im Vergleich

| Modell   | ROC-AUC | F1     | Recall | Precision | Hyperparameter                          |
|----------|---------|--------|--------|-----------|-----------------------------------------|
| RF_tuned | 0.7156  | 0.5613 | 0.6026 | 0.5253    | max_depth=5, n_estimators=200           |
| LR_tuned | 0.7157  | 0.5587 | 0.6717 | 0.4783    | C=10                                    |
| GB_tuned | 0.7010  | 0.4557 | 0.3734 | 0.5846    | lr=0.1, max_depth=5, n_estimators=200   |

---

## Interpretation pro Modell

### Random Forest (RF_tuned)

- Bestes F1-Score (0.5613) — beste Balance zwischen Precision und Recall
- ROC-AUC praktisch identisch mit LR (0.7156 vs. 0.7157, Differenz negligibel)
- Solider Recall (0.6026) ohne zu viele False Positives
- Vielseitigster Kandidat

### Logistic Regression (LR_tuned)

- Höchster Recall (0.6717) — erkennt die meisten tatsächlichen Positives
- Dafür niedrigste Precision (0.4783) — mehr False Positives
- ROC-AUC minimal am besten (0.7157)
- Empfehlenswert wenn Missed Positives teuer sind (z.B. Churn, Fraud, Diagnose)

### Gradient Boosting (GB_tuned)

- Höchste Precision (0.5846), aber auf Kosten des Recalls (0.3734)
- Damit deutlich schlechtestes F1 (0.4557) und schlechteste ROC-AUC (0.7010)
- Das Modell ist sehr "konservativ" — es macht wenige positive Vorhersagen, trifft damit aber öfter
- Schwächstes Modell im Gesamtbild

---

## Empfehlung

**Bestes Modell: RF_tuned**, da es:

1. Das höchste F1 hat (ausgewogenste Performance)
2. ROC-AUC quasi identisch mit LR (Diskriminierungsfähigkeit gleich gut)
3. Eine akzeptable Precision-Recall-Balance bietet

> **Ausnahme:** Wenn die Aufgabe maximalen Recall erfordert (kein Positives verpassen ist kritisch), wäre LR_tuned vorzuziehen — trotz mehr False Positives.

GB_tuned ist in diesem Setup klar unterlegen und müsste durch weitere Hyperparameter-Suche verbessert werden (z.B. niedrigeres max_depth oder mehr Regularisierung).

---

## Modellempfehlung auf Basis der EDA

### Was die EDA über diesen Datensatz sagt

| Eigenschaft         | Befund                                     | Relevanz für Modellwahl                                  |
|---------------------|--------------------------------------------|----------------------------------------------------------|
| Klassenverteilung   | 65.6% / 34.4% (Imbalance Ratio 1.91)       | Moderate Imbalance — Accuracy ist irreführend            |
| Hauptmetrik         | F1 (Klasse 1), sekundär Recall             | Explizit dokumentiert: False Negatives sind geschäftskritischer |
| Ausreißer           | bis 7.9% (historical_incidents_90d)        | Baumbasierte Modelle robust — LR reagiert empfindlich    |
| Schiefe Features    | 4 Features (skew 0.77–1.15)                | log1p-Transformation hilft LR stärker als Bäumen         |
| Multikollinearität  | Keine (kein Paar > Threshold)              | Kein Problem für keines der Modelle                      |
| Top-Prediktor       | transaction_volume (r=0.355)               | Starkes, lineares Signal                                 |
| Kategorische Features | Alle 4 signifikant (Chi²-Test, p<0.001) | One-Hot Encoding benötigt                                |

---

## Abgleich EDA → Modellverhalten

| Kriterium                    | RF_tuned   | LR_tuned   | GB_tuned      |
|------------------------------|------------|------------|---------------|
| F1 (Hauptmetrik)             | 0.5613     | 0.5587     | 0.4557        |
| Recall (geschäftskritisch)   | 0.6026     | 0.6717     | 0.3734 ❌     |
| Robust gegen Ausreißer       | Ja (Baum)  | Nein       | Ja (Baum)     |
| Profitiert von log1p         | Kaum       | Stark      | Kaum          |
| Precision                    | 0.5253     | 0.4783     | 0.5846        |
| ROC-AUC                      | 0.7156     | 0.7157     | 0.7010        |

---

## Empfehlung (EDA-basiert)

**Produktionsmodell: RF_tuned**

Die EDA definiert F1 als Hauptmetrik — RF gewinnt dort klar. Gleichzeitig ist RF von Natur aus robust gegen die in der EDA identifizierten Ausreißer und schiefen Verteilungen, ohne aufwändige Transformationen zu benötigen. ROC-AUC ist praktisch identisch mit LR.

**Alternative bei maximaler Recall-Priorität: LR_tuned**

Wenn das Business tatsächlich null Risikofälle verpassen will (False Negative = sehr hohe Kosten), dann LR — der Recall-Vorteil (+6.9 PP gegenüber RF) rechtfertigt die niedrigere Precision. Die log1p-Transformationen aus der EDA kommen LR besonders zugute.

**GB_tuned scheidet aus:** Ein Recall von 0.37 bedeutet, dass ~63% aller echten Risikofälle unerkannt bleiben — das widerspricht direkt dem dokumentierten Anwendungsfall (Frühwarnsystem, Priorisierung von Hochrisikofällen).

---

## Nächster Schritt

Den Entscheidungsschwellenwert von RF_tuned von 0.5 auf ~0.35–0.40 absenken, um Recall zu erhöhen ohne F1 stark zu opfern. Das würde den GB-Recall-Nachteil weiter unterstreichen und RF_tuned noch deutlicher als bestes Modell positionieren.
