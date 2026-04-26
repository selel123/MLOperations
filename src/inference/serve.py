from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import uvicorn

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
TRACKING_URI   = "http://46.225.163.17:5000"
MODEL_NAME     = "LGBM-RiskClassification"

# ---------------------------------------------------------------------------
# Modell beim Start laden
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI))
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")

print(f"✅ Modell geladen")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title=MODEL_NAME,
    description="Vorhersage von Risiken mit LightGBM",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Input Schema – alle 12 Features aus train.py
# ---------------------------------------------------------------------------
class InputData(BaseModel):
    # LOG_FEATURES
    transaction_volume: float
    processing_time_hours: float
    historical_incidents_90d: int
    change_requests_30d: int
    # PLAIN_FEATURES
    open_cases_count: int
    customer_tenure_months: float
    missing_docs_flag: int
    high_priority_source_flag: int
    # CATEGORICAL_FEATURES
    region: str
    channel: str
    customer_segment: str
    product_line: str

# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME
    }

# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([{
        # LOG_FEATURES
        "transaction_volume":        data.transaction_volume,
        "processing_time_hours":     data.processing_time_hours,
        "historical_incidents_90d":  data.historical_incidents_90d,
        "change_requests_30d":       data.change_requests_30d,
        # PLAIN_FEATURES
        "open_cases_count":          data.open_cases_count,
        "customer_tenure_months":    data.customer_tenure_months,
        "missing_docs_flag":         data.missing_docs_flag,
        "high_priority_source_flag": data.high_priority_source_flag,
        # CATEGORICAL_FEATURES
        "region":                    data.region,
        "channel":                   data.channel,
        "customer_segment":          data.customer_segment,
        "product_line":              data.product_line,
        }])
    
        prediction = model.predict(df)[0]
        probability = float(model.predict_proba(df)[0][1])

        return {"risk_flag": int(prediction), "probability": round(probability, 4), "model_name": MODEL_NAME}
    
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Fehler bei der Vorhersage: {str(e)}")

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)