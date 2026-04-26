from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import uvicorn


app = FastAPI()

# Modell wird einmal beim Start geladen
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://46.225.163.17:5000"))
model = mlflow.pyfunc.load_model("models:/LGBM-RiskClassification@production")

print(f"✅ Modell geladen")


class InputData(BaseModel):
    transaction_volume: float
    processing_time_hours: float
    historical_incidents_90d: int
    open_cases_count: int
    customer_tenure_months: int
    change_requests_30d: int
    missing_docs_flag: int
    high_priority_source_flag: int
    region: str
    channel: str
    customer_segment: str
    product_line: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": "LGBM-RiskClassification"
    }

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"risk_flag": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)