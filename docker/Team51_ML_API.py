import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Lee variables de entorno
MODEL_URI = os.environ.get("MODEL_URI")  # p.ej. "models:/my-model/Production"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

if not MODEL_URI:
    raise RuntimeError("Falta la variable de entorno MODEL_URI")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Carga el modelo pyfunc al iniciar el servicio
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Equipo 51 ML API", version="1.0.0")

class InferencePayload(BaseModel):
    # Formato tipo /invocations de MLflow: columns + data
    columns: list[str]
    data: list[list[float | int | str | None]]

@app.get("/health")
def health():
    return {"estatus": "ok"}

@app.post("/predict")
def predict(payload: InferencePayload):
    df = pd.DataFrame(payload.data, columns=payload.columns)
    preds = model.predict(df)

    # Convertir a lista serializable
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    return {"predictions": preds}