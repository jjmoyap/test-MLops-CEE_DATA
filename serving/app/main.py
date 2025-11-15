# ============================================================
# main.py — API de servicio del modelo con FastAPI
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import pandas as pd
import os
import glob

# 🔧 Ajuste de importaciones (antes eran src.api.*)
from serving.app.schemas import PredictionRequest, PredictionResponse
from serving.app.utils import load_model, prepare_input_data, make_prediction

# ============================================================
# 1️. Inicialización de la aplicación
# ============================================================

app = FastAPI(
    title="CEE Student Performance Prediction API",
    description="Servicio FastAPI para predecir el desempeño académico de estudiantes",
    version="1.0.0"
)

# ============================================================
# 2️. Cargar el modelo local usando utils.py
# ============================================================
MODELS_DIR = "models"
pattern = os.path.join(MODELS_DIR, "best_model_*.pkl")
matches = glob.glob(pattern)
try:
    model, model_version = load_model(matches[0]) if matches else (None, "N/A")
except Exception as e:
    model = None
    model_version = "N/A"
    print(f"   No se pudo cargar el modelo: {e}")

# ============================================================
# 3️. Endpoint principal de predicción
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Realiza una predicción del grupo de desempeño académico.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible actualmente.")

    try:
        # Convertir input en DataFrame
        input_data = prepare_input_data(request.data.dict())

        # Realizar predicción usando utils
        prediction, prob = make_prediction(model, input_data)

        return PredictionResponse(
            prediction=prediction,
            probability=prob,
            model_version=model_version
        )

    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=f"Error de validación: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

# ============================================================
# 4️. Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "API de predicción de desempeño académico (CEE Project)",
        "status": "running",
        "model_version": model_version
    }
