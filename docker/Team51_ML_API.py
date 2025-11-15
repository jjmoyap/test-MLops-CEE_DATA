from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import os
from typing import Any, List, Dict

# ============================================================
# 1. Configuración y Carga de Variables de Entorno
# ============================================================

# Nombre del modelo en el MLflow Model Registry (debe coincidir con tu training_loop.py)
MODEL_NAME = "best_model_global_RandomForest_20251109_1602" 

# Obtener las URIs del entorno (crucial para Docker)
# MLFLOW_MODEL_URI: models:/StudentPerformancePrediction/Production
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", f"models:/{MODEL_NAME}/Production") 

# MLFLOW_TRACKING_URI: http://host.docker.internal:8080 definido en Dockerfile
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") 

# ============================================================
# 2. Inicialización de MLflow y Carga del Modelo
# ============================================================

model = None
try:
    if MLFLOW_TRACKING_URI:
        # Establece la conexión al servidor de MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"MLflow Tracking URI configurado en: {MLFLOW_TRACKING_URI}")
    
    # Carga el modelo desde el Registry (Production) o desde la URI especificada
    print(f"Cargando el modelo desde: {MLFLOW_MODEL_URI}")
    model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
    print("✅ Modelo de MLflow cargado exitosamente.")
    
except Exception as e:
    print(f"❌ ERROR: No se pudo cargar el modelo de MLflow. Verifique el tracking URI y el nombre/etapa del modelo. Error: {e}")
    # En un entorno de producción, podrías querer terminar la aplicación aquí si el modelo es vital.

# ============================================================
# 3. Inicialización de FastAPI
# ============================================================
app = FastAPI(
    title="MLflow Model Prediction API",
    description=f"API para obtener predicciones usando el modelo '{MODEL_NAME}' en etapa 'Production'.",
    version="1.0.0"
)

# ============================================================
# 4. Definición del Esquema de Datos (Pydantic)
# ============================================================

# Define la estructura de las características de entrada
class DataIn(BaseModel):
    # Usamos Field con 'example' para que se muestre en la documentación de Swagger/Redoc
    Gender: str = Field(..., example="Male")
    Caste: str = Field(..., example="OBC")
    coaching: str = Field(..., example="Yes")
    time: str = Field(..., example="1-2 hours")
    Class_ten_education: str = Field(..., example="CBSE")
    twelve_education: str = Field(..., example="State Board")
    medium: str = Field(..., example="English")
    Father_occupation: str = Field(..., example="Private Service")
    Mother_occupation: str = Field(..., example="Housewife")
    Class_X_Percentage: str = Field(..., example="Good")
    Class_XII_Percentage: str = Field(..., example="Vg")

# Contenedor para la solicitud POST (coincide con tu JSON de curl)
class PredictionIn(BaseModel):
    data: DataIn

# Esquema de la respuesta de la predicción
class PredictionOut(BaseModel):
    prediction: List[Any]
    status: str
    model_name: str

# ============================================================
# 5. Endpoints de la API
# ============================================================

@app.get("/health")
async def health_check():
    """Verifica si la API está en funcionamiento y si el modelo está cargado."""
    model_loaded = model is not None
    return {
        "api_status": "running",
        "model_loaded": model_loaded,
        "model_uri": MLFLOW_MODEL_URI,
        "tracking_uri": MLFLOW_TRACKING_URI
    }

@app.post("/predict", response_model=PredictionOut)
async def predict_endpoint(input_data: PredictionIn):
    """
    Recibe los datos de entrada en formato JSON y devuelve la predicción del modelo.
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Servicio no disponible: El modelo de MLflow no se pudo cargar."
        )

    try:
        # Convertir el objeto Pydantic (input_data.data) a un diccionario
        data_dict = input_data.data.model_dump()

        # Convertir a DataFrame de Pandas (formato esperado por el modelo MLflow)
        # Nota: Pandas es importante para mantener el orden y los nombres de las columnas
        input_df = pd.DataFrame([data_dict])

        # Realizar la predicción
        prediction = model.predict(input_df)

        # Devolver el resultado
        return {
            "prediction": prediction.tolist(), # Convertir numpy array a lista para JSON
            "status": "success",
            "model_name": MODEL_NAME
        }

    except Exception as e:
        # Manejo de errores durante el preprocesamiento o la predicción
        raise HTTPException(
            status_code=500, 
            detail=f"Error durante la predicción: {str(e)}"
        )

# Para ejecutar localmente (sin Docker):
# uvicorn app:app --reload --host 0.0.0.0 --port 8000