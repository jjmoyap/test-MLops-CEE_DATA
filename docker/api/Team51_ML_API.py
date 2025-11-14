import os
from typing import List

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------
# Configuración de MLflow (se puede sobreescribir con envs)
# Es un hola mundo de un modelo de regresión simple de calificaciones
# de estudiantes basado en su edad y horas de estudio.
# Un Hola Mundo básico para demostrar despliegue de modelos con MLflow y FastAPI.
# ---------------------------------------------------------

MODEL_URI = "/ml/model"#os.getenv("MODEL_URI")
PORT=8880

print(f"Usando MODEL_URI = {MODEL_URI}")

#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Cargar el modelo desde MLflow Model Registry
print("Cargando modelo desde MLflow...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("✅ Modelo cargado correctamente.")

# ---------------------------------------------------------
# Definición del API con FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Prueba de API con MLflow - Team 51",
    description="API para predecir la calificación de un estudiante usando un modelo registrado en MLflow.",
    version="1.0.0",
)


class Student(BaseModel):
    edad: int
    horas_estudio: float


class StudentsBatch(BaseModel):
    students: List[Student]


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/hola_mundo")
def hola_mundo():
    return {"mensaje": "Hola Mundo desde la API de Team 51 con MLflow! y FastAPI"}


@app.post("/predict_one")
def predict_one(student: Student):
    """
    Predice la calificación de un solo estudiante.
    """
    df = pd.DataFrame([student.dict()])
    pred = model.predict(df)[0]
    return {
        "input": student.dict(),
        "calificacion_predicha": float(pred),
    }


@app.post("/predict_batch")
def predict_batch(batch: StudentsBatch):
    """
    Predice la calificación para varios estudiantes a la vez.
    """
    data = [s.dict() for s in batch.students]
    df = pd.DataFrame(data)
    preds = model.predict(df)

    return {
        "inputs": data,
        "calificaciones_predichas": [float(p) for p in preds],
    }