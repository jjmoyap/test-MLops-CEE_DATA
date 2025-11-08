# ============================================================
# schemas.py — Definición de schemas para la API de predicción
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional

# ============================================================
# 1️. Schema para las características del estudiante
# ============================================================

class StudentFeatures(BaseModel):
    Gender: str = Field(..., description="Género del estudiante (por ejemplo, 'Male' o 'Female')")
    Caste: str = Field(..., description="Grupo de casta o comunidad del estudiante")
    coaching: str = Field(..., description="Indica si el estudiante recibió coaching o clases adicionales ('Yes'/'No')")
    time: str = Field(..., description="Tiempo de estudio diario (por ejemplo, '<1 hour', '1-2 hours')")
    Class_ten_education: str = Field(..., description="Tipo de educación en décimo grado (por ejemplo, 'State Board', 'CBSE')")
    twelve_education: str = Field(..., description="Tipo de educación en duodécimo grado")
    medium: str = Field(..., description="Medio de instrucción (por ejemplo, 'English', 'Hindi')")
    Father_occupation: str = Field(..., description="Ocupación del padre del estudiante")
    Mother_occupation: str = Field(..., description="Ocupación de la madre del estudiante")
    Class_X_Percentage: str = Field(..., description="Categoría de rendimiento en décimo grado ('Poor', 'Average', 'Good', 'Vg', 'Excellent')")
    Class_XII_Percentage: str = Field(..., description="Categoría de rendimiento en duodécimo grado ('Poor', 'Average', 'Good', 'Vg', 'Excellent')")

# ============================================================
# 2️. Schema para la solicitud de predicción
# ============================================================

class PredictionRequest(BaseModel):
    data: StudentFeatures = Field(..., description="Características de un estudiante para predecir su desempeño académico")

# ============================================================
# 3️. Schema para la respuesta de predicción
# ============================================================

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicción del grupo de desempeño (por ejemplo, 'Average/Good', 'Vg', 'Excellent')")
    probability: Optional[float] = Field(None, description="Probabilidad o confianza asociada a la predicción (si el modelo la provee)")
    model_version: str = Field(..., description="Versión del modelo utilizado para la predicción")

    # Configuración para evitar conflictos con protected_namespaces
    model_config = {
        "protected_namespaces": ()
    }
