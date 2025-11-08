# ============================================================
# utils.py — Funciones auxiliares para FastAPI y modelo ML
# ============================================================

import os
import joblib
import pandas as pd
from datetime import datetime
from typing import Tuple, Any

# ============================================================
# 1️. Cargar modelo de forma segura
# ============================================================

def load_model(model_path: str) -> Tuple[Any, str]:
    """
    Carga el modelo desde un archivo .pkl y retorna el modelo junto con su versión (nombre de archivo).

    Args:
        model_path (str): Ruta completa del modelo.

    Returns:
        Tuple[Any, str]: (modelo cargado, versión del modelo)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    try:
        model = joblib.load(model_path)
        version = os.path.basename(model_path)
        print(f"  Modelo cargado exitosamente: {version}")
        return model, version
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {e}")

# ============================================================
# 2️. Validar y preparar datos de entrada
# ============================================================

def prepare_input_data(input_dict: dict) -> pd.DataFrame:
    """
    Convierte el diccionario de entrada a un DataFrame listo para predicción,
    aplicando exactamente las mismas transformaciones que en el pipeline de entrenamiento.

    Args:
        input_dict (dict): Datos recibidos desde el request.

    Returns:
        pd.DataFrame: DataFrame transformado listo para el modelo.
    """
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([input_dict])

        # Renombrar columnas para coincidir con el pipeline
        df.rename(columns={
            "Class_X_Percentage": "Class_ X_Percentage",
            "Class_XII_Percentage": "Class_XII_Percentage"
        }, inplace=True)

        if df.empty:
            raise ValueError("Los datos de entrada están vacíos.")

        # ====================================================
        # Aplicar las transformaciones del FeatureEngineering
        # ====================================================
        fe_path = os.path.join("models", "feature_engineering.pkl")
        if not os.path.exists(fe_path):
            raise FileNotFoundError(f"No se encontró el transformador en {fe_path}")

        fe = joblib.load(fe_path)

        # Columnas categóricas y ordinales
        categorical_cols = [
            'Gender', 'Caste', 'coaching', 'time', 'Class_ten_education',
            'twelve_education', 'medium', 'Father_occupation', 'Mother_occupation'
        ]
        ordinal_cols = ["Class_ X_Percentage", "Class_XII_Percentage"]

        # Aplicar combinaciones de categorías raras usando los maps guardados
        for col in categorical_cols:
            df = fe.combine_rare(df, col)  # Threshold ya guardado en fe

        # Ordinal encoding y Academic_Score
        df = fe.create_ordinal_features(df, ordinal_cols)  # Usa ordinal encoder guardado

        # Features de frecuencia y mean encoding usando los mappings guardados
        for col in categorical_cols:
            if hasattr(fe, 'freq_maps') and col in fe.freq_maps:
                df[col + '_freq'] = df[col].map(fe.freq_maps[col])
            if hasattr(fe, 'target_mean_maps') and col in fe.target_mean_maps:
                df[col + '_target_mean'] = df[col].map(fe.target_mean_maps[col])

        return df

    except Exception as e:
        raise ValueError(f"Error al preparar los datos de entrada: {e}")

# ============================================================
# 3️. Obtener información del modelo
# ============================================================

def get_model_metadata(model_path: str) -> dict:
    """
    Retorna metadatos del modelo, como fecha de creación, tamaño y nombre.

    Args:
        model_path (str): Ruta del archivo de modelo.

    Returns:
        dict: Información descriptiva del modelo.
    """
    if not os.path.exists(model_path):
        return {"exists": False, "error": "Archivo no encontrado"}

    stats = os.stat(model_path)
    metadata = {
        "model_name": os.path.basename(model_path),
        "size_MB": round(stats.st_size / (1024 * 1024), 2),
        "last_modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "exists": True
    }
    return metadata

# ============================================================
# 4️. Predicción segura (manejo de errores y probabilidad)
# ============================================================

def make_prediction(model, input_df: pd.DataFrame) -> Tuple[str, float]:
    """
    Ejecuta una predicción con el modelo, manejando errores y retornando probabilidad si aplica.

    Args:
        model: Modelo cargado.
        input_df (pd.DataFrame): Datos preprocesados para predicción.

    Returns:
        Tuple[str, float]: (predicción, probabilidad)
    """
    try:
        pred = model.predict(input_df)
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(max(model.predict_proba(input_df)[0]))
        return str(pred[0]), probability
    except Exception as e:
        raise RuntimeError(f"Error al realizar la predicción: {e}")
