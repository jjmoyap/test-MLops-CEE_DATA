# ============================================================
# data_preparation.py — SAFE TRAIN/TEST SPLIT + NUMERIC GRADES
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.load_data import ARFFLoader
import joblib


def load_and_prepare_data(data_path):
    """
    Carga el dataset ARFF, crea el target numérico (3 clases) y
    features numéricas a partir de los porcentajes.
    """

    # 1. Cargar dataset desde ARFF
    loader = ARFFLoader(data_path)
    df = loader.load()

    # 2. Limpieza básica
    df.columns = df.columns.str.strip()
    df = df.fillna("Unknown")

    
    perf_map_3 = {
        "Average": 0,
        "Good": 1,
        "Vg": 1,
        "Excellent": 2,
    }

    df["Performance_num"] = df["Performance"].map(perf_map_3)

    # ====================================================
    # 4. Convertir porcentajes a scores numéricos (ordinales)
    #    usando el mapeo original de 4 niveles
    # ====================================================
    ordinal_map = {
        "Average": 0,
        "Good": 1,
        "Vg": 2,
        "Excellent": 3,
    }

    df["Class_X_score"] = df["Class_ X_Percentage"].map(ordinal_map)
    df["Class_XII_score"] = df["Class_XII_Percentage"].map(ordinal_map)

    # ====================================================
    # 5. Features derivadas
    # ====================================================
    df["MeanScore"] = (df["Class_X_score"] + df["Class_XII_score"]) / 2.0
    df["DiffScore"] = df["Class_X_score"] - df["Class_XII_score"]

    # ====================================================
    # 6. Columnas categóricas restantes
    # ====================================================
    categorical_cols = [
        "Gender",
        "Caste",
        "coaching",
        "time",
        "Class_ten_education",
        "twelve_education",
        "medium",
        "Father_occupation",
        "Mother_occupation",
    ]

    # ====================================================
    # 7. Construir X e y
    # ====================================================
    drop_cols = [
        "Performance",
        "Performance_num",          # target
        "Class_ X_Percentage",
        "Class_XII_Percentage",
    ]

    X = df.drop(columns=drop_cols)
    y = df["Performance_num"]

    # ====================================================
    # 8. Train/test split estratificado
    # ====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ====================================================
    # 9. Guardar metadata para inferencia
    # ====================================================
    os.makedirs("models", exist_ok=True)
    metadata = {
        "categorical_cols": categorical_cols,
        "feature_columns": list(X.columns),
        "perf_map": perf_map_3,           # MAPEADO NUEVO
    }
    joblib.dump(metadata, "models/data_metadata.pkl")

    return X_train, X_test, y_train, y_test, categorical_cols
