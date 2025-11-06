# ============================================================
# data_preparation.py — Carga y preparación de datos
# ============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.data.load_data import ARFFLoader
from src.features.feature_engineering import FeatureEngineering


def load_and_prepare_data(data_path):
    """Carga el dataset, aplica feature engineering y retorna splits."""
    loader = ARFFLoader(data_path)
    df = loader.load()

    # Columnas categóricas y ordinales
    categorical_cols = [
        'Gender', 'Caste', 'coaching', 'time', 'Class_ten_education',
        'twelve_education', 'medium', 'Father_occupation', 'Mother_occupation'
    ]
    ordinal_cols = ["Class_ X_Percentage", "Class_XII_Percentage"]
    ord_map = ["Poor", "Average", "Good", "Vg", "Excellent"]

    fe = FeatureEngineering(ordinal_map=ord_map)

    # Combinar categorías raras
    for col in categorical_cols:
        df = fe.combine_rare(df, col, threshold=0.2)

    # Ordinal encoding y Academic_Score
    df = fe.create_ordinal_features(df, ordinal_cols)

    # Agrupar target
    df['Performance_grouped'] = df['Performance'].replace({
        'Average': 'Average/Good', 'Good': 'Average/Good',
        'Vg': 'Vg', 'Excellent': 'Excellent'
    })
    df['Performance_num'] = LabelEncoder().fit_transform(df['Performance_grouped'])

    # Features de frecuencia y mean encoding
    df = fe.add_frequency_features(df, categorical_cols, target_col='Performance_num')

    # Dividir datos
    X = df.drop(columns=['Performance', 'Performance_grouped', 'Performance_num'])
    y = df['Performance_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, categorical_cols
