# ============================================================
# tests/test_data_preparation.py — Pruebas unitarias para data_preparation
# ============================================================

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.pipelines.data_preparation import load_and_prepare_data


@pytest.fixture
def mock_dataframe():
    """Crea un DataFrame simulado con la misma estructura esperada."""
    return pd.DataFrame({
        'Gender': np.random.choice(['M', 'F'], 15),
        'Caste': np.random.choice(['Gen', 'OBC', 'SC'], 15),
        'coaching': np.random.choice(['Yes', 'No'], 15),
        'time': np.random.choice(['Morning', 'Evening'], 15),
        'Class_ten_education': np.random.choice(['Public', 'Private'], 15),
        'twelve_education': np.random.choice(['Public', 'Private'], 15),
        'medium': np.random.choice(['English', 'Hindi'], 15),
        'Father_occupation': np.random.choice(['Teacher', 'Farmer', 'Engineer'], 15),
        'Mother_occupation': np.random.choice(['Housewife', 'Teacher'], 15),
        'Class_ X_Percentage': np.random.choice(['Poor', 'Average', 'Good'], 15),
        'Class_XII_Percentage': np.random.choice(['Average', 'Good', 'Excellent'], 15),
        'Performance': np.random.choice(['Average', 'Good', 'Vg', 'Excellent'], 15)
    })


@patch('src.pipelines.data_preparation.ARFFLoader')
def test_load_and_prepare_data(mock_loader_class, mock_dataframe):
    """Prueba unitaria: valida que el pipeline de preparación devuelve objetos correctos."""
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = mock_dataframe

    X_train, X_test, y_train, y_test, categorical_cols = load_and_prepare_data("fake_path.arff")

    # Verifica tipos
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert isinstance(categorical_cols, list)

    # Verifica tamaño de splits
    total = len(mock_dataframe)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) + len(y_test) == total

    # Verifica columnas clave
    assert all(col in X_train.columns for col in categorical_cols)
    assert 'Performance' not in X_train.columns


@patch('src.pipelines.data_preparation.ARFFLoader')
def test_integration_preparation_flow(mock_loader_class, mock_dataframe):
    """Prueba de integración: valida ejecución completa sin errores."""
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = mock_dataframe

    try:
        load_and_prepare_data("dummy_path.arff")
    except Exception as e:
        pytest.fail(f"El flujo de preparación de datos falló con error: {e}")
