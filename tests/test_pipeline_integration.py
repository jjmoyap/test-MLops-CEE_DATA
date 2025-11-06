# ============================================================
# tests/test_pipeline_integration.py — Prueba de integración end-to-end
# ============================================================

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.pipelines import ml_pipeline


@pytest.fixture
def mock_full_dataframe():
    """DataFrame que simula datos reales para el pipeline completo."""
    return pd.DataFrame({
        'Gender': np.random.choice(['M', 'F'], 20),
        'Caste': np.random.choice(['Gen', 'OBC', 'SC'], 20),
        'coaching': np.random.choice(['Yes', 'No'], 20),
        'time': np.random.choice(['Morning', 'Evening'], 20),
        'Class_ten_education': np.random.choice(['Public', 'Private'], 20),
        'twelve_education': np.random.choice(['Public', 'Private'], 20),
        'medium': np.random.choice(['English', 'Hindi'], 20),
        'Father_occupation': np.random.choice(['Teacher', 'Farmer', 'Engineer'], 20),
        'Mother_occupation': np.random.choice(['Housewife', 'Teacher'], 20),
        'Class_ X_Percentage': np.random.choice(['Poor', 'Average', 'Good'], 20),
        'Class_XII_Percentage': np.random.choice(['Average', 'Good', 'Excellent'], 20),
        'Performance': np.random.choice(['Average', 'Good', 'Vg', 'Excellent'], 20)
    })


@patch('src.pipelines.data_preparation.ARFFLoader')
@patch('src.pipelines.training_loop.run_training_loop')
def test_full_pipeline_execution(mock_run_training, mock_loader_class, mock_full_dataframe):
    """Prueba de integración completa del pipeline principal."""
    # Mock de carga de datos
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = mock_full_dataframe

    # Mock del entrenamiento
    mock_run_training.return_value = pd.DataFrame({
        "model": ["LogisticRegression"],
        "F1_CV": [0.91]
    })

    # Ejecuta el pipeline principal
    try:
        import importlib
        importlib.reload(ml_pipeline)
    except Exception as e:
        pytest.fail(f"El pipeline completo falló con excepción: {e}")

    # Verifica que se llamaron los componentes principales
    assert mock_loader.load.called, "ARFFLoader.load() debió ser invocado"
    assert mock_run_training.called, "run_training_loop() debió ser invocado"

    # Verifica formato del resultado simulado
    result_df = mock_run_training.return_value
    assert "F1_CV" in result_df.columns or "score" in result_df.columns, \
        "El DataFrame devuelto debe contener métricas de evaluación"
