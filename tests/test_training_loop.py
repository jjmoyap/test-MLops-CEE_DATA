# ============================================================
# tests/test_training_loop.py — Pruebas unitarias para training_loop
# ============================================================

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.pipelines.training_loop import run_training_loop
import warnings
import os


@pytest.fixture
def mock_train_test_data():
    """Genera datos aleatorios para simular entrenamiento, con suficientes muestras por clase para SMOTE."""
    np.random.seed(42)
    n_features = 5
    n_classes = 3

    # Número de muestras divisible por n_classes
    n_samples_train = n_classes * 20  # 20 muestras por clase
    n_samples_test = n_classes * 7    # 7 muestras por clase

    X_train = pd.DataFrame(
        np.random.rand(n_samples_train, n_features),
        columns=[f"f{i}" for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.rand(n_samples_test, n_features),
        columns=[f"f{i}" for i in range(n_features)]
    )

    # Crear etiquetas con suficiente representación
    y_train = pd.Series(np.repeat(np.arange(n_classes), 20))
    y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)

    y_test = pd.Series(np.repeat(np.arange(n_classes), 7))
    y_test = y_test.sample(frac=1, random_state=42).reset_index(drop=True)

    return X_train, X_test, y_train, y_test


@pytest.fixture
def mock_models():
    """Define un modelo simple para pruebas."""
    return {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=200),
            "params": {"classifier__C": [0.1, 1.0]}
        }
    }


def test_run_training_loop_output(mock_models, mock_train_test_data, tmp_path):
    """Verifica que la función ejecuta entrenamiento completo sin errores."""
    X_train, X_test, y_train, y_test = mock_train_test_data

    # Silenciar warnings de entrenamiento
    warnings.filterwarnings("ignore")

    # Ruta temporal para tracking
    tracking_path = "file:///" + str(tmp_path / "mlruns").replace("\\", "/")

    # Ejecuta el loop de entrenamiento
    summary = run_training_loop(
        models=mock_models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_dir=tmp_path,
        experiment_name="TestExperiment",
        tracking_uri=tracking_path
    )

    # Asegura que no haya errores en ejecución
    assert summary is None or isinstance(summary, pd.DataFrame), \
        "El pipeline debe ejecutarse sin errores y retornar None o un DataFrame opcional."

    # Verifica que se haya guardado al menos un modelo entrenado
    saved_models = [f for f in os.listdir(tmp_path) if f.endswith(".pkl")]
    assert len(saved_models) > 0, "Debe haberse guardado al menos un modelo entrenado."

    # Verifica que se haya creado carpeta de MLflow tracking
    mlruns_dir = tmp_path / "mlruns"
    assert mlruns_dir.exists(), "Debe haberse creado el directorio de seguimiento MLflow."

    # Si devuelve un DataFrame, verifica que contenga métricas
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        expected_metrics = {"F1_CV", "score", "Accuracy", "Precision", "Recall"}
        found_metrics = any(col in summary.columns for col in expected_metrics)
        assert found_metrics, "El resumen debe contener al menos una métrica de desempeño."


def test_run_training_loop_invalid_input(mock_models):
    """Verifica manejo de error si se pasan datos vacíos."""
    with pytest.raises(Exception):
        run_training_loop(
            models=mock_models,
            X_train=pd.DataFrame(),
            X_test=pd.DataFrame(),
            y_train=pd.Series(dtype=int),
            y_test=pd.Series(dtype=int),
            model_dir="dummy_dir",
            experiment_name="TestExperiment",
            tracking_uri="file://dummy"
        )
