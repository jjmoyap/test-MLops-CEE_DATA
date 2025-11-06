# ============================================================
# ml_pipeline.py — Orquestador principal del pipeline ML
# ============================================================

import sys
from pathlib import Path

# Determinar la raíz del proyecto (dos niveles arriba)
root_path = Path(__file__).resolve().parents[2]

# Agregarla al path si no está
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

print(f"[DEBUG] Proyecto raíz añadido a sys.path: {root_path}")

from datetime import datetime
from src.pipelines.config import (
    root_path, data_path, model_dir,
    tracking_uri, experiment_name, results_dir
)
from src.models.model_configs import models
from src.pipelines.data_preparation import load_and_prepare_data
from src.pipelines.training_loop import run_training_loop
from src.models.mlflow import MLflowServer

# ============================================================
# 1️ Configurar servidor MLflow
# ============================================================

mlflow_server = MLflowServer(
    experiment_name=experiment_name,
    host="127.0.0.1",
    port=5000,
    tracking_uri=tracking_uri,
    backend_store_path=root_path / "mlruns"
)
mlflow_server.start_ui(background=True)

# ============================================================
# 2️ Cargar y preparar datos
# ============================================================

X_train, X_test, y_train, y_test, categorical_cols = load_and_prepare_data(data_path)

# ============================================================
# 3️ Entrenar modelos
# ============================================================

summary = run_training_loop(
    models=models,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    model_dir=model_dir,
    experiment_name=experiment_name,
    tracking_uri=tracking_uri
)

# ============================================================
# 4️ Guardar resumen final
# ============================================================

summary_path = results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
summary.to_csv(summary_path, index=False)

print(f"\nResumen guardado en {summary_path}")
print(f"Experimento '{experiment_name}' registrado en MLflow ({tracking_uri})")
