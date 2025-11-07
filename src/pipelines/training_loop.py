# training_loop.py — Entrenamiento y registro de modelos ML

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow

from src.visualization.plot_utils import Visualizer
from src.models.train_models import PipelineML
from src.models.mlflow import MLflowTracker


def run_training_loop(models, X_train, X_test, y_train, y_test,
                      model_dir, experiment_name, tracking_uri):
    """Ejecuta entrenamiento completo para cada modelo, guardando solo el mejor global."""
    ml_tracker = MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)
    pipeline_ml = PipelineML(model_dir=model_dir, mlflow_experiment=experiment_name, cv=5)
    viz = Visualizer()

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Lista para almacenar resultados de todos los modelos
    results = []

    # ============================================================
    # 1️. Entrenar y evaluar cada modelo
    # ============================================================
    for model_name, model_dict in models.items():
        print(f"Entrenando {model_name}...")

        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model_dict['model'])
        ])

        with mlflow.start_run(run_name=model_name):
            print(f"   -> Registrando parámetros y configuración...")
            ml_tracker.log_pipeline_params(
                model_params=model_dict.get("params", {}),
                preprocessing="StandardScaler + OneHotEncoder",
                imbalance_technique="SMOTE",
                cv=pipeline_ml.cv
            )

            print(f"   -> Entrenando y evaluando modelo...")
            result = pipeline_ml.train_and_evaluate_model(
                pipe, X_train, y_train, X_test, y_test,
                params=model_dict.get("params", {})
            )

            best_model = result["model"]
            best_score = result["score"]
            best_params = result["params"]
            y_pred = result["y_pred"]

            # Guardar resultados parciales en memoria (no archivos)
            results.append({
                "model_name": model_name,
                "model": best_model,
                "score": best_score,
                "params": best_params,
                "y_pred": y_pred
            })

            # Visualizaciones
            viz.log_confusion_matrix(y_test, y_pred, title=f"Matriz de Confusión: {model_name}")
            viz.log_metrics_bar({model_name: best_score}, title=f"Métricas: {model_name}")

            print(f"   -> Registrando métricas y artefactos en MLflow...")
            ml_tracker.log_model_results(
                model=best_model,
                metrics={"F1_CV": best_score},
                best_params=best_params,
                name="model",
                save_path=None,
                X_sample=X_test[:1]
            )

        print("-" * 60)

    # ============================================================
    # 2️. Seleccionar el mejor modelo global
    # ============================================================
    if results:
        best_result = max(results, key=lambda r: r["score"])
        best_model = best_result["model"]
        best_model_name = best_result["model_name"]
        best_score = best_result["score"]

        print(f"\n{'#' * 70}")
        print(f"  Mejor modelo global: {best_model_name} — F1_CV = {best_score:.4f}")
        print(f"{'#' * 70}\n")

        # Guardar solo el mejor modelo global
        best_model_path = pipeline_ml.save_model(best_model, name=f"best_model_global_{best_model_name}")
        print(f"Modelo global guardado en: {best_model_path}")

        # Registrar información en MLflow (opcional pero útil para trazabilidad)
        with mlflow.start_run(run_name=f"Best_Global_Model_{best_model_name}"):
            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_metric("best_F1_CV", best_score)
            mlflow.log_artifact(best_model_path)
            mlflow.set_tag("best_model_global", True)

    # ============================================================
    # 3️. Guardar resumen general
    # ============================================================
    summary = pd.DataFrame([
        {"Model": r["model_name"], "F1_CV": r["score"], "Best_Params": r["params"]}
        for r in results
    ])

    pipeline_ml.results_summary = summary.to_dict(orient="records")

    return summary
