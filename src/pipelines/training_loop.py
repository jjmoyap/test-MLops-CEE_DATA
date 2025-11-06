# ============================================================
# training_loop.py — Entrenamiento y registro de modelos ML
# ============================================================

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
    """Ejecuta entrenamiento completo para cada modelo."""
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
            best_model, best_score, best_params, save_path, y_pred = pipeline_ml.train_and_evaluate_model(
                pipe, X_train, y_train, X_test, y_test,
                params=model_dict.get("params", {})
            )

            # Visualizaciones
            viz.log_confusion_matrix(y_test, y_pred, title=f"Matriz de Confusión: {model_name}")
            viz.log_metrics_bar({model_name: best_score}, title=f"Métricas: {model_name}")

            print(f"   -> Registrando métricas y artefactos en MLflow...")
            ml_tracker.log_model_results(
                model=best_model,
                metrics={"F1_CV": best_score},
                best_params=best_params,
                name="model",
                save_path=save_path,
                X_sample=X_test[:1]
            )

        print("-" * 60)

    # Guardar resumen
    summary = pd.DataFrame(pipeline_ml.results_summary)
    return summary
