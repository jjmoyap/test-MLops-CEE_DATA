# ============================================================
# training_loop.py — Entrenamiento y registro de modelos ML
# ============================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow

from src.visualization.plot_utils import Visualizer
from src.models.train_models import PipelineML
from src.models.mlflow import MLflowTracker
from src.features.feature_engineering import FeatureEngineering
from collections import Counter



def run_training_loop(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
    model_dir,
    experiment_name,
    tracking_uri,
):
    """
    Ejecuta entrenamiento completo para cada modelo, guardando solo el mejor global.
    Usa FeatureEngineering + StandardScaler + SMOTE dentro del pipeline
    para evitar fugas y manejar desbalance.
    """
    ml_tracker = MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
    pipeline_ml = PipelineML(
        model_dir=model_dir,
        mlflow_experiment=experiment_name,
        cv=5,
    )
    viz = Visualizer()

    # Asumimos que X_train ya viene con columnas numéricas (scores, MeanScore, DiffScore)
    # y cols categóricas definidas en categorical_cols (pasadas desde data_preparation).
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [
        col
        for col in X_train.columns
        if col not in numeric_cols
    ]

    print(f"[DEBUG] Columnas numéricas: {numeric_cols}")
    print(f"[DEBUG] Columnas categóricas: {categorical_cols}")

    results = []

    # ============================================================
    # 1️. Entrenar y evaluar cada modelo
    # ============================================================
    for model_name, model_dict in models.items():
        print(f"\nEntrenando {model_name}...")
        print("-" * 60)

        # 1. Definir pipeline con FeatureEngineering + StandardScaler + SMOTE + clasificador
        fe_step = FeatureEngineering(
            categorical_cols=categorical_cols,
            rare_threshold=0.03,
            alpha=10.0,
        )

        # ======================================================
        # 1. Configurar SMOTE SELECTIVO (solo para clase 2)
        # ======================================================
        class_counts = Counter(y_train)
        target_class = 2  # Excellent

        if target_class in class_counts:
            # igualar la clase 2 con la clase 1 (Good/Vg)
            desired_n_class2 = class_counts[1]
            sampling_strategy = {target_class: desired_n_class2}
        else:
            sampling_strategy = "not minority"  # fallback seguro

        print(f"[INFO] SMOTE selective sampling_strategy: {sampling_strategy}")

        # ======================================================
        # 2. Pesos de clase dinámicos (solo si el modelo es CatBoost)
        # ======================================================
        # Pesos = inverso de la frecuencia
        n_total = sum(class_counts.values())
        n_cls = len(class_counts)

        class_weights_dict = {
            cls: n_total / (n_cls * cnt)
            for cls, cnt in class_counts.items()
        }

        # Orden CatBoost: peso para clase 0,1,2
        class_weights = [
            class_weights_dict.get(0, 1.0),
            class_weights_dict.get(1, 1.0),
            class_weights_dict.get(2, 1.0),
        ]

        # Crear instancia del modelo
        model_instance = model_dict["model"]

        # Si es CatBoost, aplicamos los class_weights
        if "CatBoost" in model_name:
            model_instance.set_params(class_weights=class_weights)
            print("[INFO] Usando class_weights en CatBoost:", class_weights)

        # ======================================================
        # 3. PIPELINE MEJORADO (FE + escalado + SMOTE selectivo + modelo)
        # ======================================================
        pipe = ImbPipeline(
            steps=[
                ("fe", fe_step),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(
                    random_state=42,
                    sampling_strategy=sampling_strategy
                )),
                ("classifier", model_instance),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            print("   -> Registrando parámetros y configuración...")
            ml_tracker.log_pipeline_params(
                model_params=model_dict.get("params", {}),
                preprocessing="FeatureEngineering (rare + target mean) + StandardScaler",
                imbalance_technique="SMOTE",
                cv=pipeline_ml.cv,
            )

            print("   -> Entrenando y evaluando modelo...")
            result = pipeline_ml.train_and_evaluate_model(
                pipe,
                X_train,
                y_train,
                X_test,
                y_test,
                params=model_dict.get("params", {}),
            )

            best_model = result["model"]
            best_score = result["score"]
            best_params = result["params"]
            y_pred = result["y_pred"]

            print(f"   -> Mejor F1_CV {model_name}: {best_score:.4f}")
            print(f"   -> Mejores hiperparámetros: {best_params}")

            results.append(
                {
                    "model_name": model_name,
                    "model": best_model,
                    "score": best_score,
                    "params": best_params,
                    "y_pred": y_pred,
                }
            )

            # Visualizaciones
            viz.log_confusion_matrix(
                y_test,
                y_pred,
                title=f"Matriz de Confusión: {model_name}",
            )
            viz.log_metrics_bar(
                {model_name: best_score},
                title=f"Métricas: {model_name}",
            )

            print("   -> Registrando métricas y artefactos en MLflow...")
            ml_tracker.log_model_results(
                model=best_model,
                metrics={"F1_CV": best_score},
                best_params=best_params,
                name="model",
                save_path=None,
                X_sample=X_test[:1],
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

        print("\n" + "#" * 70)
        print(f"  Mejor modelo global: {best_model_name} — F1_CV = {best_score:.4f}")
        print("#" * 70 + "\n")

        # Guardar solo el mejor modelo global
        best_model_path = pipeline_ml.save_model(
            best_model,
            name=f"best_model_global_{best_model_name}",
        )
        print(f"Modelo global guardado en: {best_model_path}")

        # Registrar información en MLflow
        with mlflow.start_run(run_name=f"Best_Global_Model_{best_model_name}"):
            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_metric("best_F1_CV", best_score)
            mlflow.log_artifact(best_model_path)
            mlflow.set_tag("best_model_global", True)

    # ============================================================
    # 3️. Guardar resumen general
    # ============================================================
    summary = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "F1_CV": r["score"],
                "Best_Params": r["params"],
            }
            for r in results
        ]
    )

    pipeline_ml.results_summary = summary.to_dict(orient="records")

    return summary
