# src/models/train_models.py

import pickle
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report
from pathlib import Path

from .mlflow import MLflowTracker


class PipelineML:
    """
    Clase que encapsula todo el flujo de entrenamiento, evaluación y logging de modelos ML.
    """

    def __init__(self, model_dir, mlflow_experiment, cv=5):
        """
        Inicializa paths y experimentos.
        """
        self.model_dir = model_dir
        self.cv = cv
        self.results_summary = []
        self.mlflow_experiment = mlflow_experiment

        mlflow.set_experiment(mlflow_experiment)


    ########################################################################
    # Entrenamiento base
    ########################################################################
    def train_model(self, model, params, X, y):
        """Entrena un modelo con GridSearchCV usando F1-score macro."""
        f1_scorer = make_scorer(f1_score, average='macro')
        if params:
            gs = GridSearchCV(model, params, cv=self.cv, scoring=f1_scorer, n_jobs=-1)
            gs.fit(X, y)
            return gs.best_estimator_, gs.best_score_, gs.best_params_
        else:
            model.fit(X, y)
            f1_score_train = f1_score(y, model.predict(X), average='macro')
            return model, f1_score_train, {}


    ########################################################################
    # Evaluación local
    ########################################################################
    def evaluate_model(self, model, X_test, y_test, thresholds=None):
        """Evalúa el modelo y retorna las predicciones."""
        if hasattr(model, "predict_proba") and thresholds:
            probs = model.predict_proba(X_test)
            y_pred_adj = []
            for p in probs:
                candidates = [i for i, prob in enumerate(p) if prob >= thresholds.get(i, 0.5)]
                if candidates:
                    y_pred_adj.append(candidates[np.argmax([p[i] for i in candidates])])
                else:
                    y_pred_adj.append(int(np.argmax(p)))
            y_pred = np.array(y_pred_adj)
        else:
            y_pred = model.predict(X_test)

        print("\n=== Reporte de Clasificación ===")
        print(classification_report(y_test, y_pred))
        return y_pred


    ########################################################################
    # Guardar modelo localmente
    ########################################################################
    def save_model(self, model, name="best_model_global"):
        """Guarda modelo serializado con timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        save_path = self.model_dir / f"{name}_{timestamp}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        return save_path


    ########################################################################
    # Pipeline completo: entrenamiento, evaluación y logging
    ########################################################################
    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test, params=None):
        """
        Ejecuta el flujo completo: entrenamiento, evaluación y logging en MLflow.
        Retorna información del modelo (sin guardar localmente aquí).
        """
        if params is None:
            params = {}

        # Entrenamiento
        best_model, best_score, best_params = self.train_model(model, params, X_train, y_train)

        # Evaluación
        y_pred = self.evaluate_model(best_model, X_test, y_test)

        # Registrar resultados en MLflow
        classifier_name = type(model.named_steps['classifier']).__name__
        tracker = MLflowTracker(experiment_name=self.mlflow_experiment)
        metrics_log = {"F1_CV": float(best_score)}

        tracker.log_model_results(
            model=best_model,
            metrics=metrics_log,
            best_params=best_params or {},
            name=classifier_name,
            save_path=None,  # No se guarda aquí
            X_sample=X_test[:1]  # Ejemplo pequeño para inferir firma
        )

        # Matriz de confusión como artefacto
        tracker.log_confusion_matrix(y_test=y_test, y_pred=y_pred, artifact_path=f"confusion_matrix_{classifier_name}.png")

        return {
            "name": classifier_name,
            "model": best_model,
            "score": best_score,
            "params": best_params,
            "y_pred": y_pred
        }
