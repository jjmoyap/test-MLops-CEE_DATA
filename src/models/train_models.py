import pickle
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix

from .mlflow import MLflowTracker
from pathlib import Path
from datetime import datetime

class PipelineML:
    """
    Clase que encapsula todo el flujo de entrenamiento, evaluación y logging de modelos ML.
    """

    def __init__(self, model_dir, mlflow_experiment, cv=5):
        """
        Inicializa paths y experimentos.

        Args:
            model_dir (Path): Carpeta donde se guardarán los modelos.
            mlflow_experiment (str): Nombre del experimento en MLflow.
            cv (int): Número de folds para cross-validation.
        """
        self.model_dir = model_dir
        self.cv = cv
        self.results_summary = []
        self.mlflow_experiment = mlflow_experiment

        # Configurar MLflow
        mlflow.set_experiment(mlflow_experiment)

    def train_model(self, model, params, X, y):
        """
        Entrena un modelo usando GridSearchCV con F1-score macro.

        Args:
            model: Estimador de sklearn
            params (dict): Parámetros para GridSearch
            X, y: Datos de entrenamiento

        Returns:
            best_model, best_score, best_params
        """
        f1_scorer = make_scorer(f1_score, average='macro')
        if params:
            gs = GridSearchCV(model, params, cv=self.cv, scoring=f1_scorer, n_jobs=-1)
            gs.fit(X, y)
            return gs.best_estimator_, gs.best_score_, gs.best_params_
        else:
            # Entrenamiento sin GridSearch si no hay parámetros
            model.fit(X, y)
            f1_score_train = f1_score(y, model.predict(X), average='macro')
            return model, f1_score_train, {}

    def evaluate_model(self, model, X_test, y_test, thresholds=None, plot_cm=True, out_dir="results"):
        """
        Evalúa y, opcionalmente, guarda la matriz de confusión normalizada **en disco**.
        NO hace logging a MLflow; solo retorna lo necesario.
        Devuelve: (y_pred, cm_path)
        """
        # Predicción (con umbrales si aplica)
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

        # Reporte rápido a consola (opcional)
        try:
            print(classification_report(y_test, y_pred))
        except Exception:
            pass

        cm_path = None

        return y_pred, cm_path

    def save_model(self, model, name):
        """
        Guarda modelo serializado con timestamp.

        Args:
            model: Estimador entrenado
            name (str): Nombre del modelo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = self.model_dir / f"{name}_best_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        return save_path

    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test, params=None):
        """
        Orquesta el ciclo y REGISTRA en MLflow de forma uniforme.
        Devuelve: best_model, best_score, best_params, save_path, y_pred
        """
        if params is None:
            params = {}
        # Entrenamiento con Validacion cruzada o GridSearch
        best_model, best_score, best_params = self.train_model(model, params, X_train, y_train)

        y_pred, cm_path = self.evaluate_model(best_model, X_test, y_test, plot_cm=True, out_dir="results")

        classifier_name = type(model.named_steps['classifier']).__name__
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(Path(self.model_dir) / f"{classifier_name}_best_model__{ts}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(best_model, f)

        tracker = MLflowTracker(experiment_name=self.mlflow_experiment)
        params_log = {"model_name": classifier_name}
        if best_params:
            # evita colisiones y deja claro que son óptimos
            params_log.update({f"best_{k}": v for k, v in best_params.items()})

        metrics_log = {"F1_CV": float(best_score)}

        tracker.log_model_results(
            model=best_model,
            metrics=metrics_log,
            best_params=best_params or {},
            artifact_path="models",
            save_path=save_path,
        )
        if cm_path:
            tracker.log_confusion_matrix(
                y_test=y_test, y_pred=y_pred, artifact_path=cm_path)

        return best_model, best_score, best_params, save_path, y_pred
