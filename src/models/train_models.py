import pickle
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix

class PipelineML:
    """
    Clase que encapsula todo el flujo de entrenamiento, evaluación y logging de modelos ML.
    """

    def __init__(self, model_dir, mlflow_experiment="default", cv=5):
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

    def evaluate_model(self, model, X_test, y_test, thresholds=None, plot_cm=True):
        """
        Evalúa un modelo, imprime métricas y grafica matriz de confusión.

        Args:
            model: Estimador entrenado
            X_test, y_test: Datos de test
            thresholds (dict): Umbrales opcionales para predict_proba
            plot_cm (bool): Si True, genera matriz de confusión

        Returns:
            y_pred: Predicciones finales
        """
        if hasattr(model, "predict_proba") and thresholds:
            probs = model.predict_proba(X_test)
            y_pred_adj = []
            for p in probs:
                candidates = [i for i, prob in enumerate(p) if prob >= thresholds.get(i, 0.5)]
                if candidates:
                    y_pred_adj.append(candidates[np.argmax([p[i] for i in candidates])])
                else:
                    y_pred_adj.append(np.argmax(p))
            y_pred = np.array(y_pred_adj)
        else:
            y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(6,5))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
            plt.title('Matriz de Confusión Normalizada')
            plt.ylabel("True")
            plt.xlabel("Pred")
            plt.show()
            # Log figura en MLflow
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

        return y_pred

    def save_model(self, model, name):
        """
        Guarda modelo serializado con timestamp.

        Args:
            model: Estimador entrenado
            name (str): Nombre del modelo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = self.model_dir / f"{name}_best_model_{timestamp}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        return save_path

    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test, params=None):
        """
        Función principal para entrenar y evaluar un modelo con logging en MLflow.

        Args:
            model: Estimador sklearn
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de test
            params (dict, optional): Parámetros para GridSearchCV. Default None.

        Returns:
            best_model, best_score, best_params
        """
        if params is None:
            params = {}

        best_model, best_score, best_params = self.train_model(model, params, X_train, y_train)
        self.evaluate_model(best_model, X_test, y_test)

        # Log en MLflow
        if best_params:
            mlflow.log_params(best_params)
        mlflow.log_metric("F1_CV", best_score)
        mlflow.sklearn.log_model(best_model, artifact_path="models")

        self.results_summary.append({
            "Model": type(model).__name__,
            "F1_CV": best_score
        })

        # Guardar modelo
        save_path = self.save_model(best_model, type(model).__name__)
        mlflow.log_artifact(save_path)

        return best_model, best_score, best_params
