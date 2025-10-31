import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import mlflow

class Visualizer:
    """
    Clase de utilidades para generar y registrar gráficos directamente en MLflow.
    Incluye: matriz de confusión normalizada y gráficos de métricas.
    """

    def __init__(self):
        pass  # No requiere directorio de salida

    def log_confusion_matrix(self, y_true, y_pred, title="Matriz de Confusión Normalizada", artifact_path="plots/confusion_matrix"):
        """
        Genera y registra la matriz de confusión normalizada directamente en MLflow.

        Args:
            y_true (array-like): Etiquetas reales.
            y_pred (array-like): Etiquetas predichas.
            title (str): Título del gráfico.
            artifact_path (str): Carpeta dentro de MLflow donde se guardará la figura.
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("True")
        ax.set_xlabel("Pred")

        mlflow.log_figure(fig, f"{artifact_path}.png")
        plt.close(fig)

    def log_metrics_bar(self, metrics_dict, title="Comparación de Métricas", artifact_path="plots/metrics_comparison"):
        """
        Crea y registra un gráfico de barras para comparar métricas entre modelos directamente en MLflow.

        Args:
            metrics_dict (dict): Ejemplo -> {"RandomForest": 0.85, "XGBoost": 0.88}
            title (str): Título del gráfico.
            artifact_path (str): Carpeta dentro de MLflow donde se guardará la figura.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()), palette="viridis", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("F1-score (macro)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        mlflow.log_figure(fig, f"{artifact_path}.png")
        plt.close(fig)
