# src/models/mlflow.py

import subprocess
import socket
import time
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

################################################################
# Clase de Servidor MLFlow #
################################################################
class MLflowServer:
    """Clase para manejar el servidor local de MLflow UI."""

    def __init__(self, experiment_name,
                 host="0.0.0.0", 
                 port=5000, 
                 tracking_uri=None, 
                 backend_store_path=None 
                 ):
        
        self.host = host
        self.port = port
        self.tracking_uri = tracking_uri
        self.process = None
        
        # --- Configuración del Backend Store para el comando 'mlflow ui' ---
        # 1. Almacenamos el URI de archivo, que es lo que 'mlflow ui' necesita.
        self.backend_store_uri_for_ui = None
        if backend_store_path:
            # Convertimos la ruta de disco (string) a un objeto Path, 
            # y luego a URI (añadiendo 'file:///' y usando forward slashes).
            path_obj = Path(backend_store_path)
            self.backend_store_uri_for_ui = path_obj.as_uri()
        
        # --- Configuración de la Librería MLflow ---
        # 1. Establecer el tracking URI (dónde ESCRIBIR los logs)
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        # 2. Establecer el nombre del experimento
        mlflow.set_experiment(experiment_name)


    def _is_port_in_use(self):
        """Verifica si el puerto ya está en uso."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # connect_ex retorna 0 si la conexión es exitosa (puerto ocupado)
            return s.connect_ex((self.host, self.port)) == 0

    def start_ui(self, background=True):
        """
        Inicia MLflow UI en host:port. 
        Utiliza el URI de archivo correcto para `--backend-store-uri`.
        """
        if self._is_port_in_use():
            print(f"⚠️ MLflow UI ya corriendo en http://{self.host}:{self.port}")
            return
        
        # Construcción del comando base
        cmd = [
            "mlflow", "ui", 
            "--host", self.host, 
            "--port", str(self.port)
        ]
        
        # Agregar el URI correctamente formateado (file:///...) al comando CLI
        if self.backend_store_uri_for_ui:
             cmd.extend(["--backend-store-uri", self.backend_store_uri_for_ui]) 

        # Ejecución del comando
        if background:
            # Ejecutar en segundo plano, ocultando la salida de la consola del servidor
            self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            # Ejecutar y esperar a que el usuario detenga el servidor
            subprocess.run(cmd)
            
        time.sleep(2) # Espera para asegurar que el servidor arranque

        # Comprobación final
        if self._is_port_in_use():
            print(f"✅ MLflow UI iniciado en http://{self.host}:{self.port}")
            if self.backend_store_uri_for_ui:
                print(f"   Leyendo datos de: {self.backend_store_uri_for_ui}")
        else:
             print("❌ Error al iniciar MLflow UI. Revisa la consola o logs.")


    def stop_ui(self):
        """Detiene el servidor si estaba corriendo en background."""
        if self.process:
            self.process.terminate()
            # Intenta esperar un momento antes de forzar la terminación
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("🛑 MLflow UI detenido")
            
################################################################
# Clase de Tracking MLFlow #
################################################################

class MLflowTracker:
    """
    Clase para gestionar el registro de runs, parámetros, métricas y artefactos en MLflow.
    Permite centralizar todo el logging de experimentos en una interfaz limpia y reutilizable.
    """

    def __init__(self, experiment_name: str, tracking_uri: str = None):
        """
        Inicializa la conexión con MLflow y configura el experimento activo.

        Args:
            experiment_name (str): Nombre del experimento en MLflow.
            tracking_uri (str, optional): URI del tracking server o carpeta local (p. ej. "file:///ruta/a/mlruns").
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    # ==============================================================
    # MÉTODOS DE CONTROL DE RUNS
    # ==============================================================

    def start_run(self, run_name: str):
        """
        Inicia un nuevo run en MLflow.

        Args:
            run_name (str): Nombre identificador del run (por ejemplo, el nombre del modelo).

        Returns:
            Context manager de MLflow (usable con 'with').
        """
        return mlflow.start_run(run_name=run_name)

    # ==============================================================
    # MÉTODOS DE LOGGING DE PARÁMETROS Y CONFIGURACIÓN
    # ==============================================================

    def log_pipeline_params(
        self,
        model_params: dict,
        preprocessing: str,
        imbalance_technique: str,
        cv: int,
        extra_params: dict = None
    ):
        """
        Registra la configuración general del pipeline.

        Args:
            model_params (dict): Hiperparámetros del modelo.
            preprocessing (str): Descripción del preprocesamiento aplicado.
            imbalance_technique (str): Técnica de balanceo usada (e.g., 'SMOTE', 'None').
            cv (int): Número de folds de validación cruzada.
            extra_params (dict, optional): Parámetros adicionales a registrar (e.g., tamaño de dataset, versión de datos, etc.).
        """
        if model_params:
            mlflow.log_params(model_params)

        mlflow.log_param("data_preprocessing", preprocessing)
        mlflow.log_param("imbalance_technique", imbalance_technique)
        mlflow.log_param("cross_validation_folds", cv)

        if extra_params:
            mlflow.log_params(extra_params)

    # ==============================================================
    # MÉTODOS DE LOGGING DE RESULTADOS
    # ==============================================================

    def log_model_results(
        self,
        model,
        metrics: dict,
        best_params: dict = None,
        artifact_path: str = "model",
        save_path: str = None
    ):
        """
        Registra resultados, métricas y artefactos del modelo.

        Args:
            model: Modelo entrenado (por ejemplo, un estimador sklearn).
            metrics (dict): Métricas a registrar (e.g., {"F1_CV": 0.87, "accuracy": 0.91}).
            best_params (dict, optional): Hiperparámetros óptimos encontrados durante la búsqueda.
            artifact_path (str, optional): Carpeta donde guardar el modelo dentro del run.
            save_path (str, optional): Ruta local de un archivo adicional (por ejemplo, el modelo serializado o un resumen CSV).
        """
    # Evita colisiones con parámetros anteriores
        if best_params:
            best_params_prefixed = {f"best_{k}": v for k, v in best_params.items()}
            mlflow.log_params(best_params_prefixed)

        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

        # Loguear el modelo completo en MLflow
        mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)

        # Loguear artefactos adicionales (si existen)
        if save_path:
            mlflow.log_artifact(save_path)

    # ==============================================================
    # MÉTODOS AUXILIARES
    # ==============================================================

    @staticmethod
    def log_artifacts_from_dir(dir_path: str):
        """
        Registra todos los archivos de un directorio como artefactos en MLflow.

        Args:
            dir_path (str): Ruta del directorio con artefactos (por ejemplo, gráficos, reportes, CSVs).
        """
        mlflow.log_artifacts(dir_path)

    def log_confusion_matrix(self, y_test, y_pred, artifact_path="confusion_matrix.png"):
        """
        Genera, guarda y registra en MLflow la matriz de confusión.

        Args:
            y_true: Etiquetas verdaderas.
            y_pred: Etiquetas predichas.
            artifact_path (str): Nombre o ruta local del archivo imagen donde se guarda la matriz.
        """
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
        plt.title('Matriz de Confusión Normalizada')
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.savefig(artifact_path)
        plt.close()
        mlflow.log_artifact(artifact_path)
