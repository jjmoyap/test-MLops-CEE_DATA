# ============================================================
# ml_pipeline.py — Ejecución completa del pipeline ML
# ============================================================

import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0️ Configuración inicial y librerías
# ============================================================

# Ajustar ruta raíz del proyecto
root_path = Path(__file__).resolve().parents[2]  # sube dos niveles: src/pipelines → raíz
sys.path.append(str(root_path))

# Directorio donde se guardarán los modelos entrenados
model_dir = Path(root_path) / "models"
model_dir.mkdir(exist_ok=True)

# Librerías principales
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Modelos ML
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# SMOTE y Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Componentes propios del proyecto
from src.data.load_data import ARFFLoader
from src.features.feature_engineering import FeatureEngineering
from src.models.train_models import PipelineML
from src.visualization.plot_utils import Visualizer
from src.models.mlflow import MLflowServer, MLflowTracker

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# ============================================================
# 1️ Configuración de MLflow y servidor local
# ============================================================

MLFLOW_ARTIFACT_ROOT_PATH = (root_path / "mlruns").resolve()
MLFLOW_ARTIFACT_ROOT_PATH.mkdir(exist_ok=True)
TRACKING_URI = MLFLOW_ARTIFACT_ROOT_PATH.as_uri()

# Experimento único por ejecución
experiment_name = f"CEE-Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Instancia del servidor MLflow local
mlflow_server_handler = MLflowServer(
    experiment_name=experiment_name,
    host="127.0.0.1",
    port=5000,
    tracking_uri=TRACKING_URI,
    backend_store_path=MLFLOW_ARTIFACT_ROOT_PATH
)
mlflow_server_handler.start_ui(background=True)
mlflow.set_experiment(experiment_name)

# Instancia del visualizador
viz = Visualizer()

# ============================================================
# 2️ Cargar y preparar datos
# ============================================================

base_path = root_path
data_path = base_path / "data/raw/CEE_DATA.arff"

loader = ARFFLoader(data_path)
df = loader.load()

# Columnas categóricas y ordinales
categorical_cols = [
    'Gender', 'Caste', 'coaching', 'time', 'Class_ten_education',
    'twelve_education', 'medium', 'Father_occupation', 'Mother_occupation'
]
ordinal_cols = ["Class_ X_Percentage", "Class_XII_Percentage"]
ord_map = ["Poor", "Average", "Good", "Vg", "Excellent"]

# ============================================================
# 3️ Feature Engineering
# ============================================================

fe = FeatureEngineering(ordinal_map=ord_map)

# Combinar categorías raras
for col in categorical_cols:
    df = fe.combine_rare(df, col, threshold=0.2)

# Ordinal encoding y Academic_Score
df = fe.create_ordinal_features(df, ordinal_cols)

# Agrupar target
df['Performance_grouped'] = df['Performance'].replace({
    'Average': 'Average/Good', 'Good': 'Average/Good',
    'Vg': 'Vg', 'Excellent': 'Excellent'
})
df['Performance_num'] = LabelEncoder().fit_transform(df['Performance_grouped'])

# Features de frecuencia y mean encoding
df = fe.add_frequency_features(df, categorical_cols, target_col='Performance_num')

# ============================================================
# 4️ Preparar datos para entrenamiento
# ============================================================

X = df.drop(columns=['Performance', 'Performance_grouped', 'Performance_num'])
y = df['Performance_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 5️ Definir modelos y parámetros
# ============================================================

models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [350, 400],
            'classifier__max_depth': [15, 17],
            'classifier__min_samples_leaf': [4, 5, 6],
            'classifier__min_samples_split': [14, 15, 17],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__bootstrap': [True, False]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                               random_state=42, tree_method='hist', n_jobs=-1),
        'params': {
            'classifier__n_estimators': [100, 150],
            'classifier__max_depth': [3, 4],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__subsample': [0.7, 0.9],
            'classifier__colsample_bytree': [0.7, 0.9],
            'classifier__gamma': [0, 1],
            'classifier__reg_alpha': [0, 0.1],
            'classifier__reg_lambda': [1, 1.5],
            'classifier__min_child_weight': [1, 3, 5]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(iterations=300, verbose=0, random_seed=42),
        'params': {
            'classifier__depth': [4, 6],
            'classifier__learning_rate': [0.05, 0.07],
            'classifier__l2_leaf_reg': [1, 3, 5],
            'classifier__border_count': [64]
        }
    },
    'ExtraTrees': {
        "model": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__max_features": ["sqrt", "log2"]
        }
    }
}

# ============================================================
# 6️ Entrenamiento y Registro MLflow
# ============================================================

ml_tracker = MLflowTracker(experiment_name=experiment_name, tracking_uri=TRACKING_URI)
pipeline_ml = PipelineML(model_dir=model_dir, mlflow_experiment=experiment_name, cv=5)

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Definir preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Bucle principal de entrenamiento
for model_name, model_dict in models.items():
    print(f"Entrenando {model_name}...")

    # Crear pipeline completo
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model_dict['model'])
    ])

    # Registrar experimento MLflow
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

# ============================================================
# 7️ Guardar resumen de resultados
# ============================================================

results_dir = base_path / "results"
results_dir.mkdir(exist_ok=True)
summary_path = results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

summary = pd.DataFrame(pipeline_ml.results_summary)
summary.to_csv(summary_path, index=False)

print(f"\nResumen guardado en {summary_path}")
print(f"Experimento '{experiment_name}' registrado en MLflow ({TRACKING_URI})")
