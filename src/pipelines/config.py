# ============================================================
# config.py — Configuración general del proyecto y MLflow
# ============================================================

import sys
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# Ajustar ruta raíz del proyecto
root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

# Directorios base
model_dir = root_path / "models"
model_dir.mkdir(exist_ok=True)

mlruns_path = (root_path / "mlruns").resolve()
mlruns_path.mkdir(exist_ok=True)

# MLflow configuración
experiment_name = f"CEE-Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
tracking_uri = mlruns_path.as_uri()

# Rutas de datos
data_path = root_path / "data" / "raw" / "CEE_DATA.arff"
results_dir = root_path / "results"
results_dir.mkdir(exist_ok=True)
