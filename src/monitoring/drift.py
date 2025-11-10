# ============================================================
# src/monitoring/drift.py — Simulación de Data Drift y Detección de Pérdida de Performance
# ============================================================

# --- IMPORTS BASE ---
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import joblib
import pickle
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Asegurar imports de src/* (coloca esto ARRIBA, antes de importar src.*)
# -----------------------------------------------------------------
# .../src/monitoring -> src -> <root>
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Imports locales
try:
    from src.data.load_data import ARFFLoader
    from src.models.mlflow import MLflowTracker
except Exception:
    MLflowTracker = None  # si MLflow no está disponible, seguimos sin logging


def find_best_model(models_dir: Path) -> Path:
    candidates = sorted(list(models_dir.glob("*.pkl")))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontraron modelos .pkl en {models_dir}")
    for c in candidates:
        if c.name.startswith("best_model_global_"):
            return c
    m = models_dir / "model.pkl"
    if m.exists():
        return m
    return max(candidates, key=lambda p: p.stat().st_mtime)


def jensen_shannon(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def population_stability_index(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    ref = ref_counts / (ref_counts.sum() + 1e-12)
    cur = cur_counts / (cur_counts.sum() + 1e-12)
    eps = 1e-6
    return float(np.sum((cur - ref) * np.log((cur + eps) / (ref + eps))))


def categorical_drift_metrics(ref: pd.Series, cur: pd.Series):
    cats = sorted(set(ref.astype(str).unique()).union(
        set(cur.astype(str).unique())))
    ref_counts = np.array([(ref.astype(str) == c).sum() for c in cats])
    cur_counts = np.array([(cur.astype(str) == c).sum() for c in cats])
    js = jensen_shannon(ref_counts, cur_counts)
    psi = population_stability_index(ref_counts, cur_counts)
    return {"psi": psi, "js": js, "categories": cats,
            "ref_counts": ref_counts.tolist(), "cur_counts": cur_counts.tolist()}


def simulate_categorical_shift(X: pd.DataFrame, shift_cols=None, strength: float = 0.35, random_state: int = 42) -> pd.DataFrame:
    """Simula cambio de frecuencias: aumenta la prob. de la categoría más frecuente en cada columna categórica."""
    rng = np.random.default_rng(random_state)
    X2 = X.copy()
    if shift_cols is None:
        shift_cols = [c for c in X2.columns if X2[c].dtype ==
                      object or str(X2[c].dtype).startswith("category")]
    for col in shift_cols:
        vc = X2[col].astype(str).value_counts(normalize=True)
        if vc.empty:
            continue
        top_cat = vc.idxmax()
        cats = vc.index.tolist()
        probs = vc.values.copy()
        for i, c in enumerate(cats):
            if c == top_cat:
                probs[i] = min(1.0, probs[i] * (1.0 + strength))
        probs = probs / probs.sum()
        X2[col] = rng.choice(cats, size=len(X2), p=probs)
    return X2


def introduce_missingness(X: pd.DataFrame, cols=None, frac: float = 0.10, random_state: int = 42) -> pd.DataFrame:
    """Introduce NaNs en una fracción de filas para columnas categóricas (simula fallas de captura/ingesta)."""
    rng = np.random.default_rng(random_state)
    X2 = X.copy()
    if cols is None:
        cols = [c for c in X2.columns if X2[c].dtype ==
                object or str(X2[c].dtype).startswith("category")]
    n = len(X2)
    for col in cols:
        if n == 0:
            continue
        k = int(frac * n)
        if k <= 0:
            continue
        idx = rng.choice(n, size=k, replace=False)
        X2.loc[X2.index[idx], col] = np.nan
    return X2


def _as_numeric_set_safe(vals):
    try:
        arr = pd.to_numeric(pd.Series(list(vals)), errors="coerce")
        if arr.isna().any():
            return None
        return set(arr.astype(int).tolist())
    except Exception:
        return None


def _build_ordinal_indexer(y: pd.Series, fe=None):
    """
    Devuelve (indexer: dict[str->int], levels: list[str]) con el orden ordinal.
    Usa fe.ordinal_map si existe; si no, intenta un orden razonable.
    """
    # Orden de entrenamiento si existe
    if fe is not None and hasattr(fe, "ordinal_map") and fe.ordinal_map:
        levels = list(fe.ordinal_map)
    else:
        # fallback razonable para este dataset
        # (ajusta aquí si tu orden real es distinto)
        levels = ['Average', 'Good', 'Vg', 'Excellent']

    # normalizamos a string para mapear
    levels_norm = [str(x) for x in levels]
    indexer = {lvl: i for i, lvl in enumerate(levels_norm)}
    return indexer, levels_norm


def normalize_target(y: pd.Series, model, fe=None) -> pd.Series:
    """
    Normaliza y al espacio de clases del modelo.
    - Si el modelo es numérico (e.g., [0,1,2]) y y es ordinal string:
      mapea strings -> índices ordinales -> colapsa si hay más niveles que clases.
    - Si ambos son numéricos, verifica/conforma tipos.
    - Si el modelo es string, normaliza strings.
    """
    # Caso: modelo con clases numéricas
    if hasattr(model, "classes_"):
        model_num_set = _as_numeric_set_safe(model.classes_)
    else:
        model_num_set = None

    if model_num_set is not None:
        # ¿y ya es numérico?
        y_num = pd.to_numeric(y, errors="coerce")
        if not y_num.isna().any():
            # Asegura que cabe en las clases del modelo
            if set(y_num.astype(int).unique()).issubset(model_num_set):
                return y_num.astype(type(list(model_num_set)[0]))

        # y es ordinal string -> convertir a índice ordinal
        indexer, levels = _build_ordinal_indexer(y, fe=fe)
        y_str = y.astype(str)
        if not set(y_str.unique()).issubset(set(levels)):
            # Intento tolerante: cualquier etiqueta no vista va al rango más cercano (nivel 0)
            y_ord = y_str.map(lambda s: indexer.get(s, 0)).astype(int)
        else:
            y_ord = y_str.map(indexer).astype(int)

        # colapsar si y tiene más niveles que el #clases del modelo
        k_model = len(model_num_set)  # e.g., 3
        if k_model <= 0:
            raise ValueError("model.classes_ vacío o inválido.")
        # Regla: recortar al último bin (fusiona niveles superiores)
        y_collapsed = np.minimum(y_ord, k_model - 1).astype(int)

        # Asegurar tipo (int) y que pertenezca al conjunto de clases
        if set(y_collapsed.unique()).issubset(model_num_set):
            return pd.Series(y_collapsed, index=y.index, dtype=type(list(model_num_set)[0]))

        # Si por alguna razón no coincide, hacemos cast al tipo del primer elemento
        return pd.Series(y_collapsed, index=y.index, dtype=int)

    # Caso: modelo con clases de texto
    if hasattr(model, "classes_"):
        model_str_set = set(
            map(lambda s: str(s).strip().lower(), model.classes_))
        y_str = y.astype(str).str.strip().str.lower()
        if set(y_str.unique()).issubset(model_str_set):
            return y_str
        raise ValueError(
            f"No pude alinear etiquetas con model.classes_.\n"
            f"y_test uniques={sorted(map(str, y.unique()))}\n"
            f"model.classes_={list(model.classes_)}"
        )

    # Sin referencia del modelo: último recurso, intenta numérico
    y_num = pd.to_numeric(y, errors="coerce")
    if not y_num.isna().any():
        return y_num.astype(int)
    return y  # fallback

def evaluate(model, X, y):
    y_pred = model.predict(X)
    macro_f1 = f1_score(y, y_pred, average="macro")
    acc = accuracy_score(y, y_pred)
    report = classification_report(
        y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
    return {"macro_f1": float(macro_f1), "accuracy": float(acc), "report": report, "cm": cm.tolist()}, y_pred


def compute_feature_drifts(X_ref: pd.DataFrame, X_cur: pd.DataFrame):
    """Calcula PSI/JS por feature comparando referencia vs monitoreo en ESPACIO CRUDO."""
    rows = []
    for col in X_ref.columns:
        try:
            if X_ref[col].dtype == object or str(X_ref[col].dtype).startswith("category"):
                m = categorical_drift_metrics(X_ref[col], X_cur[col])
                rows.append({"feature": col, "type": "categorical",
                            "psi": m["psi"], "js": m["js"]})
            else:
                ref_vals = pd.to_numeric(
                    X_ref[col], errors="coerce").astype(float)
                cur_vals = pd.to_numeric(
                    X_cur[col], errors="coerce").astype(float)
                ref_counts, _ = np.histogram(
                    ref_vals[~np.isnan(ref_vals)], bins=10)
                cur_counts, _ = np.histogram(
                    cur_vals[~np.isnan(cur_vals)], bins=10)
                psi = population_stability_index(ref_counts, cur_counts)
                js = jensen_shannon(ref_counts, cur_counts)
                rows.append(
                    {"feature": col, "type": "numeric", "psi": psi, "js": js})
        except Exception:
            rows.append({"feature": col, "type": "unknown",
                        "psi": np.nan, "js": np.nan})
    return pd.DataFrame(rows)


def plot_psi_bar(df_drift: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    df_sorted = df_drift.sort_values("psi", ascending=False)
    ax.bar(df_sorted["feature"], df_sorted["psi"])
    ax.set_title("PSI por Feature (referencia vs monitoreo)")
    ax.set_ylabel("PSI")
    ax.set_xlabel("Feature")
    ax.set_xticklabels(df_sorted["feature"], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================
# Feature Engineering a partir de MAPAS guardados en el .pkl
# ============================================================

def load_feature_engineering(models_dir: Path):
    """Carga models/feature_engineering.pkl (artefacto de entrenamiento)."""
    fe_path = models_dir / "feature_engineering.pkl"
    if not fe_path.exists():
        return None
    try:
        return joblib.load(fe_path)
    except Exception:
        try:
            with open(fe_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


def _ordinal_to_num(series: pd.Series, ordinal_map):
    """Mapea categorías ordinales a números usando el orden de ordinal_map."""
    if ordinal_map is None or not len(ordinal_map):
        # fallback razonable si no hay mapa
        ordinal_map = ['Average', 'Good', 'Vg', 'Excellent']
    rank = {cat: i for i, cat in enumerate(ordinal_map)}
    return series.astype(str).map(rank).astype(float)


def fe_transform_with_maps(fe_obj, X: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruye las columnas que el modelo espera usando:
      - fe_obj.ordinal_map      -> *_num y Academic_Score
      - fe_obj.freq_maps        -> *_freq
      - fe_obj.target_mean_maps -> *_target_mean
    """
    X2 = X.copy()

    # 1) columnas ordinales conocidas del dataset
    col_x = 'Class_ X_Percentage'
    col_xii = 'Class_XII_Percentage'
    if col_x in X2.columns:
        X2[col_x + '_num'] = _ordinal_to_num(X2[col_x],
                                             getattr(fe_obj, 'ordinal_map', None))
    if col_xii in X2.columns:
        X2[col_xii + '_num'] = _ordinal_to_num(
            X2[col_xii], getattr(fe_obj, 'ordinal_map', None))

    # Academic_Score = promedio de ambos si existen
    if col_x + '_num' in X2.columns and col_xii + '_num' in X2.columns:
        X2['Academic_Score'] = (X2[col_x + '_num'] +
                                X2[col_xii + '_num']) / 2.0

    # 2) columnas categóricas para freq y target mean (según mapas guardados)
    freq_maps = getattr(fe_obj, 'freq_maps', {}) or {}
    tmean_maps = getattr(fe_obj, 'target_mean_maps', {}) or {}

    def _map_with_others(series: pd.Series, mapping: dict):
        if not mapping:
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        # Si entrenaste con 'OTHERS', usaremos ese fallback; si no, 0
        has_others = 'OTHERS' in mapping

        def _mapper(v):
            v_str = str(v)
            if v_str in mapping:
                return mapping[v_str]
            if has_others:
                return mapping['OTHERS']
            return 0.0
        return series.astype(str).map(_mapper).astype(float)

    # aplicar *_freq
    for col, mp in freq_maps.items():
        if col in X2.columns:
            X2[f'{col}_freq'] = _map_with_others(X2[col], mp)

    # aplicar *_target_mean
    for col, mp in tmean_maps.items():
        if col in X2.columns:
            X2[f'{col}_target_mean'] = _map_with_others(X2[col], mp)

    return X2


# ============================================================
# Alineación, gráficos y debug
# ============================================================

def align_columns_if_needed(X_df: pd.DataFrame, model) -> pd.DataFrame:
    """Alinea columnas al orden/esperado por el modelo si éste expone referencias."""
    target_cols = None
    if hasattr(model, "feature_names_in_"):
        target_cols = list(model.feature_names_in_)
    elif hasattr(model, "get_feature_names_out"):
        try:
            target_cols = list(model.get_feature_names_out())
        except Exception:
            target_cols = None

    if target_cols is None:
        return X_df

    X_aligned = X_df.copy()
    for c in target_cols:
        if c not in X_aligned.columns:
            X_aligned[c] = 0
    # Filtrar extras y ordenar
    X_aligned = X_aligned[target_cols]
    return X_aligned


def save_confusion_matrix(cm, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _debug_columns(name, Xdf, model=None, max_cols=12):
    try:
        print(f"[DEBUG] {name} shape={Xdf.shape}")
        cols = list(Xdf.columns[:max_cols])
        print(f"[DEBUG] {name} columns sample: {cols}")
        if model is not None and hasattr(model, "feature_names_in_"):
            print(
                f"[DEBUG] model.feature_names_in_ sample: {list(model.feature_names_in_[:max_cols])}")
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main(args):
    raw_path = Path(args.raw)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results)
    reports_dir = Path(args.reports)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar datos crudos
    df = ARFFLoader(str(raw_path)).load()
    target_col = "Performance"
    y = df[target_col]
    X = df.drop(columns=[target_col])
    print(f"[OK] Datos crudos cargados: {X.shape[0]} filas, {X.shape[1]} columnas.")

    # 2) Split para baseline (usa estratificación)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"[OK] Datos divididos en train ({X_train.shape[0]} filas) y test ({X_test.shape[0]} filas).")

    # 3) Cargar modelo entrenado
    model_path = find_best_model(models_dir)
    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    print(f"[OK] Modelo cargado desde {model_path}.")

    # 4) Cargar artefacto FE y RECONSTRUIR features con sus mapas
    fe = load_feature_engineering(models_dir)
    if fe is None:
        raise FileNotFoundError(
            "No se encontró models/feature_engineering.pkl. "
            "Tu modelo requiere columnas ya ingenierizadas (e.g., *_freq, *_target_mean, *_num, Academic_Score). "
            "Guarda el artefacto de feature engineering del entrenamiento en models/feature_engineering.pkl."
        )
    print(f"[OK] Artefacto de Feature Engineering cargado.")

    # 5) Transformar baseline y drift con los MAPAS (sin .transform)
    X_test_fe = fe_transform_with_maps(fe, X_test)
    X_test_eval = align_columns_if_needed(X_test_fe, model)
    _debug_columns("X_test_eval", X_test_eval, model=model)
    print(f"[OK] Features transformadas para evaluación baseline.")

    # 6) Evaluación baseline
    y_test_norm = normalize_target(y_test, model)
    base_metrics, y_pred_base = evaluate(model, X_test_eval, y_test_norm)   
    print(f"[OK] Evaluación baseline completada: Macro F1={base_metrics['macro_f1']:.4f}, Accuracy={base_metrics['accuracy']:.4f}")

    # Guardar matriz de confusión baseline
    labels_sorted = sorted(y_test.unique())
    cm_baseline = np.array(base_metrics["cm"])
    cm_base_path = reports_dir / "confusion_matrix_baseline.png"
    save_confusion_matrix(cm_baseline, labels_sorted,
                          cm_base_path, "Matriz de Confusión — Baseline")
    print(f"[OK] Matriz de confusión baseline guardada en {cm_base_path}.")

    # 7) Simulación de drift (en CRUDO) y misma FE basada en mapas
    drifted_raw = simulate_categorical_shift(
        X_test, shift_cols=None, strength=0.50, random_state=7)
    drifted_raw = introduce_missingness(
        drifted_raw, cols=None, frac=0.10, random_state=7)

    drifted_fe = fe_transform_with_maps(fe, drifted_raw)
    X_drift_eval = align_columns_if_needed(drifted_fe, model)
    _debug_columns("X_drift_eval", X_drift_eval, model=model)
    print(f"[OK] Datos con drift simulados y features transformadas para evaluación.")

    # 8) Evaluación con drift
    drift_metrics, y_pred_drift = evaluate(model, X_drift_eval, y_test_norm)
    print(f"[OK] Evaluación con drift completada: Macro F1={drift_metrics['macro_f1']:.4f}, Accuracy={drift_metrics['accuracy']:.4f}")

    # 9) Métricas de drift de distribución en ESPACIO CRUDO (negocio)
    df_drift = compute_feature_drifts(X_test, drifted_raw)
    df_drift_path = results_dir / "feature_drift_metrics.csv"
    df_drift.to_csv(df_drift_path, index=False)
    print(f"[OK] Métricas de drift por feature guardadas en {df_drift_path}.")

    # 10) Gráfico de PSI por feature
    psi_plot_path = reports_dir / "psi_by_feature.png"
    plot_psi_bar(df_drift, psi_plot_path)
    print(f"[OK] Gráfico de PSI por feature guardado en {psi_plot_path}.")

    # 11) Comparativa de desempeño (baseline vs drift)
    comp = pd.DataFrame([
        {"metric": "macro_f1",
            "baseline": base_metrics["macro_f1"], "drifted": drift_metrics["macro_f1"]},
        {"metric": "accuracy",
            "baseline": base_metrics["accuracy"], "drifted": drift_metrics["accuracy"]},
    ])
    comp["abs_drop"] = comp["baseline"] - comp["drifted"]
    comp["rel_drop_%"] = np.where(
        comp["baseline"] > 0, 100.0 * comp["abs_drop"] / comp["baseline"], np.nan)
    comp_path = results_dir / "performance_comparison.csv"
    comp.to_csv(comp_path, index=False)
    print(f"[OK] Comparativa de desempeño guardada en {comp_path}.")

    # (Opcional) matriz de confusión con drift
    cm_drift = np.array(drift_metrics["cm"])
    cm_drift_path = reports_dir / "confusion_matrix_drift.png"
    save_confusion_matrix(cm_drift, labels_sorted,
                          cm_drift_path, "Matriz de Confusión — Drift")
    print(f"[OK] Matriz de confusión con drift guardada en {cm_drift_path}.")

    # 12) Lógica de alerta + resumen JSON
    max_psi = float(df_drift["psi"].max() if len(df_drift) else 0.0)
    macro_f1_drop_rel = float(comp.loc[comp["metric"] == "macro_f1", "rel_drop_%"].iloc[0])

    alert = ((macro_f1_drop_rel >= args.f1_drop_rel_threshold) or (max_psi >= args.psi_threshold))
    action = "Revisar pipeline de features y considerar retrain" if alert else "Continuar monitoreo"

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "baseline": {"macro_f1": base_metrics["macro_f1"], "accuracy": base_metrics["accuracy"]},
        "drifted": {"macro_f1": drift_metrics["macro_f1"], "accuracy": drift_metrics["accuracy"]},
        "max_feature_psi": max_psi,
        "macro_f1_drop_rel_%": macro_f1_drop_rel,
        "psi_threshold": args.psi_threshold,
        "f1_drop_rel_threshold": args.f1_drop_rel_threshold,
        "alert": bool(alert),
        "proposed_action": action,
        "notes": "Drift simulado con reponderación de categorías (+50%) e introducción de 10% de missing values."
    }
    summary_path = results_dir / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 13) Logging a MLflow (si está disponible)
    if MLflowTracker is not None:
        try:
            tracker = MLflowTracker(
                experiment_name=args.experiment, tracking_uri=args.tracking_uri)
            with tracker.start_run(run_name=f"drift_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow = tracker.mlflow
                mlflow.set_tag("drift_simulation", True)
                mlflow.set_tag("data_source", str(raw_path))
                mlflow.log_params({
                    "psi_threshold": args.psi_threshold,
                    "f1_drop_rel_threshold": args.f1_drop_rel_threshold
                })
                mlflow.log_metric("baseline_macro_f1",
                                  base_metrics["macro_f1"])
                mlflow.log_metric("baseline_accuracy",
                                  base_metrics["accuracy"])
                mlflow.log_metric("drift_macro_f1", drift_metrics["macro_f1"])
                mlflow.log_metric("drift_accuracy", drift_metrics["accuracy"])
                mlflow.log_metric("macro_f1_drop_rel_%", macro_f1_drop_rel)
                mlflow.log_metric("max_feature_psi", max_psi)
                for _, row in df_drift.iterrows():
                    if pd.notna(row["psi"]):
                        mlflow.log_metric(
                            f"psi__{row['feature']}", float(row["psi"]))
                    if pd.notna(row["js"]):
                        mlflow.log_metric(
                            f"js__{row['feature']}", float(row["js"]))
                mlflow.log_artifact(str(df_drift_path), artifact_path="drift")
                mlflow.log_artifact(str(comp_path), artifact_path="drift")
                mlflow.log_artifact(str(summary_path), artifact_path="drift")
                mlflow.log_artifact(str(psi_plot_path), artifact_path="plots")
                mlflow.log_artifact(str(cm_base_path), artifact_path="plots")
                mlflow.log_artifact(str(cm_drift_path), artifact_path="plots")
        except Exception as e:
            print(f"[WARN] MLflow no disponible o falló el logging: {e}")
    else:
        print("[INFO] MLflowTracker no disponible; se omite logging en MLflow.")

    print(f"[OK] Drift simulation finished.")
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] PSI plot: {psi_plot_path}")
    print(f"[OK] Comparison CSV: {comp_path}")
    print(f"[OK] Feature drift CSV: {df_drift_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación de Data Drift y Detección de Pérdida de Performance")
    parser.add_argument("--raw", type=str, default="data/raw/CEE_DATA.arff")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--results", type=str, default="results_drift")
    parser.add_argument("--reports", type=str, default="reports_drift")
    parser.add_argument("--experiment", type=str,
                        default="Data Drift Monitoring")
    parser.add_argument("--tracking_uri", type=str, default="mlruns")
    parser.add_argument("--psi_threshold", type=float, default=0.25)
    parser.add_argument("--f1_drop_rel_threshold", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
