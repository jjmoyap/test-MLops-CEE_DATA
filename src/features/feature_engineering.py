# ============================================================
# feature_engineering.py — Rare Category + Target Mean Encoding
# ============================================================

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Aplica:
      1. Rare category grouping para variables categóricas de alta cardinalidad.
      2. Target mean encoding con suavizado (smoothing).
      3. Elimina las columnas categóricas originales.
    """

    def __init__(self, categorical_cols=None, rare_threshold=0.03, alpha=10.0):
        """
        Parameters
        ----------
        categorical_cols : list[str]
            Columnas categóricas a transformar.
        rare_threshold : float
            Frecuencia mínima relativa para NO ser considerada 'rara'.
        alpha : float
            Parámetro de suavizado para target mean encoding.
        """
        self.categorical_cols = categorical_cols
        self.rare_threshold = rare_threshold
        self.alpha = alpha

        # Se aprenden en fit
        self.rare_maps_ = {}
        self.target_mean_maps_ = {}
        self.global_mean_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("FeatureEngineering.fit requiere un 'y' para target mean encoding.")

        X = pd.DataFrame(X).copy()
        y = pd.Series(y)

        # Inferir columnas categóricas si no se pasan explícitamente
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # 1. Aprender rare category grouping
        self.rare_maps_ = {}
        for col in self.categorical_cols:
            freq = X[col].value_counts(normalize=True)
            rare_values = freq[freq < self.rare_threshold].index
            self.rare_maps_[col] = set(rare_values)

        # 2. Aplicar rare grouping temporalmente para calcular medias
        X_tmp = X.copy()
        for col in self.categorical_cols:
            rare_set = self.rare_maps_[col]
            X_tmp[col] = X_tmp[col].apply(lambda v: "OTHERS" if v in rare_set else v)

        # 3. Media global del target
        self.global_mean_ = y.mean()

        # 4. Calcular target mean encoding (suavizado) por categoría
        self.target_mean_maps_ = {}
        for col in self.categorical_cols:
            df_temp = pd.concat([X_tmp[col], y], axis=1)
            df_temp.columns = [col, "target"]

            counts = df_temp.groupby(col)["target"].count()
            means = df_temp.groupby(col)["target"].mean()

            smooth = (counts * means + self.alpha * self.global_mean_) / (counts + self.alpha)
            self.target_mean_maps_[col] = smooth.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # 1. Rare grouping
        for col in self.categorical_cols:
            rare_set = self.rare_maps_.get(col, set())
            X[col] = X[col].apply(lambda v: "OTHERS" if v in rare_set else v)

        # 2. Target mean encoding → nuevas features
        for col in self.categorical_cols:
            mapping = self.target_mean_maps_.get(col, {})
            X[col + "_te"] = X[col].map(mapping).fillna(self.global_mean_)

        # 3. Eliminar columnas categóricas originales
        X = X.drop(columns=self.categorical_cols, errors="ignore")

        return X
