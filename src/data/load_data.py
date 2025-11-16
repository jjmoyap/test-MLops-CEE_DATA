# src/data/load_data.py
from scipy.io import arff
import pandas as pd
import numpy as np

class ARFFLoader:
    """
    Clase para cargar archivos ARFF y devolver un DataFrame limpio de pandas.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """
        Carga el archivo ARFF, limpia y decodifica los datos.
        """
        data = arff.loadarff(self.file_path)
        df = pd.DataFrame(data[0])
        df = self._decode_bytes_columns(df)
        df = self._clean_data(df)
        df = self._ensure_types(df)
        return df

    def _decode_bytes_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decodifica las columnas tipo bytes a strings."""
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].apply(lambda x: x.decode("utf-8").strip() if isinstance(x, bytes) else x)
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza básica:
        - Elimina espacios extra
        - Reemplaza valores nulos o '?'
        """
        # Replace '?' and empty strings with NaN
        df = df.replace(["?", "nan", "NaN", ""], np.nan)

        # Strip spaces from string values
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # Optionally, drop rows with all NaN or fill
        df = df.dropna(how="all")

        return df

    def _ensure_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas numéricas que fueron cargadas como strings.
        """
        for col in df.columns:
            # Try to cast to numeric when possible
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return df
