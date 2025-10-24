# src/data/load_data.py
from scipy.io import arff
import pandas as pd

class ARFFLoader:
    """
    Clase para cargar archivos ARFF y devolver un DataFrame de pandas.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """
        Carga el archivo ARFF y devuelve un DataFrame de pandas.
        Decodifica automáticamente las columnas tipo bytes.
        """
        data = arff.loadarff(self.file_path)
        df = pd.DataFrame(data[0])
        self._decode_bytes_columns(df)
        return df

    def _decode_bytes_columns(self, df: pd.DataFrame):
        """Decodifica las columnas tipo bytes a strings (uso interno)."""
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
