from scipy.io import arff
import pandas as pd

def load_arff(file_path):
    """Carga un archivo ARFF y devuelve un DataFrame de pandas."""
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    return df
