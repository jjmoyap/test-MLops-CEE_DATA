from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

class FeatureEngineering:
    """
    Clase para manejar todas las transformaciones de features,
    incluyendo encoding ordinal, features de frecuencia y SMOTE.
    """

    def __init__(self, ordinal_map=None):
        """
        Args:
            ordinal_map (list): Lista ordenada de categorías ordinales.
        """
        self.ordinal_map = ordinal_map
        self.ord_enc = None

    def combine_rare(self, df, col, threshold=0.2):
        """
        Combina categorías raras en 'OTHERS'.
        """
        counts = df[col].value_counts(normalize=True)
        rare = counts[counts < threshold].index
        df[col] = df[col].replace(rare, 'OTHERS')
        return df

    def create_ordinal_features(self, df, ordinal_cols):
        """
        Crea features ordinales y un score promedio académico.
        """
        self.ord_enc = OrdinalEncoder(categories=[self.ordinal_map]*len(ordinal_cols))
        df[[col+'_num' for col in ordinal_cols]] = self.ord_enc.fit_transform(df[ordinal_cols])
        df['Academic_Score'] = df[[col+'_num' for col in ordinal_cols]].mean(axis=1)
        return df

    def add_frequency_features(self, df, categorical_cols, target_col='Performance_num'):
        """
        Crea features de frecuencia y mean encoding.
        """
        for col in categorical_cols:
            df[col+'_freq'] = df[col].map(df[col].value_counts(normalize=True))
            target_mean = df.groupby(col)[target_col].mean()
            df[col+'_target_mean'] = df[col].map(target_mean)
        return df

    def apply_smote(self, X, y):
        """
        Aplica SMOTE al conjunto de entrenamiento.
        """
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
