# ============================================================
# model_configs.py — Definición de modelos y sus hiperparámetros
# ============================================================

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
        'model': XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        ),
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
