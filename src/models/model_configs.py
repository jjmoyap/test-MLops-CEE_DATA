# ============================================================
# model_configs.py — Definición de modelos y sus hiperparámetros
# ============================================================

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

models = {
    "RandomForest": {
        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "params": {
            "classifier__n_estimators": [300],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__criterion": ["gini", "entropy"],
        },
    },

    "XGBoost": {
        "model": XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        ),
        "params": {
            "classifier__n_estimators": [300],
            "classifier__max_depth": [5, 7],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__subsample": [0.8],
            "classifier__colsample_bytree": [0.8],
            "classifier__gamma": [0, 0.5],
            "classifier__reg_alpha": [0, 0.1],
            "classifier__reg_lambda": [1.0],
            "classifier__min_child_weight": [1, 3],
        },
    },

    "CatBoost": {
        "model": CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            border_count=120,
            loss_function="MultiClass",
            verbose=0,
            random_seed=42
        ),
        "params": {
            "classifier__depth": [4, 6],
            "classifier__learning_rate": [0.03, 0.05],
            "classifier__l2_leaf_reg": [3, 5],
            "classifier__border_count": [64, 120],
        }   
    },

    "ExtraTrees": {
        "model": ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "params": {
            "classifier__n_estimators": [300],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__criterion": ["gini", "entropy"],
        },
    }
}
