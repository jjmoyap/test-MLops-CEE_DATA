import joblib, pickle
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_model(model, params, X, y, cv=5):
    """
    Entrena un modelo usando GridSearchCV y devuelve el mejor modelo, score y parámetros.
    """
    f1_scorer = make_scorer(f1_score, average='macro')
    gs = GridSearchCV(model, params, cv=cv, scoring=f1_scorer, n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_score_, gs.best_params_

def evaluate_model(model, X_test, y_test, thresholds=None, plot_cm=True):
    """
    Evalúa el modelo, imprime métricas y muestra matriz de confusión normalizada.
    """
    if hasattr(model, "predict_proba") and thresholds:
        probs = model.predict_proba(X_test)
        y_pred_adj = []
        for p in probs:
            candidates = [i for i, prob in enumerate(p) if prob >= thresholds.get(i, 0.5)]
            if candidates:
                y_pred_adj.append(candidates[np.argmax([p[i] for i in candidates])])
            else:
                y_pred_adj.append(np.argmax(p))
        y_pred_adj = np.array(y_pred_adj)
    else:
        y_pred_adj = model.predict(X_test)
    
    print(classification_report(y_test, y_pred_adj))
    
    if plot_cm:
        cm = confusion_matrix(y_test, y_pred_adj)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
        plt.title('Matriz de Confusión Normalizada')
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.show()
    
    return y_pred_adj

def save_model(model, path):
    """
    Guarda el modelo en disco usando joblib.
    """
    joblib.dump(model, path)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, params=None, cv=5):
    """
    Función combinada: entrena y evalúa un modelo. Devuelve modelo entrenado, F1 score y mejores parámetros.
    """
    if params is None:
        params = {}
    best_model, best_score, best_params = train_model(model, params, X_train, y_train, cv=cv)
    evaluate_model(best_model, X_test, y_test)
    return best_model, best_score, best_params
