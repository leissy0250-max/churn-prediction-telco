"""Construcción, entrenamiento y persistencia del modelo final (secciones 4.2, 5.1 y 6.2).

Modelo seleccionado: Regresión Logística (C = 10, penalización L2, sin balanceo)
con umbral de decisión elegido sobre las predicciones out-of-fold del entrenamiento.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def construir_preprocesador() -> ColumnTransformer:
    """Escalado de numéricas, one-hot de categóricas y paso directo de SeniorCitizen (45 columnas)."""
    return ColumnTransformer([
        ("num", StandardScaler(), config.VARS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         config.VARS_CATEGORICAS),
        ("bin", "passthrough", config.VARS_BINARIAS),
    ])


def construir_pipeline() -> Pipeline:
    """Pipeline completo: preprocesamiento + Regresión Logística."""
    return Pipeline([
        ("pre", construir_preprocesador()),
        ("clf", LogisticRegression(C=config.C_REGULARIZACION,
                                   max_iter=config.MAX_ITER)),
    ])


def predicciones_out_of_fold(X_train, y_train) -> np.ndarray:
    """Probabilidades out-of-fold con validación cruzada estratificada de 5 folds."""
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                         random_state=config.SEMILLA)
    return cross_val_predict(construir_pipeline(), X_train, y_train, cv=cv,
                             method="predict_proba")[:, 1]


def elegir_umbral(y_true, p_oof) -> tuple[float, float]:
    """Umbral que maximiza el F1 sobre las predicciones out-of-fold (el test no interviene)."""
    f1s = [f1_score(y_true, (p_oof >= u).astype(int)) for u in config.UMBRALES_CANDIDATOS]
    k = int(np.argmax(f1s))
    return float(config.UMBRALES_CANDIDATOS[k]), float(f1s[k])


def entrenar(X_train, y_train) -> dict:
    """Entrena el modelo final y devuelve un diccionario con el pipeline y su umbral."""
    p_oof = predicciones_out_of_fold(X_train, y_train)
    umbral, f1_oof = elegir_umbral(y_train, p_oof)
    pipeline = construir_pipeline().fit(X_train, y_train)
    # Perfil promedio del entrenamiento (en el espacio transformado): sirve de
    # referencia para explicar cada predicción individual en la app.
    media = pipeline.named_steps["pre"].transform(X_train).mean(axis=0)
    return {
        "pipeline": pipeline,
        "umbral": umbral,
        "media_transformada": media,
        "auc_cv": float(roc_auc_score(y_train, p_oof)),
        "auc_entrenamiento": float(roc_auc_score(
            y_train, pipeline.predict_proba(X_train)[:, 1])),
        "f1_cv": f1_oof,
    }


def guardar(artefacto: dict, ficha: dict) -> None:
    """Guarda el pipeline entrenado (joblib) y su ficha técnica (JSON)."""
    config.RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": artefacto["pipeline"],
                 "umbral": artefacto["umbral"],
                 "media_transformada": artefacto["media_transformada"]},
                config.RUTA_MODELO)
    with open(config.RUTA_FICHA, "w", encoding="utf-8") as f:
        json.dump(ficha, f, ensure_ascii=False, indent=2)


def cargar(ruta=config.RUTA_MODELO) -> dict:
    """Carga el modelo guardado: {'pipeline', 'umbral', 'media_transformada'}."""
    return joblib.load(ruta)


def cargar_ficha(ruta=config.RUTA_FICHA) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def entorno() -> dict:
    """Versiones del entorno de entrenamiento (trazabilidad del modelo)."""
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
