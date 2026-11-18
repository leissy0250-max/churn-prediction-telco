"""Métricas y gráficos de evaluación (secciones 6.1 y 6.2 del informe)."""
from __future__ import annotations

import operator

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

from . import config

_OPERADORES = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}


def calcular_metricas(y_true, p, umbral: float) -> dict:
    """Calcula las métricas de la tabla de la sección 6.1 para un umbral dado."""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    y_pred = (p >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    mse = float(np.mean((y_true - p) ** 2))
    return {
        "umbral": float(umbral),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "especificidad": float(vn / (vn + fp)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "mae": float(mean_absolute_error(y_true, p)),
        "rmse": float(np.sqrt(mse)),
        "matriz_confusion": {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)},
    }


def verificar_criterios(metricas: dict) -> dict:
    """Compara las métricas con los criterios de aceptación de la sección 6.1."""
    resultado = {}
    for nombre, (op, meta) in config.CRITERIOS_ACEPTACION.items():
        valor = metricas[nombre]
        resultado[nombre] = {
            "valor": round(valor, 4),
            "meta": f"{op} {meta}",
            "cumple": bool(_OPERADORES[op](valor, meta)),
        }
    return resultado


def razones_de_momios(pipeline, top: int = 10):
    """Coeficientes de la Regresión Logística convertidos a razones de momios (odds ratio)."""
    nombres = pipeline.named_steps["pre"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    orden = np.argsort(np.abs(coefs))[::-1][:top]
    return [(nombres[i].split("__", 1)[1], float(coefs[i]), float(np.exp(coefs[i])))
            for i in orden]


# ---------------------------------------------------------------------------
# Gráficos (se usan en reports/ y en la app web)
# ---------------------------------------------------------------------------
def figura_matriz_confusion(mc: dict):
    import matplotlib.pyplot as plt

    m = np.array([[mc["VN"], mc["FP"]], [mc["FN"], mc["VP"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(m, cmap="Blues")
    etiquetas = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if m[i, j] > m.max() / 2 else "#1f2937"
            ax.text(j, i, f"{m[i, j]}\n({etiquetas[i][j]})", ha="center",
                    va="center", color=color, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1], ["No abandona", "Abandona"])
    ax.set_yticks([0, 1], ["No abandona", "Abandona"])
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión (conjunto de prueba)", fontsize=10)
    fig.tight_layout()
    return fig


def figura_roc(y_true, p, auc: float):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, p)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"Regresión Logística (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1, label="Azar (AUC = 0.500)")
    ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_title("Curva ROC (conjunto de prueba)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


_VALORES_ES = {
    "yes": "sí", "no": "no", "fiber optic": "fibra óptica", "dsl": "DSL",
    "month-to-month": "mes a mes", "one year": "un año", "two year": "dos años",
    "electronic check": "cheque electrónico", "mailed check": "cheque por correo",
    "bank transfer (automatic)": "transferencia automática",
    "credit card (automatic)": "tarjeta automática", "female": "femenino", "male": "masculino",
}


def nombre_legible(columna: str) -> str:
    """'Contract_month-to-month' -> 'Tipo de contrato: mes a mes'."""
    for var in config.VARS_CATEGORICAS:
        if columna.startswith(var + "_"):
            valor = columna[len(var) + 1:]
            return f"{config.NOMBRES_ES[var]}: {_VALORES_ES.get(valor, valor)}"
    return config.NOMBRES_ES.get(columna, columna)


def figura_razones_momios(pipeline, top: int = 10):
    """Razones de momios de las categorías con mayor efecto.

    Se omiten las numéricas (cargo mensual, cargo total y antigüedad están muy
    correlacionadas y sus coeficientes no se leen por separado) y las categorías
    redundantes 'no internet service' / 'no phone service'.
    """
    import matplotlib.pyplot as plt

    datos = [d for d in razones_de_momios(pipeline, top=60)
             if d[0] not in config.VARS_NUMERICAS + config.VARS_BINARIAS
             and "no internet service" not in d[0] and "no phone service" not in d[0]][:top]
    datos = datos[::-1]
    nombres = [nombre_legible(d[0]) for d in datos]
    odds = [d[2] for d in datos]
    colores = ["#dc2626" if o > 1 else "#16a34a" for o in odds]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(nombres, [o - 1 for o in odds], left=1, color=colores)
    ax.axvline(1, color="#374151", lw=1)
    for y, o in enumerate(odds):
        ax.text(o + (0.04 if o > 1 else -0.04), y, f"{o:.2f}", va="center",
                ha="left" if o > 1 else "right", fontsize=7)
    ax.set_xlim(0, max(odds) + 0.6)
    ax.set_xlabel("Razón de momios (> 1 aumenta el riesgo, < 1 lo reduce)", fontsize=8)
    ax.set_title("Categorías con mayor efecto en la Regresión Logística", fontsize=10)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return fig
