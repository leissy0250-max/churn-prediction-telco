"""Generación de alertas de abandono para clientes nuevos (requisito RF-03).

Flujo: validar_entrada -> predecir -> (opcional) explicar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class ErrorDeEntrada(ValueError):
    """Error con un mensaje claro para el usuario cuando los datos no son válidos."""


# ---------------------------------------------------------------------------
# Validación y normalización de datos de entrada
# ---------------------------------------------------------------------------
def validar_entrada(df: pd.DataFrame) -> pd.DataFrame:
    """Comprueba columnas, tipos y categorías; devuelve las 19 variables listas para el modelo.

    - Normaliza el texto (minúsculas y sin espacios sobrantes), igual que en el entrenamiento.
    - Convierte las numéricas; si TotalCharges viene vacío (cliente nuevo) se estima
      como tenure x MonthlyCharges.
    - Lanza ErrorDeEntrada con la lista de problemas encontrados.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ErrorDeEntrada("No se recibieron registros de clientes.")

    faltantes = [c for c in config.VARIABLES if c not in df.columns]
    if faltantes:
        raise ErrorDeEntrada("Faltan columnas obligatorias: " + ", ".join(faltantes))

    X = df[config.VARIABLES].copy()
    problemas = []

    for c in config.VARS_CATEGORICAS:
        X[c] = X[c].astype(str).str.strip().str.lower()
        invalidos = sorted(set(X[c]) - set(config.VALORES_VALIDOS[c]))
        if invalidos:
            problemas.append(f"{c}: valores no válidos {invalidos}; "
                             f"permitidos {config.VALORES_VALIDOS[c]}")

    for c in config.VARS_NUMERICAS + config.VARS_BINARIAS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Cliente nuevo sin facturación acumulada: se estima el cargo total
    sin_total = X["TotalCharges"].isna()
    X.loc[sin_total, "TotalCharges"] = X.loc[sin_total, "tenure"] * X.loc[sin_total, "MonthlyCharges"]

    for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if X[c].isna().any():
            problemas.append(f"{c}: hay valores vacíos o no numéricos")
        elif (X[c] < 0).any():
            problemas.append(f"{c}: no se admiten valores negativos")
    if not X["SeniorCitizen"].isna().any() and not X["SeniorCitizen"].isin([0, 1]).all():
        problemas.append("SeniorCitizen: solo admite 0 (no) o 1 (sí)")

    if problemas:
        raise ErrorDeEntrada("Datos no válidos -> " + " | ".join(problemas))

    X["SeniorCitizen"] = X["SeniorCitizen"].astype(int)
    return X


# ---------------------------------------------------------------------------
# Predicción y nivel de riesgo
# ---------------------------------------------------------------------------
def nivel_riesgo(p: float) -> str:
    """bajo: p < 0.30 | medio: 0.30 <= p < 0.60 | alto: p >= 0.60."""
    if p >= config.CORTE_RIESGO_ALTO:
        return "Alto"
    if p >= config.CORTE_RIESGO_MEDIO:
        return "Medio"
    return "Bajo"


def predecir(modelo: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve probabilidad de abandono, nivel de riesgo, predicción y acción recomendada."""
    X = validar_entrada(df)
    umbral = modelo["umbral"]
    p = modelo["pipeline"].predict_proba(X)[:, 1]

    salida = pd.DataFrame(index=df.index)
    if config.COL_ID in df.columns:
        salida[config.COL_ID] = df[config.COL_ID].values
    salida["probabilidad_abandono"] = np.round(p, 4)
    salida["nivel_riesgo"] = [nivel_riesgo(v) for v in p]
    salida["prediccion"] = np.where(p >= umbral, "Abandona", "Permanece")
    salida["accion_recomendada"] = np.where(
        p >= umbral, "Contactar para retención", "Seguimiento regular")
    return salida


# ---------------------------------------------------------------------------
# Explicación de una predicción individual
# ---------------------------------------------------------------------------
def _variable_de_cada_columna(preprocesador) -> list[str]:
    """Asigna a cada una de las 45 columnas transformadas su variable original."""
    variables = list(config.VARS_NUMERICAS)
    codificador = preprocesador.named_transformers_["cat"]
    for var, categorias in zip(config.VARS_CATEGORICAS, codificador.categories_):
        variables += [var] * len(categorias)
    variables += list(config.VARS_BINARIAS)
    return variables


def explicar(modelo: dict, registro: pd.DataFrame) -> pd.DataFrame:
    """Aporte de cada variable al log-odds del cliente frente al cliente promedio.

    Como la Regresión Logística es lineal en el log-odds, el aporte de la columna j es
    beta_j * (z_j - media_j). Se suman los aportes de las columnas de una misma variable.
    Valores positivos aumentan el riesgo de abandono; negativos lo reducen.
    """
    X = validar_entrada(registro.iloc[[0]])
    pre = modelo["pipeline"].named_steps["pre"]
    coef = modelo["pipeline"].named_steps["clf"].coef_[0]
    z = pre.transform(X)[0]
    aportes = coef * (z - modelo["media_transformada"])
    tabla = (pd.DataFrame({"variable": _variable_de_cada_columna(pre), "aporte": aportes})
             .groupby("variable", as_index=False)["aporte"].sum())
    tabla["valor"] = [X.iloc[0][v] for v in tabla["variable"]]
    tabla["variable_es"] = tabla["variable"].map(config.NOMBRES_ES)
    return tabla.sort_values("aporte", key=np.abs, ascending=False).reset_index(drop=True)


def explicar_por_grupo(modelo: dict, registro: pd.DataFrame) -> pd.DataFrame:
    """Suma los aportes por grupo de variables (ver config.GRUPOS_EXPLICACION)."""
    detalle = explicar(modelo, registro)
    grupo_de = {v: g for g, vs in config.GRUPOS_EXPLICACION.items() for v in vs}
    detalle["grupo"] = detalle["variable"].map(grupo_de)
    tabla = detalle.groupby("grupo", as_index=False)["aporte"].sum()
    tabla["efecto"] = np.where(tabla["aporte"] > 0, "Aumenta el riesgo", "Reduce el riesgo")
    return tabla.sort_values("aporte", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga segura del modelo (para la app desplegada)
# ---------------------------------------------------------------------------
def cargar_modelo_seguro() -> tuple[dict, str]:
    """Carga models/modelo_churn.joblib; si falla (p. ej. otra versión de scikit-learn),
    reentrena el modelo con la misma semilla y los mismos datos. Devuelve (modelo, origen)."""
    from . import modelo as m

    try:
        artefacto = m.cargar()
        if {"pipeline", "umbral", "media_transformada"} <= set(artefacto):
            return artefacto, "archivo"
    except Exception:  # noqa: BLE001 - cualquier fallo de carga activa el reentrenamiento
        pass

    from .datos import obtener_particion

    X_train, _, y_train, _ = obtener_particion()
    artefacto = m.entrenar(X_train, y_train)
    return artefacto, "reentrenado"
