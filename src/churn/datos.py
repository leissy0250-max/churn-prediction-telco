"""Carga, limpieza y partición del dataset (secciones 3.1 a 3.3 del informe)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def cargar_datos(ruta=config.RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV original de Telco Customer Churn (7,043 x 21)."""
    return pd.read_csv(ruta)


def normalizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Pasa a minúsculas y quita espacios sobrantes en las columnas de texto."""
    df = df.copy()
    columnas_texto = [c for c in config.VARS_CATEGORICAS + [config.COL_OBJETIVO]
                      if c in df.columns]
    for c in columnas_texto:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza descrita en la sección 3.2.

    1. Normaliza el texto de las variables categóricas.
    2. Convierte TotalCharges a número (los espacios en blanco pasan a NaN).
    3. Elimina los 11 registros sin TotalCharges (clientes con antigüedad 0).
    """
    df = normalizar_texto(df)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df


def separar_x_y(df: pd.DataFrame):
    """Devuelve la matriz de predictores X (19 variables) y la etiqueta y (1 = abandona)."""
    y = (df[config.COL_OBJETIVO] == "yes").astype(int)
    X = df[config.VARIABLES].copy()
    return X, y


def particionar(X, y):
    """Partición estratificada 80/20 con semilla 42 -> 5,625 / 1,407 registros."""
    return train_test_split(
        X, y,
        test_size=config.PROP_PRUEBA,
        stratify=y,
        random_state=config.SEMILLA,
    )


def obtener_particion():
    """Atajo: carga, limpia y particiona. Devuelve X_train, X_test, y_train, y_test."""
    df = limpiar_datos(cargar_datos())
    X, y = separar_x_y(df)
    return particionar(X, y)
