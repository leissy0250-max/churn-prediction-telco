"""Pruebas del modelo: reproducibilidad (RNF-02) y criterios de aceptación (sección 6.1)."""
import numpy as np
import pytest

from churn import config, evaluacion, modelo


def test_modelo_guardado_y_configuracion(modelo_entrenado):
    clf = modelo_entrenado["pipeline"].named_steps["clf"]
    assert clf.C == 10
    assert modelo_entrenado["umbral"] == pytest.approx(0.33)


def test_45_columnas_tras_preprocesamiento(modelo_entrenado, particion):
    X_train = particion[0]
    Z = modelo_entrenado["pipeline"].named_steps["pre"].transform(X_train.head(5))
    assert Z.shape == (5, 45)


def test_metricas_de_prueba_reproducen_el_informe(modelo_entrenado, particion):
    _, X_test, _, y_test = particion
    p = modelo_entrenado["pipeline"].predict_proba(X_test)[:, 1]
    m = evaluacion.calcular_metricas(y_test, p, modelo_entrenado["umbral"])
    assert m["roc_auc"] == pytest.approx(0.8352, abs=1e-3)
    assert m["matriz_confusion"] == {"VN": 787, "FP": 246, "FN": 102, "VP": 272}


def test_cumple_criterios_de_aceptacion(modelo_entrenado, particion):
    _, X_test, _, y_test = particion
    p = modelo_entrenado["pipeline"].predict_proba(X_test)[:, 1]
    m = evaluacion.calcular_metricas(y_test, p, modelo_entrenado["umbral"])
    m["brecha_sobreajuste"] = 0.0  # se verifica aparte en la ficha técnica
    resultado = evaluacion.verificar_criterios(m)
    fallidos = [k for k, v in resultado.items() if not v["cumple"]]
    assert not fallidos, f"Criterios no cumplidos: {fallidos}"


def test_supera_al_clasificador_trivial(modelo_entrenado, particion):
    _, X_test, _, y_test = particion
    p = modelo_entrenado["pipeline"].predict_proba(X_test)[:, 1]
    m = evaluacion.calcular_metricas(y_test, p, modelo_entrenado["umbral"])
    assert m["accuracy"] > 1 - y_test.mean()  # 0.7342
    assert m["precision"] / y_test.mean() > 1.9  # lift aproximado de 1.98


def test_ficha_tecnica_sin_sobreajuste():
    ficha = modelo.cargar_ficha()
    assert ficha["validacion_cruzada"]["brecha_sobreajuste"] <= 0.05
    assert all(c["cumple"] for c in ficha["criterios_aceptacion"].values())


def test_reentrenamiento_reproducible(particion, modelo_entrenado):
    """Con la semilla 42, reentrenar produce el mismo umbral y las mismas probabilidades."""
    X_train, X_test, y_train, _ = particion
    nuevo = modelo.entrenar(X_train, y_train)
    assert nuevo["umbral"] == modelo_entrenado["umbral"]
    np.testing.assert_allclose(
        nuevo["pipeline"].predict_proba(X_test)[:, 1],
        modelo_entrenado["pipeline"].predict_proba(X_test)[:, 1], atol=1e-6)
    assert config.SEMILLA == 42
