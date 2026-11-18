"""Pruebas de la generación de alertas (RF-03) y de la validación de entradas."""
import pandas as pd
import pytest

from churn import config, prediccion


def test_probabilidad_entre_0_y_1(modelo_entrenado):
    df = pd.read_csv(config.DIR_EJEMPLOS / "clientes_nuevos_ejemplo.csv")
    r = prediccion.predecir(modelo_entrenado, df)
    assert len(r) == 10
    assert r["probabilidad_abandono"].between(0, 1).all()
    assert list(r["customerID"]) == list(df["customerID"])


@pytest.mark.parametrize("p, nivel", [(0.0, "Bajo"), (0.299, "Bajo"), (0.30, "Medio"),
                                      (0.599, "Medio"), (0.60, "Alto"), (1.0, "Alto")])
def test_cortes_de_nivel_de_riesgo(p, nivel):
    assert prediccion.nivel_riesgo(p) == nivel


def test_cliente_alto_riesgo(modelo_entrenado, cliente_alto_riesgo):
    r = prediccion.predecir(modelo_entrenado, cliente_alto_riesgo).iloc[0]
    assert r["probabilidad_abandono"] >= 0.60
    assert r["nivel_riesgo"] == "Alto"
    assert r["accion_recomendada"] == "Contactar para retención"


def test_cliente_bajo_riesgo(modelo_entrenado, cliente_bajo_riesgo):
    r = prediccion.predecir(modelo_entrenado, cliente_bajo_riesgo).iloc[0]
    assert r["probabilidad_abandono"] < 0.10
    assert r["nivel_riesgo"] == "Bajo"
    assert r["prediccion"] == "Permanece"


def test_normaliza_texto_como_en_el_entrenamiento(modelo_entrenado, cliente_alto_riesgo):
    sucio = cliente_alto_riesgo.copy()
    sucio["Contract"] = "  MONTH-TO-MONTH "
    sucio["InternetService"] = "Fiber Optic"
    p1 = prediccion.predecir(modelo_entrenado, cliente_alto_riesgo)["probabilidad_abandono"]
    p2 = prediccion.predecir(modelo_entrenado, sucio)["probabilidad_abandono"]
    assert p1.iloc[0] == p2.iloc[0]


def test_cliente_nuevo_sin_cargo_total(modelo_entrenado, cliente_alto_riesgo):
    nuevo = cliente_alto_riesgo.copy()
    nuevo["tenure"], nuevo["TotalCharges"] = 0, " "
    r = prediccion.predecir(modelo_entrenado, nuevo)
    assert 0 <= r["probabilidad_abandono"].iloc[0] <= 1


def test_error_si_falta_una_columna(modelo_entrenado, cliente_alto_riesgo):
    with pytest.raises(prediccion.ErrorDeEntrada, match="Contract"):
        prediccion.predecir(modelo_entrenado, cliente_alto_riesgo.drop(columns=["Contract"]))


def test_error_si_categoria_no_valida(modelo_entrenado, cliente_alto_riesgo):
    malo = cliente_alto_riesgo.copy()
    malo["Contract"] = "trimestral"
    with pytest.raises(prediccion.ErrorDeEntrada, match="trimestral"):
        prediccion.predecir(modelo_entrenado, malo)


def test_error_si_antiguedad_negativa(modelo_entrenado, cliente_alto_riesgo):
    malo = cliente_alto_riesgo.copy()
    malo["tenure"] = -3
    with pytest.raises(prediccion.ErrorDeEntrada, match="negativos"):
        prediccion.predecir(modelo_entrenado, malo)


def test_explicacion_suma_el_log_odds(modelo_entrenado, cliente_alto_riesgo):
    """La suma de aportes = log-odds del cliente - log-odds del cliente promedio."""
    import numpy as np

    detalle = prediccion.explicar(modelo_entrenado, cliente_alto_riesgo)
    X = prediccion.validar_entrada(cliente_alto_riesgo)
    clf = modelo_entrenado["pipeline"].named_steps["clf"]
    logit = modelo_entrenado["pipeline"].decision_function(X)[0]
    logit_medio = clf.intercept_[0] + clf.coef_[0] @ modelo_entrenado["media_transformada"]
    assert detalle["aporte"].sum() == pytest.approx(logit - logit_medio)
    assert len(detalle) == 19
    grupos = prediccion.explicar_por_grupo(modelo_entrenado, cliente_alto_riesgo)
    assert np.isclose(grupos["aporte"].sum(), detalle["aporte"].sum())
