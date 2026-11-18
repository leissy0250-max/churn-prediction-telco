"""Prueba de la app web (Streamlit) sin navegador, con streamlit.testing."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
RUTA_APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app():
    return AppTest.from_file(RUTA_APP, default_timeout=120).run()


def test_la_app_inicia_sin_errores(app):
    assert not app.exception
    assert "Sistema Predictivo de Abandono" in app.title[0].value
    assert [t.label for t in app.tabs] == [
        "Predicción individual", "Predicción por lote (CSV)",
        "Desempeño del modelo", "Acerca del proyecto"]


def test_prediccion_individual_desde_la_app(app):
    app.button[0].click().run()
    assert not app.exception
    etiquetas = [m.label for m in app.metric]
    assert "Probabilidad de abandono" in etiquetas
    assert any("Riesgo ALTO" in md.value for md in app.markdown)


def test_prediccion_por_lote_con_etiquetas(app):
    app.selectbox[4].select("1,407 clientes del conjunto de prueba (con etiqueta real)").run()
    assert not app.exception
    valores = {m.label: m.value for m in app.metric}
    assert valores["Clientes evaluados"] == "1,407"
    assert valores["A contactar (p ≥ umbral)"] == "518"   # 272 VP + 246 FP
    assert valores["ROC-AUC"] == "0.835"
