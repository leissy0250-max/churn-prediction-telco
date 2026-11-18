"""Pruebas de datos: requisitos RD-01 (carga) y RD-02 (limpieza) y partición 80/20."""
from churn import config, datos


def test_carga_dimensiones(df_original):
    assert df_original.shape == (7043, 21)


def test_variable_objetivo_y_tasa_de_abandono(df_original):
    churn = df_original["Churn"].str.strip().str.lower()
    assert set(churn) == {"yes", "no"}
    assert (churn == "yes").sum() == 1869  # 26.54 %


def test_limpieza_elimina_11_registros_sin_totalcharges(df_original):
    limpio = datos.limpiar_datos(df_original)
    assert len(limpio) == 7032
    assert limpio["TotalCharges"].isna().sum() == 0


def test_texto_normalizado(df_original):
    limpio = datos.limpiar_datos(df_original)
    for c in config.VARS_CATEGORICAS:
        assert set(limpio[c]) <= set(config.VALORES_VALIDOS[c]), c


def test_particion_estratificada(particion):
    X_train, X_test, y_train, y_test = particion
    assert (len(X_train), len(X_test)) == (5625, 1407)
    assert X_train.shape[1] == 19
    assert abs(y_train.mean() - y_test.mean()) < 0.005  # misma proporción de abandono
    assert int(y_test.sum()) == 374


def test_sin_fuga_entre_entrenamiento_y_prueba(particion):
    X_train, X_test, _, _ = particion
    assert set(X_train.index).isdisjoint(X_test.index)
