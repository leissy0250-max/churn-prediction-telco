"""Entrena el modelo final, lo evalúa en el conjunto de prueba y guarda los artefactos.

Uso (desde la raíz del repositorio):
    python src/entrenar.py

Genera:
    models/modelo_churn.joblib      pipeline entrenado + umbral
    models/ficha_modelo.json        ficha técnica (parámetros, métricas, entorno)
    reports/metricas_prueba.json    métricas del conjunto de prueba
    reports/figuras/*.png           matriz de confusión, curva ROC y razones de momios
    data/ejemplos/*.csv             archivos de ejemplo para la app web
"""
import json

import matplotlib

matplotlib.use("Agg")

from churn import __version__, config, datos, evaluacion, modelo  # noqa: E402


def main() -> None:
    print("1/5 Cargando y limpiando datos...")
    df = datos.limpiar_datos(datos.cargar_datos())
    X, y = datos.separar_x_y(df)
    X_train, X_test, y_train, y_test = datos.particionar(X, y)
    print(f"    registros: {len(df):,} | entrenamiento: {len(X_train):,} | prueba: {len(X_test):,}")

    print("2/5 Validación cruzada de 5 folds y selección del umbral...")
    artefacto = modelo.entrenar(X_train, y_train)
    print(f"    AUC out-of-fold = {artefacto['auc_cv']:.4f} | umbral = {artefacto['umbral']:.2f}")

    print("3/5 Evaluación única en el conjunto de prueba...")
    p_test = artefacto["pipeline"].predict_proba(X_test)[:, 1]
    met = evaluacion.calcular_metricas(y_test, p_test, artefacto["umbral"])
    brecha = artefacto["auc_entrenamiento"] - artefacto["auc_cv"]
    criterios = evaluacion.verificar_criterios({**met, "brecha_sobreajuste": brecha})
    for k in ["roc_auc", "accuracy", "precision", "recall", "f1", "brier"]:
        print(f"    {k:<10} = {met[k]:.4f}")
    print(f"    matriz de confusión = {met['matriz_confusion']}")
    cumple = sum(c["cumple"] for c in criterios.values())
    print(f"    criterios de aceptación cumplidos: {cumple} de {len(criterios)}")

    print("4/5 Guardando modelo y ficha técnica...")
    ficha = {
        "proyecto": "Sistema Predictivo de Abandono de Clientes (Churn Prediction)",
        "version_modelo": __version__,
        "algoritmo": "Regresión Logística (penalización L2)",
        "hiperparametros": {"C": config.C_REGULARIZACION, "max_iter": config.MAX_ITER,
                            "balanceo": "ninguno"},
        "umbral_decision": artefacto["umbral"],
        "cortes_riesgo": {"medio": config.CORTE_RIESGO_MEDIO, "alto": config.CORTE_RIESGO_ALTO},
        "datos": {"dataset": "Telco Customer Churn (IBM / Kaggle)", "registros": len(df),
                  "entrenamiento": len(X_train), "prueba": len(X_test),
                  "semilla": config.SEMILLA, "variables": config.VARIABLES},
        "validacion_cruzada": {"folds": config.N_FOLDS,
                               "auc_out_of_fold": round(artefacto["auc_cv"], 4),
                               "f1_out_of_fold": round(artefacto["f1_cv"], 4),
                               "auc_entrenamiento": round(artefacto["auc_entrenamiento"], 4),
                               "brecha_sobreajuste": round(brecha, 4)},
        "metricas_prueba": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()},
        "criterios_aceptacion": criterios,
        "razones_de_momios_top10": [
            {"columna": n, "coeficiente": round(c, 4), "odds_ratio": round(o, 4)}
            for n, c, o in evaluacion.razones_de_momios(artefacto["pipeline"], 10)],
        "entorno": modelo.entorno(),
    }
    modelo.guardar(artefacto, ficha)
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    with open(config.DIR_REPORTES / "metricas_prueba.json", "w", encoding="utf-8") as f:
        json.dump({"metricas": ficha["metricas_prueba"], "criterios": criterios}, f,
                  ensure_ascii=False, indent=2)

    print("5/5 Generando figuras y archivos de ejemplo...")
    evaluacion.figura_matriz_confusion(met["matriz_confusion"]).savefig(
        config.DIR_FIGURAS / "matriz_confusion.png", dpi=150)
    evaluacion.figura_roc(y_test, p_test, met["roc_auc"]).savefig(
        config.DIR_FIGURAS / "curva_roc.png", dpi=150)
    evaluacion.figura_razones_momios(artefacto["pipeline"]).savefig(
        config.DIR_FIGURAS / "razones_momios.png", dpi=150)

    # Conjunto de prueba con su etiqueta real (para validar la app con datos etiquetados)
    config.DIR_EJEMPLOS.mkdir(parents=True, exist_ok=True)
    prueba = df.loc[X_test.index, [config.COL_ID] + config.VARIABLES + [config.COL_OBJETIVO]]
    prueba.to_csv(config.DIR_EJEMPLOS / "conjunto_prueba_etiquetado.csv", index=False)
    # 10 clientes del conjunto de prueba sin la etiqueta (simulan clientes nuevos)
    prueba.drop(columns=[config.COL_OBJETIVO]).head(10).to_csv(
        config.DIR_EJEMPLOS / "clientes_nuevos_ejemplo.csv", index=False)
    print("Listo. Artefactos guardados en models/, reports/ y data/ejemplos/.")


if __name__ == "__main__":
    main()
