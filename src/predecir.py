"""Genera alertas de abandono para un archivo CSV de clientes (requisito RF-03).

Uso (desde la raíz del repositorio):
    python src/predecir.py data/ejemplos/clientes_nuevos_ejemplo.csv
    python src/predecir.py mis_clientes.csv --salida reports/alertas.csv
"""
import argparse

import pandas as pd

from churn import prediccion


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción de abandono por lote")
    parser.add_argument("entrada", help="CSV con las 19 variables del dataset")
    parser.add_argument("--salida", default="reports/alertas_abandono.csv",
                        help="CSV de resultados (por defecto reports/alertas_abandono.csv)")
    args = parser.parse_args()

    modelo, origen = prediccion.cargar_modelo_seguro()
    clientes = pd.read_csv(args.entrada)
    resultado = prediccion.predecir(modelo, clientes).sort_values(
        "probabilidad_abandono", ascending=False)
    resultado.to_csv(args.salida, index=False)

    print(f"Modelo: {origen} | umbral = {modelo['umbral']:.2f}")
    print(f"Clientes evaluados: {len(resultado)}")
    print(resultado["nivel_riesgo"].value_counts().rename("clientes").to_string())
    print(f"\nTop 5 clientes con mayor riesgo:\n{resultado.head(5).to_string(index=False)}")
    print(f"\nResultados guardados en {args.salida}")


if __name__ == "__main__":
    main()
