# Sistema Predictivo de Abandono de Clientes (Churn Prediction)

**Aprendizaje Estadístico Supervisado en Empresas de Telecomunicaciones**

[![Pruebas del sistema](https://github.com/TU-USUARIO/churn-prediction-telco/actions/workflows/pruebas.yml/badge.svg)](https://github.com/TU-USUARIO/churn-prediction-telco/actions/workflows/pruebas.yml)
[![App web](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-telco-upao.streamlit.app)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-USUARIO/churn-prediction-telco/blob/main/notebooks/Ejecucion_en_Colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Licencia](https://img.shields.io/badge/licencia-MIT-green)

Universidad Privada Antenor Orrego · Programa de Ingeniería de Sistemas e Inteligencia Artificial · Curso de Aprendizaje Estadístico · Trujillo, 2026

| Recurso | Enlace |
|---|---|
| App web desplegada | https://churn-telco-upao.streamlit.app |
| Página del proyecto (3 pestañas) | https://TU-USUARIO.github.io/churn-prediction-telco/ |
| Informe completo (PDF) | [docs/informe/Informe_Proyecto_Churn.pdf](docs/informe/Informe_Proyecto_Churn.pdf) |

---

## 1. Documentación del proyecto

### Problema
La pérdida de abonados (*churn*) reduce los ingresos recurrentes de las operadoras. En el dataset de
referencia, **1,869 de 7,043 clientes (26.54 %)** cancelaron el servicio. Sin un sistema predictivo, las
campañas de retención son tardías o masivas y otorgan incentivos a clientes que no pensaban irse.

### Objetivo
Diseñar, entrenar, evaluar y **desplegar** un modelo de aprendizaje estadístico supervisado que estime la
probabilidad de abandono de cada cliente y genere alertas para el área de fidelización.

### Datos
*Telco Customer Churn* (IBM, publicado en Kaggle): 7,043 clientes y 21 variables (perfil del cliente,
servicios contratados y condiciones del contrato). Tras eliminar 11 registros sin `TotalCharges` quedan
7,032, divididos de forma estratificada en **5,625 de entrenamiento y 1,407 de prueba** (semilla 42).

### Metodología (ciclo de desarrollo de Statistical / Machine Learning)
1. Definición del problema y requisitos → 2. Datos y preprocesamiento → 3. Elección del modelo →
4. Entrenamiento con validación cruzada de 5 folds → 5. Aplicación al conjunto de prueba →
6. Evaluación con criterios de aceptación → 7. **Despliegue (este repositorio y la app web)**.

Se compararon cuatro modelos (Regresión Logística, Regresión Logística + PCA, Random Forest y XGBoost)
con cuatro estrategias de balanceo. Se seleccionó la **Regresión Logística (C = 10, umbral 0.33)** por
cumplir los siete criterios de aceptación y por su interpretabilidad.

### Resultados en el conjunto de prueba (1,407 clientes)

| Métrica | Valor | Meta |
|---|---|---|
| ROC-AUC | **0.835** | ≥ 0.80 |
| Recall (abandonos detectados) | **72.7 %** (272 de 374) | ≥ 0.70 |
| Precisión de las alertas | **52.5 %** | ≥ 0.50 |
| F1-Score | **0.610** | ≥ 0.60 |
| Exactitud | **75.3 %** | > 73.42 % |
| Brier score | **0.140** | ≤ 0.16 |
| Brecha de sobreajuste | **0.005** | ≤ 0.05 |

Contactar a los clientes que marca el modelo es **1.98 veces más efectivo** que contactar clientes al azar.
Las variables más influyentes son el tipo de contrato, la antigüedad, el servicio de internet y los cargos.

---

## 2. Código del sistema

```
churn-prediction-telco/
├── app.py                        # App web (Streamlit): punto de entrada del despliegue
├── requirements.txt              # Dependencias con versiones fijas
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── ejemplos/                 # clientes de ejemplo y conjunto de prueba etiquetado
├── src/
│   ├── churn/                    # paquete del sistema
│   │   ├── config.py             # rutas, semilla, variables, umbral y criterios
│   │   ├── datos.py              # carga, limpieza y partición 80/20
│   │   ├── modelo.py             # pipeline, validación cruzada, umbral y guardado
│   │   ├── evaluacion.py         # métricas, criterios de aceptación y gráficos
│   │   └── prediccion.py         # validación de entradas, alertas y explicación
│   ├── entrenar.py               # script: entrena, evalúa y guarda el modelo
│   └── predecir.py               # script: alertas para un CSV de clientes
├── models/                       # modelo entrenado (.joblib) y ficha técnica (.json)
├── reports/                      # métricas y figuras de evaluación
├── notebooks/                    # cuadernos de Google Colab
├── tests/                        # pruebas automáticas (pytest)
├── docs/                         # página del proyecto (GitHub Pages) e informe PDF
└── .github/workflows/pruebas.yml # integración continua (GitHub Actions)
```

Tecnologías: Python 3.13, pandas, NumPy, scikit-learn, joblib, Matplotlib, Streamlit y pytest.

---

## 3. Ejecución y pruebas

### Ejecutar en local
```bash
git clone https://github.com/TU-USUARIO/churn-prediction-telco.git
cd churn-prediction-telco
pip install -r requirements.txt

python src/entrenar.py                                        # entrena y evalúa (≈ 5 s)
python src/predecir.py data/ejemplos/clientes_nuevos_ejemplo.csv   # alertas por lote
pytest -v                                                     # 31 pruebas automáticas
streamlit run app.py                                          # app en http://localhost:8501
```

### Ejecutar en Google Colab
Abra `notebooks/Ejecucion_en_Colab.ipynb` con el botón **Abrir en Colab** de la parte superior.

### Pruebas automáticas
| Archivo | Qué verifica | Pruebas |
|---|---|---|
| `tests/test_datos.py` | Carga (7,043 × 21), limpieza (7,032), partición 5,625/1,407 sin fuga | 6 |
| `tests/test_modelo.py` | Reproduce las métricas del informe y cumple los criterios de aceptación | 7 |
| `tests/test_prediccion.py` | Niveles de riesgo, casos de alto y bajo riesgo, validación de entradas | 15 |
| `tests/test_app.py` | La app inicia, predice un cliente y procesa un lote | 3 |

Las pruebas se ejecutan solas en cada `push` mediante GitHub Actions (pestaña **Actions**).

---

## 4. App web

La app está desplegada en **Streamlit Community Cloud**: https://churn-telco-upao.streamlit.app

- **Predicción individual:** formulario con las 19 variables → probabilidad, nivel de riesgo, acción y explicación.
- **Predicción por lote:** carga de un CSV → ranking de clientes por riesgo y descarga de resultados; si el
  archivo trae la columna `Churn`, valida el modelo (matriz de confusión y métricas).
- **Desempeño del modelo:** criterios de aceptación, matriz de confusión, curva ROC y razones de momios.

Niveles de riesgo: **bajo** p < 0.30 · **medio** 0.30 ≤ p < 0.60 · **alto** p ≥ 0.60.
Acción: *contactar para retención* si p ≥ 0.33 (umbral del modelo).

---

## Autores
Ruiz Cueva, Johan · Chavez Sampen, Luis · Lopez Fasabi, Elisson · Polo Valderrama, Benjamin ·
Cardenas Julian, Piero · Sanchez Velasquez, Gabriela Ximena

**Docente:** Sagastegui Chigne, Teobaldo Herman

## Uso responsable
El dataset es un conjunto de ejemplo de IBM. Antes de usar el sistema con clientes reales debe recalibrarse
con datos de la operadora y revisarse con métricas por subgrupos. La app no almacena los datos ingresados.

## Licencia
MIT. Ver [LICENSE](LICENSE).
