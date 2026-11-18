"""Configuración central del Sistema Predictivo de Abandono de Clientes.

Reúne en un solo lugar las rutas, la semilla, las variables y los
parámetros del modelo final para garantizar la reproducibilidad (RNF-02).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DIR_EJEMPLOS = RAIZ / "data" / "ejemplos"
RUTA_MODELO = RAIZ / "models" / "modelo_churn.joblib"
RUTA_FICHA = RAIZ / "models" / "ficha_modelo.json"
DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figuras"

# ---------------------------------------------------------------------------
# Parámetros experimentales (iguales a los del informe, secciones 3.3 y 5.1)
# ---------------------------------------------------------------------------
SEMILLA = 42            # random_state de todos los procesos aleatorios
PROP_PRUEBA = 0.20      # partición estratificada 80/20 -> 5,625 / 1,407
N_FOLDS = 5             # validación cruzada estratificada de 5 folds
C_REGULARIZACION = 10   # mejor C de la Regresión Logística (tabla 1 del informe)
MAX_ITER = 2000

# Rejilla de umbrales evaluada sobre las predicciones out-of-fold
UMBRALES_CANDIDATOS = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 ... 0.95

# Cortes de nivel de riesgo (sección 5.2 del informe):
# bajo: p < 0.30 | medio: 0.30 <= p < 0.60 | alto: p >= 0.60
CORTE_RIESGO_MEDIO = 0.30
CORTE_RIESGO_ALTO = 0.60

# Criterios de aceptación (sección 6.1 del informe)
CRITERIOS_ACEPTACION = {
    "roc_auc": (">=", 0.80),
    "recall": (">=", 0.70),
    "precision": (">=", 0.50),
    "f1": (">=", 0.60),
    "accuracy": (">", 0.7342),
    "brier": ("<=", 0.16),
    "brecha_sobreajuste": ("<=", 0.05),  # AUC entrenamiento - AUC validación cruzada
}

# ---------------------------------------------------------------------------
# Variables del dataset Telco Customer Churn
# ---------------------------------------------------------------------------
COL_ID = "customerID"
COL_OBJETIVO = "Churn"

VARS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
VARS_BINARIAS = ["SeniorCitizen"]
VARS_CATEGORICAS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# Orden original de las 19 variables predictoras
VARIABLES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Valores válidos de cada variable categórica (ya normalizados a minúsculas)
_SI_NO = ["yes", "no"]
_SERVICIO_INTERNET = ["yes", "no", "no internet service"]
VALORES_VALIDOS = {
    "gender": ["female", "male"],
    "Partner": _SI_NO,
    "Dependents": _SI_NO,
    "PhoneService": _SI_NO,
    "MultipleLines": ["yes", "no", "no phone service"],
    "InternetService": ["dsl", "fiber optic", "no"],
    "OnlineSecurity": _SERVICIO_INTERNET,
    "OnlineBackup": _SERVICIO_INTERNET,
    "DeviceProtection": _SERVICIO_INTERNET,
    "TechSupport": _SERVICIO_INTERNET,
    "StreamingTV": _SERVICIO_INTERNET,
    "StreamingMovies": _SERVICIO_INTERNET,
    "Contract": ["month-to-month", "one year", "two year"],
    "PaperlessBilling": _SI_NO,
    "PaymentMethod": [
        "electronic check", "mailed check",
        "bank transfer (automatic)", "credit card (automatic)",
    ],
}

# Nombres en español para mostrar en la app y en los reportes
NOMBRES_ES = {
    "gender": "Género",
    "SeniorCitizen": "Adulto mayor",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes",
    "tenure": "Antigüedad (meses)",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Líneas múltiples",
    "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de dispositivos",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "TV por streaming",
    "StreamingMovies": "Películas por streaming",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Facturación electrónica",
    "PaymentMethod": "Método de pago",
    "MonthlyCharges": "Cargo mensual (USD)",
    "TotalCharges": "Cargo total (USD)",
}

# Grupos de variables para explicar una predicción (agrupar evita leer por separado
# variables muy correlacionadas, como el cargo mensual y los servicios contratados)
GRUPOS_EXPLICACION = {
    "Contrato y forma de pago": ["Contract", "PaymentMethod", "PaperlessBilling"],
    "Antigüedad y facturación acumulada": ["tenure", "TotalCharges"],
    "Servicios contratados y cargo mensual": [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "MonthlyCharges"],
    "Perfil del cliente": ["gender", "SeniorCitizen", "Partner", "Dependents"],
}
