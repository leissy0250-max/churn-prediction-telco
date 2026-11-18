"""Datos y objetos compartidos por las pruebas."""
import pandas as pd
import pytest

from churn import datos, modelo


@pytest.fixture(scope="session")
def df_original():
    return datos.cargar_datos()


@pytest.fixture(scope="session")
def particion():
    return datos.obtener_particion()  # X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def modelo_entrenado():
    return modelo.cargar()


@pytest.fixture
def cliente_alto_riesgo():
    """Cliente nuevo, contrato mes a mes, fibra óptica, sin servicios de protección."""
    return pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
    }])


@pytest.fixture
def cliente_bajo_riesgo():
    """Cliente antiguo con contrato de dos años, pago automático y servicios de soporte."""
    return pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 70.0,
        "TotalCharges": 4200.0,
    }])
