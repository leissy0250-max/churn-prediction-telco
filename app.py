"""App web del Sistema Predictivo de Abandono de Clientes (Churn Prediction).

Ejecución local:   streamlit run app.py
Despliegue:        Streamlit Community Cloud (archivo principal: app.py)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from churn import __version__, config, evaluacion, prediccion  # noqa: E402
from churn import modelo as mod  # noqa: E402

# Enlaces del proyecto (se actualizan con scripts/configurar_enlaces.py)
URL_REPOSITORIO = "https://github.com/TU-USUARIO/churn-prediction-telco"
URL_PAGINA = "https://TU-USUARIO.github.io/churn-prediction-telco/"

st.set_page_config(page_title="Churn Prediction | Telco", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.badge {display:inline-block;padding:0.35rem 0.9rem;border-radius:999px;font-weight:700;
        font-size:1.05rem;color:white;}
.alto {background:#dc2626;} .medio {background:#d97706;} .bajo {background:#16a34a;}
.caja {border:1px solid #e5e7eb;border-radius:12px;padding:1rem 1.2rem;background:#f8fafc;}
.pie {color:#6b7280;font-size:0.85rem;text-align:center;margin-top:2rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Recursos en caché (el modelo se carga una sola vez por sesión del servidor)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo...")
def obtener_modelo():
    return prediccion.cargar_modelo_seguro()


@st.cache_data
def obtener_ficha():
    return mod.cargar_ficha()


@st.cache_data
def leer_csv(ruta):
    return pd.read_csv(ruta)


modelo, origen_modelo = obtener_modelo()
ficha = obtener_ficha()
met = ficha["metricas_prueba"]
UMBRAL = modelo["umbral"]

# Opciones en español -> valores del dataset
SI_NO = {"Sí": "Yes", "No": "No"}
OPC = {
    "gender": {"Femenino": "Female", "Masculino": "Male"},
    "InternetService": {"Fibra óptica": "Fiber optic", "DSL": "DSL", "No tiene": "No"},
    "Contract": {"Mes a mes": "Month-to-month", "Un año": "One year", "Dos años": "Two year"},
    "PaymentMethod": {
        "Cheque electrónico": "Electronic check",
        "Cheque por correo": "Mailed check",
        "Transferencia bancaria (automática)": "Bank transfer (automatic)",
        "Tarjeta de crédito (automática)": "Credit card (automatic)",
    },
}
SERVICIOS_INTERNET = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
NOMBRES_CRITERIOS = {
    "roc_auc": "ROC-AUC", "recall": "Recall (sensibilidad)", "precision": "Precisión",
    "f1": "F1-Score", "accuracy": "Exactitud", "brier": "Brier score",
    "brecha_sobreajuste": "Brecha de sobreajuste (AUC entren. - AUC CV)",
}
COLUMNAS_RESULTADO = {
    "customerID": "ID cliente", "probabilidad_abandono": "Probabilidad de abandono",
    "nivel_riesgo": "Nivel de riesgo", "prediccion": "Predicción",
    "accion_recomendada": "Acción recomendada",
}


def badge(nivel: str) -> str:
    return f'<span class="badge {nivel.lower()}">Riesgo {nivel.upper()}</span>'


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Sistema Predictivo de Abandono")
    st.caption("Aprendizaje Estadístico · UPAO · 2026")
    st.markdown(
        f"""
**Modelo en producción**
- Algoritmo: Regresión Logística (C = {ficha['hiperparametros']['C']})
- Umbral de decisión: **{UMBRAL:.2f}**
- ROC-AUC en prueba: **{met['roc_auc']:.3f}**
- Versión del modelo: {__version__}

**Niveles de riesgo**
- Bajo: p < {config.CORTE_RIESGO_MEDIO:.2f}
- Medio: {config.CORTE_RIESGO_MEDIO:.2f} ≤ p < {config.CORTE_RIESGO_ALTO:.2f}
- Alto: p ≥ {config.CORTE_RIESGO_ALTO:.2f}
"""
    )
    st.markdown(f"[Repositorio en GitHub]({URL_REPOSITORIO})  \n[Página del proyecto]({URL_PAGINA})")
    if origen_modelo == "reentrenado":
        st.info("El modelo se reentrenó al iniciar la app (versión distinta de scikit-learn).")

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("Sistema Predictivo de Abandono de Clientes")
st.markdown("Predicción del abandono (*churn*) en empresas de telecomunicaciones mediante "
            "aprendizaje estadístico supervisado · Dataset *Telco Customer Churn*")

k1, k2, k3, k4 = st.columns(4)
k1.metric("ROC-AUC (prueba)", f"{met['roc_auc']:.3f}")
k2.metric("Recall: abandonos detectados", f"{met['recall']:.1%}")
k3.metric("Precisión de las alertas", f"{met['precision']:.1%}")
k4.metric("Lift frente a contacto al azar", f"{met['precision'] / 0.2658:.2f}x")

tab1, tab2, tab3, tab4 = st.tabs([
    "Predicción individual", "Predicción por lote (CSV)",
    "Desempeño del modelo", "Acerca del proyecto",
])

# ---------------------------------------------------------------------------
# Pestaña 1: predicción individual
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Evaluar el riesgo de abandono de un cliente")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Perfil del cliente**")
        genero = st.selectbox("Género", list(OPC["gender"]))
        mayor = st.radio("Adulto mayor (65 años o más)", ["No", "Sí"], horizontal=True)
        pareja = st.radio("Tiene pareja", ["No", "Sí"], horizontal=True)
        dependientes = st.radio("Tiene dependientes", ["No", "Sí"], horizontal=True)
        antiguedad = st.slider("Antigüedad (meses)", 0, 72, 2)

    with c2:
        st.markdown("**Servicios contratados**")
        telefono = st.radio("Servicio telefónico", ["Sí", "No"], horizontal=True)
        lineas = st.radio("Líneas múltiples", ["No", "Sí"], horizontal=True,
                          disabled=(telefono == "No"))
        internet = st.selectbox("Servicio de internet", list(OPC["InternetService"]))
        sin_internet = internet == "No tiene"
        nombres_servicios = {
            "Seguridad en línea": "OnlineSecurity", "Respaldo en línea": "OnlineBackup",
            "Protección de dispositivos": "DeviceProtection", "Soporte técnico": "TechSupport",
            "TV por streaming": "StreamingTV", "Películas por streaming": "StreamingMovies",
        }
        elegidos = st.multiselect(
            "Servicios adicionales de internet", list(nombres_servicios),
            default=["TV por streaming", "Películas por streaming"],
            disabled=sin_internet,
            help="Solo aplica si el cliente tiene servicio de internet.")
        servicios = {v: ("Sí" if k in elegidos else "No") for k, v in nombres_servicios.items()}

    with c3:
        st.markdown("**Contrato y facturación**")
        contrato = st.selectbox("Tipo de contrato", list(OPC["Contract"]))
        electronica = st.radio("Facturación electrónica", ["Sí", "No"], horizontal=True)
        pago = st.selectbox("Método de pago", list(OPC["PaymentMethod"]))
        cargo_mensual = st.number_input("Cargo mensual (USD)", 0.0, 200.0, 95.0, step=0.5)
        estimar = st.checkbox("Estimar cargo total = antigüedad x cargo mensual", value=True)
        cargo_total = st.number_input("Cargo total (USD)", 0.0, 10000.0,
                                      float(antiguedad * cargo_mensual), step=10.0,
                                      disabled=estimar)
        if estimar:
            cargo_total = float(antiguedad * cargo_mensual)

    registro = pd.DataFrame([{
        "gender": OPC["gender"][genero],
        "SeniorCitizen": 1 if mayor == "Sí" else 0,
        "Partner": SI_NO[pareja],
        "Dependents": SI_NO[dependientes],
        "tenure": antiguedad,
        "PhoneService": SI_NO[telefono],
        "MultipleLines": "No phone service" if telefono == "No" else SI_NO[lineas],
        "InternetService": OPC["InternetService"][internet],
        **{v: ("No internet service" if sin_internet else SI_NO[servicios[v]])
           for v in SERVICIOS_INTERNET},
        "Contract": OPC["Contract"][contrato],
        "PaperlessBilling": SI_NO[electronica],
        "PaymentMethod": OPC["PaymentMethod"][pago],
        "MonthlyCharges": cargo_mensual,
        "TotalCharges": cargo_total,
    }])

    if st.button("Predecir riesgo de abandono", type="primary", width="stretch"):
        try:
            r = prediccion.predecir(modelo, registro).iloc[0]
            grupos = prediccion.explicar_por_grupo(modelo, registro)
            detalle = prediccion.explicar(modelo, registro)
        except prediccion.ErrorDeEntrada as e:
            st.error(str(e))
        else:
            st.divider()
            izq, der = st.columns([1, 1.3])
            with izq:
                st.markdown("#### Resultado")
                st.metric("Probabilidad de abandono", f"{r['probabilidad_abandono']:.1%}")
                st.progress(float(r["probabilidad_abandono"]))
                st.markdown(badge(r["nivel_riesgo"]), unsafe_allow_html=True)
                st.markdown(f"**Predicción:** {r['prediccion']} (umbral {UMBRAL:.2f})  \n"
                            f"**Acción recomendada:** {r['accion_recomendada']}")
            with der:
                st.markdown("#### ¿Qué explica este resultado?")
                fig, ax = plt.subplots(figsize=(6, 2.6))
                g = grupos.sort_values("aporte")
                colores = ["#dc2626" if a > 0 else "#16a34a" for a in g["aporte"]]
                ax.barh(g["grupo"], g["aporte"], color=colores)
                ax.axvline(0, color="#374151", lw=1)
                ax.set_xlabel("Aporte al log-odds frente al cliente promedio\n"
                              "(rojo: aumenta el riesgo | verde: lo reduce)", fontsize=8)
                ax.tick_params(labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                with st.expander("Detalle por variable"):
                    st.caption("Las variables correlacionadas (p. ej. cargo mensual y servicios) "
                               "deben leerse en conjunto; por eso el gráfico agrupa sus aportes.")
                    detalle["valor"] = detalle["valor"].astype(str)
                    st.dataframe(
                        detalle[["variable_es", "valor", "aporte"]].rename(columns={
                            "variable_es": "Variable", "valor": "Valor", "aporte": "Aporte"}),
                        hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Pestaña 2: predicción por lote
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Generar alertas para una lista de clientes")
    st.markdown("Suba un archivo CSV con las 19 variables del dataset (mismos nombres de "
                "columna). Si incluye la columna `Churn`, la app también evalúa el modelo.")

    ejemplos = {
        "10 clientes nuevos (sin etiqueta)": config.DIR_EJEMPLOS / "clientes_nuevos_ejemplo.csv",
        "1,407 clientes del conjunto de prueba (con etiqueta real)":
            config.DIR_EJEMPLOS / "conjunto_prueba_etiquetado.csv",
    }
    a, b = st.columns([1.4, 1])
    with a:
        archivo = st.file_uploader("Archivo CSV de clientes", type=["csv"])
    with b:
        eleccion = st.selectbox("...o use un archivo de ejemplo", ["(ninguno)"] + list(ejemplos))
        plantilla = leer_csv(ejemplos["10 clientes nuevos (sin etiqueta)"]).head(1)
        st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False),
                           "plantilla_clientes.csv", "text/csv")

    datos = None
    if archivo is not None:
        datos = pd.read_csv(archivo)
    elif eleccion != "(ninguno)":
        datos = leer_csv(ejemplos[eleccion])

    if datos is not None:
        try:
            res = prediccion.predecir(modelo, datos)
        except prediccion.ErrorDeEntrada as e:
            st.error(str(e))
        else:
            n_alto = int((res["nivel_riesgo"] == "Alto").sum())
            n_medio = int((res["nivel_riesgo"] == "Medio").sum())
            n_contactar = int((res["prediccion"] == "Abandona").sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clientes evaluados", f"{len(res):,}")
            m2.metric("Riesgo alto", f"{n_alto:,}")
            m3.metric("Riesgo medio", f"{n_medio:,}")
            m4.metric("A contactar (p ≥ umbral)", f"{n_contactar:,}")

            ordenado = res.sort_values("probabilidad_abandono", ascending=False)
            st.dataframe(ordenado.rename(columns=COLUMNAS_RESULTADO), hide_index=True,
                         width="stretch", height=320,
                         column_config={"Probabilidad de abandono": st.column_config.ProgressColumn(
                             "Probabilidad de abandono", min_value=0.0, max_value=1.0,
                             format="%.3f")})
            st.download_button("Descargar resultados (CSV)", ordenado.to_csv(index=False),
                               "alertas_abandono.csv", "text/csv", type="primary")

            if config.COL_OBJETIVO in datos.columns:
                st.markdown("#### Validación con la etiqueta real")
                y_real = (datos[config.COL_OBJETIVO].astype(str).str.strip().str.lower()
                          == "yes").astype(int)
                mv = evaluacion.calcular_metricas(y_real, res["probabilidad_abandono"], UMBRAL)
                v1, v2, v3, v4, v5 = st.columns(5)
                v1.metric("ROC-AUC", f"{mv['roc_auc']:.3f}")
                v2.metric("Exactitud", f"{mv['accuracy']:.3f}")
                v3.metric("Precisión", f"{mv['precision']:.3f}")
                v4.metric("Recall", f"{mv['recall']:.3f}")
                v5.metric("F1-Score", f"{mv['f1']:.3f}")
                mc = mv["matriz_confusion"]
                st.dataframe(pd.DataFrame(
                    {"Predicho: no abandona": [mc["VN"], mc["FN"]],
                     "Predicho: abandona": [mc["FP"], mc["VP"]]},
                    index=["Real: no abandona", "Real: abandona"]),
                    width="stretch")

# ---------------------------------------------------------------------------
# Pestaña 3: desempeño del modelo
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Desempeño del modelo en el conjunto de prueba (1,407 clientes)")
    criterios = pd.DataFrame([
        {"Criterio": NOMBRES_CRITERIOS.get(k, k), "Meta": v["meta"], "Valor": v["valor"],
         "Cumple": "Sí" if v["cumple"] else "No"}
        for k, v in ficha["criterios_aceptacion"].items()])
    st.dataframe(criterios, hide_index=True, width="stretch")
    f1c, f2c = st.columns(2)
    f1c.image(str(config.DIR_FIGURAS / "matriz_confusion.png"), width="stretch")
    f2c.image(str(config.DIR_FIGURAS / "curva_roc.png"), width="stretch")
    f3c, f4c = st.columns([1.4, 1])
    f3c.image(str(config.DIR_FIGURAS / "razones_momios.png"), width="stretch")
    f4c.markdown(
        "**Lectura de las razones de momios**\n\n"
        "- Una razón > 1 indica que la categoría **aumenta** la probabilidad de abandono "
        "frente a su categoría de referencia; una razón < 1 la **reduce**.\n"
        "- La fibra óptica (≈ 3.3) y el contrato mes a mes (≈ 1.8) son los factores de "
        "mayor riesgo; el contrato de dos años (≈ 0.44) es el principal factor protector.\n"
        "- El cargo mensual, el cargo total y la antigüedad no se muestran porque están "
        "muy correlacionados y sus coeficientes no deben leerse por separado.")
    with st.expander("Ficha técnica del modelo (models/ficha_modelo.json)"):
        st.json(ficha)

# ---------------------------------------------------------------------------
# Pestaña 4: acerca del proyecto
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Acerca del proyecto")
    st.markdown(f"""
**Proyecto:** Sistema Predictivo de Abandono de Clientes (Churn Prediction) mediante
Aprendizaje Estadístico Supervisado en Empresas de Telecomunicaciones.

**Curso:** Aprendizaje Estadístico · Programa de Ingeniería de Sistemas e Inteligencia
Artificial · Universidad Privada Antenor Orrego (Trujillo, 2026).

**Autores:** Ruiz Cueva, Johan · Chavez Sampen, Luis · Lopez Fasabi, Elisson ·
Polo Valderrama, Benjamin · Cardenas Julian, Piero · Sanchez Velasquez, Gabriela Ximena.

**Docente:** Sagastegui Chigne, Teobaldo Herman.

**Datos:** Telco Customer Churn (IBM), 7,043 clientes y 21 variables; 7,032 tras la limpieza.

**Modelo:** Regresión Logística seleccionada entre cuatro modelos (Regresión Logística,
Regresión Logística + PCA, Random Forest y XGBoost) por cumplir los siete criterios de
aceptación y por su interpretabilidad.

**Uso responsable:** el dataset es un conjunto de ejemplo; antes de usar el sistema con
clientes reales debe recalibrarse con datos de la operadora y revisarse por subgrupos.
La app no almacena los datos que se ingresan: se procesan en memoria.

[Repositorio en GitHub]({URL_REPOSITORIO}) · [Página del proyecto]({URL_PAGINA})
""")

st.markdown('<div class="pie">Universidad Privada Antenor Orrego · Aprendizaje '
            'Estadístico · 2026</div>', unsafe_allow_html=True)
