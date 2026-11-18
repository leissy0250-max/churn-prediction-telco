"""Reemplaza los enlaces de ejemplo por los reales del grupo.

Uso (desde la raíz del repositorio):
    python scripts/configurar_enlaces.py --usuario nombre-en-github
    python scripts/configurar_enlaces.py --usuario nombre-en-github --app https://mi-app.streamlit.app
"""
import argparse
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = ["README.md", "app.py", "docs/index.html", "notebooks/Ejecucion_en_Colab.ipynb"]
APP_EJEMPLO = "https://churn-telco-upao.streamlit.app"

parser = argparse.ArgumentParser()
parser.add_argument("--usuario", required=True, help="usuario u organización de GitHub")
parser.add_argument("--app", default=APP_EJEMPLO, help="URL real de la app en Streamlit")
args = parser.parse_args()

for nombre in ARCHIVOS:
    ruta = RAIZ / nombre
    texto = ruta.read_text(encoding="utf-8")
    nuevo = texto.replace("TU-USUARIO", args.usuario).replace(APP_EJEMPLO, args.app.rstrip("/"))
    ruta.write_text(nuevo, encoding="utf-8")
    print(f"{'actualizado' if nuevo != texto else 'sin cambios'}: {nombre}")
