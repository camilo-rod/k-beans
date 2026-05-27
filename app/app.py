import streamlit as st
import sys
from pathlib import Path
from sklearn.decomposition import PCA

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent

sys.path.append(str(ROOT_DIR))  # para src/
sys.path.append(str(APP_DIR))   # para config y ui/

from config import APP_DIR, ROOT_DIR  # ahora sí funciona
from src.preprocessing import load_data, get_features
from src.predict import load_model, load_scaler
from UI import header, dashboard, tab_prediccion, tab_analisis, tab_info

# CONFIG
st.set_page_config(page_title="K-Beans", page_icon="🌱", layout="wide")

# CSS
with open(APP_DIR / "static" / "style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# CARGA DE MODELOS Y DATOS (una sola vez gracias a st.cache_resource)
@st.cache_resource
def load_resources():
    kmeans = load_model(ROOT_DIR / "models" / "kmeans_model.pkl")
    scaler = load_scaler(ROOT_DIR / "models" / "scaler.pkl")
    df     = load_data(ROOT_DIR / "data" / "processed" / "Dry_Bean_Dataset_clean.csv")
    return kmeans, scaler, df

kmeans, scaler, df = load_resources()

# Validación de columnas
features = get_features()
missing = [col for col in features if col not in df.columns]
if missing:
    st.error(f"Faltan columnas en el dataset: {missing}")
    st.stop()

# PCA global (se usa en tab_analisis y tab_prediccion)
X_scaled = scaler.transform(df[features])
pca      = PCA(n_components=2)
X_pca    = pca.fit_transform(X_scaled)

# RENDER
header.render()
dashboard.render(kmeans, df)

tab1, tab2, tab3 = st.tabs(["Predicción", "Análisis", "Información"])

with tab1:
    tab_prediccion.render(kmeans, scaler, pca)

with tab2:
    tab_analisis.render(kmeans, X_pca)

with tab3:
    tab_info.render()
