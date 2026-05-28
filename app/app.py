import streamlit as st
import sys
from pathlib import Path
from sklearn.decomposition import PCA

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent

sys.path.append(str(ROOT_DIR))
sys.path.append(str(APP_DIR))

from config import APP_DIR, ROOT_DIR
from src.preprocessing import load_data, get_features
from src.predict import load_model, load_scaler
from UI import header, dashboard, main_page

# CONFIG
st.set_page_config(page_title="K-Beans", page_icon=str(APP_DIR / "static" / "logo.png"), layout="wide")

# CSS
with open(APP_DIR / "static" / "style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    kmeans = load_model(ROOT_DIR / "models" / "kmeans_model.pkl")
    scaler = load_scaler(ROOT_DIR / "models" / "scaler.pkl")
    df     = load_data(ROOT_DIR / "data" / "processed" / "Dry_Bean_Dataset_clean.csv")
    return kmeans, scaler, df

kmeans, scaler, df = load_resources()

features = get_features()
missing = [col for col in features if col not in df.columns]
if missing:
    st.error(f"Faltan columnas en el dataset: {missing}")
    st.stop()

# PCA global
X_scaled = scaler.transform(df[features])
pca      = PCA(n_components=2)
X_pca    = pca.fit_transform(X_scaled)

# RENDER — una sola página, flujo lineal
header.render()
dashboard.render(kmeans, df)

st.markdown("---")

main_page.render(kmeans, scaler, pca, X_pca)