import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.decomposition import PCA

import sys

# ======================================================
# PATHS
# ======================================================

APP_DIR = Path(__file__).parent

ROOT_DIR = APP_DIR.parent

sys.path.append(str(ROOT_DIR))

# ======================================================
# IMPORTS
# ======================================================

from src.preprocessing import load_data, get_features
from src.predict import load_model, load_scaler, predict_cluster

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(
    page_title="K-Beans AI",
    page_icon="🌱",
    layout="wide"
)

# ======================================================
# CSS
# ======================================================

css_path = APP_DIR / "static" / "style.css"

with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ======================================================
# MODELOS
# ======================================================

kmeans = load_model(
    ROOT_DIR / "models" / "kmeans_model.pkl"
)

scaler = load_scaler(
    ROOT_DIR / "models" / "scaler.pkl"
)

df = load_data(
    ROOT_DIR / "data" / "processed" / "Dry_Bean_Dataset_clean.csv"
)

features = get_features()

# ======================================================
# INFORMACIÓN FRIJOLES
# ======================================================

CLUSTER_INFO = {

    0: {
        "name": "Cali",
        "image": APP_DIR / "static" / "bean0.png",
        "description": "Frijol grande y claro con textura uniforme."
    },

    1: {
        "name": "Bombay",
        "image": APP_DIR / "static" / "bean1.png",
        "description": "Frijol compacto y de gran tamaño."
    },

    2: {
        "name": "Sira",
        "image": APP_DIR / "static" / "bean2.png",
        "description": "Frijol equilibrado y uniforme."
    },

    3: {
        "name": "Dermason",
        "image": APP_DIR / "static" / "bean3.png",
        "description": "Frijol pequeño y ligeramente alargado."
    },

    4: {
        "name": "Horoz",
        "image": APP_DIR / "static" / "bean4.png",
        "description": "Frijol robusto y denso."
    },

    5: {
        "name": "Barbunya",
        "image": APP_DIR / "static" / "bean5.png",
        "description": "Frijol irregular y ancho."
    },

    6: {
        "name": "Seker",
        "image": APP_DIR / "static" / "bean6.png",
        "description": "Frijol pequeño y compacto."
    }

}

# ======================================================
# HEADER
# ======================================================

st.markdown("""
<div class="hero">

<h1>K-Beans AI</h1>

<p>
Sistema inteligente de clasificación automática
de frijoles usando Machine Learning y análisis PCA.
</p>

</div>
""", unsafe_allow_html=True)

# ======================================================
# MÉTRICAS
# ======================================================

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Clusters", "7")

with m2:
    st.metric("Modelo", "K-Means")

with m3:
    st.metric("Registros", len(df))

# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3 = st.tabs([
    "Predicción",
    "Análisis PCA",
    "Información"
])

# ======================================================
# TAB 1
# ======================================================

with tab1:

    st.subheader("Características del Frijol")

    c1, c2 = st.columns(2)

    with c1:

        area = st.number_input(
            "Área",
            min_value=1000,
            value=32000
        )

        perimeter = st.number_input(
            "Perímetro",
            min_value=100,
            value=850
        )

        major_axis = st.number_input(
            "MajorAxisLength",
            min_value=50,
            value=300
        )

    with c2:

        minor_axis = st.number_input(
            "MinorAxisLength",
            min_value=20,
            value=150
        )

        compactness = st.slider(
            "Compactness",
            0.60,
            0.95,
            0.80
        )

    # ==================================================
    # BOTÓN
    # ==================================================

    if st.button("Analizar Frijol"):

        cluster, scaled = predict_cluster(
            kmeans,
            scaler,
            area,
            perimeter,
            major_axis,
            minor_axis,
            compactness
        )

        bean = CLUSTER_INFO.get(cluster)

        # ==============================================
        # SIMILITUD
        # ==============================================

        distance = np.min(
            kmeans.transform(scaled)
        )

        similarity = max(
            0,
            100 - (distance * 10)
        )

        # ==============================================
        # RESULTADO
        # ==============================================

        st.markdown(f"""
        <div class="result-card">

        <h2>{bean['name']}</h2>

        <p>{bean['description']}</p>

        </div>
        """, unsafe_allow_html=True)

        # ==============================================
        # IMAGEN
        # ==============================================

        if bean["image"].exists():

            st.image(
                str(bean["image"]),
                width=320
            )

        # ==============================================
        # BARRA
        # ==============================================

        st.subheader("Nivel de similitud")

        st.progress(
            int(similarity)
        )

        st.write(
            f"{similarity:.1f}% de similitud"
        )

# ======================================================
# TAB 2
# ======================================================

with tab2:

    X = df[features]

    X_scaled = scaler.transform(X)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    labels = kmeans.labels_

    pca_df = pd.DataFrame({

        "PCA1": X_pca[:, 0],
        "PCA2": X_pca[:, 1],
        "Cluster": labels.astype(str)

    })

    fig = px.scatter(

        pca_df,

        x="PCA1",
        y="PCA2",

        color="Cluster",

        title="Distribución PCA de Frijoles",

        template="plotly_dark",

        width=1200,
        height=700

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================================================
# TAB 3
# ======================================================

with tab3:

    st.markdown("""

## Dataset

Dry Bean Dataset

## Variables usadas

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- Compactness

## Algoritmo

K-Means Clustering

## Tecnologías

- Python
- Streamlit
- Scikit-Learn
- Plotly
- PCA

## Objetivo

Clasificar automáticamente frijoles
según características geométricas.

""")

# ======================================================
# FOOTER
# ======================================================

st.markdown("""

<br><br>

<center>

K-Beans AI • Machine Learning Project

</center>

""", unsafe_allow_html=True)
