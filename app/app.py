import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.decomposition import PCA

from src.preprocessing import load_data, get_features
from src.predict import load_model, load_scaler, predict_cluster

# =========================================
# CONFIG
# =========================================

st.set_page_config(
    page_title="K-Beans AI",
    page_icon="🌱",
    layout="wide"
)

# =========================================
# CSS
# =========================================

css_path = Path(__file__).parent / "static" / "style.css"

with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =========================================
# MODELOS
# =========================================

kmeans = load_model("models/kmeans_model.pkl")

scaler = load_scaler("models/scaler.pkl")

df = load_data("data/processed/Dry_Bean_Dataset_clean.csv")

features = get_features()

# =========================================
# MAPEO REAL
# =========================================

CLUSTER_INFO = {

    0: {
        "name": "Cali",
        "image": "static/bean0.png",
        "description": "Frijol grande, claro y de textura uniforme."
    },

    1: {
        "name": "Bombay",
        "image": "static/bean1.png",
        "description": "Frijol compacto y de gran tamaño."
    },

    2: {
        "name": "Sira",
        "image": "static/bean2.png",
        "description": "Frijol equilibrado y uniforme."
    },

    3: {
        "name": "Dermason",
        "image": "static/bean3.png",
        "description": "Frijol pequeño y ligeramente alargado."
    },

    4: {
        "name": "Horoz",
        "image": "static/bean4.png",
        "description": "Frijol denso y robusto."
    },

    5: {
        "name": "Barbunya",
        "image": "static/bean5.png",
        "description": "Frijol irregular con superficie amplia."
    },

    6: {
        "name": "Seker",
        "image": "static/bean6.png",
        "description": "Frijol pequeño y compacto."
    }

}

# =========================================
# HEADER
# =========================================

logo_path = Path(__file__).parent / "static" / "logo.png"

st.markdown("""
<div class="hero">

<div class="hero-text">

<h1>K-Beans AI</h1>

<p>
Sistema inteligente de clasificación automática
de frijoles mediante Machine Learning y análisis PCA
</p>

</div>

</div>
""", unsafe_allow_html=True)

# =========================================
# METRICAS
# =========================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Clusters", "7")

with c2:
    st.metric("Modelo", "K-Means")

with c3:
    st.metric("Registros", len(df))

# =========================================
# TABS
# =========================================

tab1, tab2, tab3 = st.tabs([
    "Predicción",
    "Análisis PCA",
    "Información"
])

# =========================================
# TAB 1
# =========================================

with tab1:

    st.subheader("Características del frijol")

    col1, col2 = st.columns(2)

    with col1:

        area = st.number_input(
            "Área",
            value=32000
        )

        perimeter = st.number_input(
            "Perímetro",
            value=850
        )

        major_axis = st.number_input(
            "Major Axis Length",
            value=300
        )

    with col2:

        minor_axis = st.number_input(
            "Minor Axis Length",
            value=150
        )

        compactness = st.slider(
            "Compactness",
            0.60,
            0.95,
            0.80
        )

    # =====================================
    # PREDICCION
    # =====================================

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

        bean = CLUSTER_INFO[cluster]

        # SCORE

        distance = np.min(
            kmeans.transform(scaled)
        )

        similarity = max(
            0,
            100 - (distance * 10)
        )

        # CARD

        st.markdown(f"""
        <div class="result-card">

        <h2>{bean['name']}</h2>

        <p>{bean['description']}</p>

        </div>
        """, unsafe_allow_html=True)

        # IMAGE

        st.image(
            bean["image"],
            width=320
        )

        # PROGRESS

        st.subheader("Nivel de similitud")

        st.progress(
            int(similarity)
        )

        st.write(
            f"{similarity:.1f}% de similitud"
        )

# =========================================
# TAB 2
# =========================================

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

# =========================================
# TAB 3
# =========================================

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

## Objetivo

Clasificar automáticamente frijoles
según características geométricas.

## Tecnologías

- Python
- Streamlit
- Scikit-Learn
- Plotly
- PCA
""")
