import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import sys
import os

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
    page_title="K-Beans",
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
# LOGO + HEADER
# ======================================================

logo_path = APP_DIR / "static" / "logo.png"

col1, col2 = st.columns([1, 6])

with col1:
    st.image(str(logo_path), width=120)

with col2:

    st.markdown("""
    <div class="kb-header">

    <div class="kb-header-content">

    <h1 class="kb-title">
    <span class="kb-title-k">K</span>
    <span class="kb-title-dash">—</span>
    <span class="kb-title-beans">Beans</span>
    </h1>

    <p class="kb-subtitle">
    Agrupación inteligente de frijoles mediante
    K-Means y análisis de componentes principales
    </p>

    </div>

    </div>
    """, unsafe_allow_html=True)

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
# VALIDACIÓN
# ======================================================

missing = [

    col

    for col in features

    if col not in df.columns

]

if missing:

    st.error(
        f"Faltan columnas: {missing}"
    )

    st.stop()

# ======================================================
# PCA
# ======================================================

X = df[features]

X_scaled = scaler.transform(X)

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

labels = kmeans.labels_

# ======================================================
# DASHBOARD
# ======================================================

st.subheader(
    "Resumen del modelo"
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Clusters",
        kmeans.n_clusters
    )

with c2:

    st.metric(
        "Modelo",
        "K-Means"
    )

with c3:

    st.metric(
        "Registros",
        len(df)
    )

st.markdown("""

K-Beans utiliza Machine Learning
para detectar patrones geométricos
y clasificar frijoles según sus
características físicas.

""")

# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3 = st.tabs(

    [
        "Predicción",
        "Análisis",
        "Información"
    ]

)

# ======================================================
# TAB 1
# ======================================================

with tab1:

    st.subheader(
        "Ingresar datos"
    )

    col1, col2 = st.columns(2)

    with col1:

        area = st.slider(
            "Área",
            10000,
            80000,
            42000
        )

        perimeter = st.slider(
            "Perímetro",
            400,
            1200,
            700
        )

        major_axis = st.slider(
            "MajorAxisLength",
            150,
            500,
            250
        )

    with col2:

        minor_axis = st.slider(
            "MinorAxisLength",
            80,
            250,
            150
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

    if st.button(
        "Predecir Cluster"
    ):

        cluster, new_scaled = (

            predict_cluster(

                kmeans,
                scaler,

                area,
                perimeter,
                major_axis,
                minor_axis,
                compactness

            )
        )

        st.session_state["cluster"] = cluster

        new_pca = pca.transform(
            new_scaled
        )

        st.session_state[
            "new_pca"
        ] = new_pca

        st.success(
            f"Cluster detectado: {cluster}"
        )

        interpretations = {

            0: "Frijoles grandes y compactos",
            1: "Frijoles pequeños y alargados",
            2: "Frijoles medianos y uniformes",
            3: "Geometría irregular",
            4: "Frijoles densos",
            5: "Grupo especializado",
            6: "Características mixtas"

        }

        st.info(
            interpretations.get(
                cluster
            )
        )

        # ==================================================
        # SIMILITUD
        # ==================================================

        distance = np.min(

            kmeans.transform(
                new_scaled
            )

        )

        score = max(
            0,
            100 - (distance * 10)
        )

        st.subheader(
            "Nivel de similitud"
        )

        st.progress(
            int(score)
        )

        st.write(
            f"{score:.1f}%"
        )

        # ==================================================
        # IMAGENES
        # ==================================================

        bean_images = {

            0: APP_DIR / "static" / "bean0.png",
            1: APP_DIR / "static" / "bean1.png",
            2: APP_DIR / "static" / "bean2.png",
            3: APP_DIR / "static" / "bean3.png",
            4: APP_DIR / "static" / "bean4.png",
            5: APP_DIR / "static" / "bean5.png",
            6: APP_DIR / "static" / "bean6.png"

        }

        image_path = bean_images.get(
            cluster
        )

        if image_path.exists():

            st.image(
                str(image_path),
                width=300
            )

# ======================================================
# TAB 2
# ======================================================

with tab2:

    palette = [

        '#4CAF50',
        '#FF9800',
        '#00BCD4',
        '#9C27B0',
        '#E91E63',
        '#FFC107',
        '#795548'

    ]

    colors = palette[
        :kmeans.n_clusters
    ]

    centroids_pca = np.array([

        X_pca[
            labels == i
        ].mean(axis=0)

        for i in range(
            kmeans.n_clusters
        )

    ])

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    ax.set_facecolor(
        "#ffffff"
    )

    for i in range(
        kmeans.n_clusters
    ):

        mask = labels == i

        ax.scatter(

            X_pca[mask, 0],
            X_pca[mask, 1],

            c=colors[i],

            s=80,

            alpha=.7,

            label=f"Cluster {i}"

        )

    if "new_pca" in st.session_state:

        point = st.session_state[
            "new_pca"
        ]

        ax.scatter(

            point[:, 0],
            point[:, 1],

            s=500,

            marker="*",

            color="red",

            label="Nuevo"

        )

    ax.legend()

    st.pyplot(
        fig
    )

# ======================================================
# TAB 3
# ======================================================

with tab3:

    st.write("""

### Dataset

Dry Bean Dataset

### Variables

• Area  
• Perimeter  
• MajorAxisLength  
• MinorAxisLength  
• Compactness  

### Algoritmo

K-Means

### Preprocesamiento

StandardScaler

""")

with st.expander(
    "Información del proyecto"
):

    st.write("""

K-Beans es un sistema inteligente
para clasificación automática
de frijoles usando Machine Learning.

Autor:
Victor Daniel

""")
