import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import sys

# PATHS
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
sys.path.append(str(ROOT_DIR))

# IMPORTS
from src.preprocessing import load_data, get_features
from src.predict import load_model, load_scaler, predict_cluster

# CONFIG
st.set_page_config(page_title="K-Beans", page_icon="🌱", layout="wide")

# CSS
with open(APP_DIR / "static" / "style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# HEADER
import base64
with open(APP_DIR / "static" / "logo.png", "rb") as f:
    logo_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="kb-header">
  <div class="kb-bg-texture"></div>
  <div class="kb-header-content" style="display:flex; align-items:center; gap:24px;">
    <img src="{logo_b64}" width="90" style="border-radius:20px; background:white; padding:10px; box-shadow:0 8px 25px rgba(0,0,0,.15); flex-shrink:0;"/>
    <div>
      <h1 class="kb-title"><span class="kb-title-k">K</span><span class="kb-title-dash">—</span><span class="kb-title-beans">Beans</span></h1>
      <p class="kb-subtitle">Agrupación inteligente de frijoles mediante K-Means y análisis de componentes principales</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# MODELOS
kmeans = load_model(ROOT_DIR / "models" / "kmeans_model.pkl")
scaler = load_scaler(ROOT_DIR / "models" / "scaler.pkl")
df = load_data(ROOT_DIR / "data" / "processed" / "Dry_Bean_Dataset_clean.csv")
features = get_features()

# VALIDACIÓN
missing = [col for col in features if col not in df.columns]
if missing:
    st.error(f"Faltan columnas: {missing}")
    st.stop()

# PCA
X = df[features]
X_scaled = scaler.transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
labels = kmeans.labels_

# MAPEO
CLUSTER_NAMES = {0: "Cali", 1: "Bombay", 2: "Sira", 3: "Dermason", 4: "Horoz", 5: "Barbunya", 6: "Seker"}

# DESCRIPCIONES
BEAN_DESCRIPTIONS = {
    "Cali": "Frijol grande y claro con textura uniforme.",
    "Bombay": "Frijol robusto y compacto de gran tamaño.",
    "Sira": "Frijol equilibrado y uniforme.",
    "Dermason": "Frijol pequeño y ligeramente alargado.",
    "Horoz": "Frijol denso con estructura fuerte.",
    "Barbunya": "Frijol irregular y ancho.",
    "Seker": "Frijol pequeño y compacto."
}

# IMAGENES
BEAN_IMAGES = {i: APP_DIR / "static" / f"bean{i}.png" for i in range(7)}

# DASHBOARD
st.subheader("Resumen del modelo")
c1, c2, c3 = st.columns(3)
c1.metric("Clusters", kmeans.n_clusters)
c2.metric("Modelo", "K-Means")
c3.metric("Registros", len(df))
st.markdown("K-Beans utiliza Machine Learning para detectar patrones geométricos y clasificar frijoles según sus características físicas.")

# TABS
tab1, tab2, tab3 = st.tabs(["Predicción", "Análisis", "Información"])

# TAB 1
with tab1:
    st.markdown("""
### Analiza tu frijol
Ingresa las medidas físicas del frijol que deseas clasificar.
Puedes obtener estos valores con herramientas de análisis de imagen o medición digital.
""")

    st.info("""
**¿Cómo medir tu frijol?**
- **Área**: superficie total del frijol en píxeles (10 000 – 80 000)
- **Perímetro**: longitud del contorno del frijol (400 – 1 200)
- **MajorAxisLength**: largo del eje más largo del frijol (150 – 500)
- **MinorAxisLength**: largo del eje más corto del frijol (80 – 250)
- **Compactness**: qué tan redondo es el frijol, donde 1.0 es perfectamente redondo (0.60 – 0.95)
""")

    col1, col2 = st.columns(2)
    with col1:
        area = st.slider("Área (px²)", 10000, 80000, 42000, help="Superficie total del frijol")
        perimeter = st.slider("Perímetro (px)", 400, 1200, 700, help="Longitud del contorno")
        major_axis = st.slider("MajorAxisLength (px)", 150, 500, 250, help="Eje más largo del frijol")
    with col2:
        minor_axis = st.slider("MinorAxisLength (px)", 80, 250, 150, help="Eje más corto del frijol")
        compactness = st.slider("Compactness", 0.60, 0.95, 0.80, help="Qué tan redondo es el frijol")

    # BOTÓN
    if st.button("Predecir Frijol"):
        cluster, new_scaled = predict_cluster(kmeans, scaler, area, perimeter, major_axis, minor_axis, compactness)
        st.session_state["cluster"] = cluster
        st.session_state["new_pca"] = pca.transform(new_scaled)

        bean_name = CLUSTER_NAMES.get(cluster, "Desconocido")
        st.success(f"Frijol detectado: {bean_name}")

        # SCORE
        distance = np.min(kmeans.transform(new_scaled))
        score = max(0, 100 - (distance * 10))

        # IMAGEN + DESCRIPCIÓN
        image_path = BEAN_IMAGES.get(cluster)
        if image_path.exists():
            img_col, text_col = st.columns([1, 2])
            with img_col:
                st.image(str(image_path), width=260)
            with text_col:
                st.markdown(f"### {bean_name}\n\n{BEAN_DESCRIPTIONS.get(bean_name)}\n\n---\n\n### Nivel de similitud\n\n{score:.1f}%")

        # PROGRESS
        st.progress(int(score))


# TAB 2
with tab2:
    import seaborn as sns
    import pandas as pd

    PALETTE = ['#4a90c4', '#d85a30', '#3b9e75', '#e8a020', '#9b59b6', '#e74c3c', '#1abc9c']

    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': labels
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=pca_df,
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        palette=PALETTE[:kmeans.n_clusters],
        alpha=0.7,
        s=40,
        edgecolor='white',
        linewidth=0.3,
        ax=ax
    )

    if "new_pca" in st.session_state:
        point = st.session_state["new_pca"]
        ax.scatter(point[:, 0], point[:, 1], s=500, marker="*", color="red", label="Nuevo punto")

    ax.set_title('Clusters de frijoles', fontsize=13, pad=12)
    ax.set_xlabel('Componente principal 1', fontsize=11)
    ax.set_ylabel('Componente principal 2', fontsize=11)
    ax.grid(linestyle='--', linewidth=0.7, alpha=0.7, color='gray')
    ax.set_axisbelow(True)

    legend = ax.get_legend()
    legend.set_title('Cluster', prop={'size': 10})
    for text in legend.get_texts():
        text.set_fontsize(9)

    for spine in ax.spines.values():
        spine.set_edgecolor('#4a90c4')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    st.pyplot(fig)

# TAB 3
with tab3:
    st.write("""
### Dataset
Dry Bean Dataset

### Variables
- Area • Perimeter • MajorAxisLength • MinorAxisLength • Compactness

### Algoritmo
K-Means

### Preprocesamiento
StandardScaler
""")
    with st.expander("Información del proyecto"):
        st.write("K-Beans es un sistema inteligente para clasificación automática de frijoles usando Machine Learning.\n\nAutor: Victor Daniel")