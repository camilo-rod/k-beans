import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.decomposition import PCA
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import load_data, get_features
from src.predict import load_model, load_scaler, predict_cluster

# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title='K-Beans',
    page_icon='🌱',
    layout='wide'
)

# =========================
# INYECTAR CSS PERSONALIZADO
# =========================

css_path = Path(__file__).parent / 'static' / 'style.css'
with open(css_path, 'r', encoding='utf-8') as f:
    css_content = f.read()

st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

# =========================
# INYECTAR HEADER HTML
# =========================

header_path = Path(__file__).parent / 'templates' / 'header.html'
with open(header_path, 'r', encoding='utf-8') as f:
    raw_html = f.read()

body_match = re.search(r'<body>(.*?)</body>', raw_html, re.DOTALL)
header_html = body_match.group(1).strip() if body_match else raw_html

st.markdown(header_html, unsafe_allow_html=True)

# =========================
# CARGAR MODELO, SCALER Y DATOS
# =========================

kmeans = load_model('models/kmeans_model.pkl')
scaler = load_scaler('models/scaler.pkl')

df = load_data('data/processed/Dry_Bean_Dataset_clean.csv')
features = get_features()

# =========================
# VALIDAR FEATURES
# =========================

missing = [col for col in features if col not in df.columns]

if len(missing) > 0:
    st.error(f'Columnas faltantes: {missing}')
    st.stop()

# =========================
# DATOS PARA VISUALIZACIÓN
# =========================

X = df[features].copy()
X_scaled = scaler.transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================
# INPUTS
# =========================

st.subheader('Ingresar datos del frijol')

st.markdown('''
Ingresa los valores dentro de los rangos permitidos:

- **Area**: 10 000 – 80 000
- **Perimeter**: 400 – 1 200
- **MajorAxisLength**: 150 – 500
- **MinorAxisLength**: 80 – 250
- **Compactness**: 0.60 – 0.95
''')

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        'Area',
        min_value=10000.0, max_value=80000.0,
        value=42000.0, step=100.0
    )
    perimeter = st.number_input(
        'Perimeter',
        min_value=400.0, max_value=1200.0,
        value=700.0, step=10.0
    )
    major_axis = st.number_input(
        'MajorAxisLength',
        min_value=150.0, max_value=500.0,
        value=250.0, step=5.0
    )

with col2:
    minor_axis = st.number_input(
        'MinorAxisLength',
        min_value=80.0, max_value=250.0,
        value=150.0, step=5.0
    )
    compactness = st.number_input(
        'Compactness',
        min_value=0.60, max_value=0.95,
        value=0.80, step=0.01,
        format="%.2f"
    )

# =========================
# VALIDACIÓN
# =========================

valid = True

if area < 10000 or area > 80000:
    st.error('Digite un valor válido para Area.')
    valid = False

if perimeter < 400 or perimeter > 1200:
    st.error('Digite un valor válido para Perimeter.')
    valid = False

if major_axis < 150 or major_axis > 500:
    st.error('Digite un valor válido para MajorAxisLength.')
    valid = False

if minor_axis < 80 or minor_axis > 250:
    st.error('Digite un valor válido para MinorAxisLength.')
    valid = False

if compactness < 0.60 or compactness > 0.95:
    st.error('Digite un valor válido para Compactness.')
    valid = False

# =========================
# BOTÓN DE PREDICCIÓN
# =========================

if st.button('Predecir cluster'):

    if valid:

        # Predicción usando src/predict.py
        cluster, new_scaled = predict_cluster(
            kmeans, scaler,
            area, perimeter, major_axis, minor_axis, compactness
        )

        # PCA nuevo punto
        new_pca = pca.transform(new_scaled)

        # =========================
        # RESULTADO
        # =========================

        st.subheader('Resultado de la predicción')
        st.success(f'  Cluster predicho: **{cluster}**')

        # =========================
        # INTERPRETACIÓN
        # =========================

        interpretations = {
            0: 'Frijoles grandes y compactos',
            1: 'Frijoles pequeños y alargados',
            2: 'Frijoles medianos y uniformes',
            3: 'Frijoles con geometría irregular',
            4: 'Frijoles densos y redondos',
            5: 'Grupo especializado de frijoles',
            6: 'Frijoles con características mixtas'
        }

        if cluster in interpretations:
            st.info(f'  {interpretations[cluster]}')

        # =========================
        # VISUALIZACIÓN
        # =========================

        st.subheader('Visualización de clusters')

        SKLEARN_COLORS = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2',
        ]

        n_clusters  = kmeans.n_clusters
        colors      = SKLEARN_COLORS[:n_clusters]
        labels      = kmeans.labels_

        centroids_pca = np.array([
            X_pca[labels == i].mean(axis=0)
            for i in range(n_clusters)
        ])

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#FDFAF5')
        ax.set_facecolor('#FDFAF5')

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='#C4B49A')
        ax.set_axisbelow(True)

        for i in range(n_clusters):
            mask = labels == i
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], alpha=0.35, s=18,
                edgecolors='none', label=f'Cluster {i}'
            )

        ax.scatter(
            centroids_pca[:, 0], centroids_pca[:, 1],
            s=220, marker='o',
            c=[colors[i] for i in range(n_clusters)],
            edgecolors='#3B2A1A', linewidths=1.8,
            zorder=5, label='Centroides'
        )

        for i, (cx, cy) in enumerate(centroids_pca):
            ax.text(
                cx, cy, str(i),
                fontsize=9, fontweight='bold',
                ha='center', va='center',
                color='white', zorder=6
            )

        ax.scatter(
            new_pca[:, 0], new_pca[:, 1],
            s=280, marker='*', color='#C4714A',
            edgecolors='white', linewidths=0.8,
            zorder=7, label='Nuevo frijol'
        )

        ax.set_title(
            'K-Means Clustering — Proyección PCA (2 componentes)',
            fontsize=13, fontweight='bold', pad=14, color='#3B2A1A'
        )
        ax.set_xlabel('Componente principal 1', fontsize=11, color='#5A4A38', labelpad=8)
        ax.set_ylabel('Componente principal 2', fontsize=11, color='#5A4A38', labelpad=8)

        ax.legend(
            loc='upper right', fontsize=9,
            framealpha=0.9, edgecolor='#E0D5C5', fancybox=False
        )

        for spine in ax.spines.values():
            spine.set_edgecolor('#E0D5C5')
            spine.set_linewidth(0.8)

        ax.tick_params(colors='#5A4A38', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)

# =========================
# INFORMACIÓN DEL MODELO
# =========================

st.subheader('Información del modelo')
st.write(f'**Número de clusters:** {kmeans.n_clusters}')
st.write('**Modelo:** K-Means Clustering')
st.write('**Preprocesamiento:** StandardScaler')