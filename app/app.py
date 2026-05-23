import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title='K-Beans',
    page_icon='🌱',
    layout='wide'
)

# =========================
# TÍTULO
# =========================

st.title('Sistema de Clustering de Frijoles con K-Means')

st.markdown('''
Esta aplicación utiliza Machine Learning no supervisado
para agrupar frijoles según características físicas.
''')

# =========================
# CARGAR MODELOS
# =========================

kmeans = joblib.load('../models/kmeans_model.pkl')

scaler = joblib.load('../models/scaler.pkl')

# =========================
# CARGAR DATASET
# =========================

df = pd.read_csv('../data/raw/dry_bean_dataset.csv')

# =========================
# FEATURES
# =========================

features = [
    'Area',
    'Perimeter',
    'MajorAxisLength',
    'MinorAxisLength',
    'Compactness'
]

X = df[features]

# =========================
# ESCALAR DATOS
# =========================

X_scaled = scaler.transform(X)

# =========================
# PCA
# =========================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# =========================
# SIDEBAR
# =========================

st.sidebar.header('Ingresar Datos del Frijol')

area = st.sidebar.number_input('Area', value=25000.0)

perimeter = st.sidebar.number_input('Perimeter', value=700.0)

major_axis = st.sidebar.number_input('MajorAxisLength', value=250.0)

minor_axis = st.sidebar.number_input('MinorAxisLength', value=150.0)

aspect_ratio = st.sidebar.number_input('AspectRation', value=1.5)

eccentricity = st.sidebar.number_input('Eccentricity', value=0.7)

convex_area = st.sidebar.number_input('ConvexArea', value=26000.0)

equiv_diameter = st.sidebar.number_input('EquivDiameter', value=190.0)

extent = st.sidebar.number_input('Extent', value=0.75)

compactness = st.sidebar.number_input('Compactness', value=0.8)

# =========================
# NUEVO REGISTRO
# =========================

new_data = pd.DataFrame({
    'Area': [area],
    'Perimeter': [perimeter],
    'MajorAxisLength': [major_axis],
    'MinorAxisLength': [minor_axis],
    'AspectRation': [aspect_ratio],
    'Eccentricity': [eccentricity],
    'ConvexArea': [convex_area],
    'EquivDiameter': [equiv_diameter],
    'Extent': [extent],
    'Compactness': [compactness]
})

# =========================
# BOTÓN DE PREDICCIÓN
# =========================

if st.sidebar.button('Predecir Cluster'):

    # Escalar
    new_scaled = scaler.transform(new_data)

    # Predicción
    prediction = kmeans.predict(new_scaled)

    # PCA del nuevo punto
    new_pca = pca.transform(new_scaled)

    # =========================
    # RESULTADO
    # =========================

    st.subheader('Resultado de la Predicción')

    st.success(f'Cluster Predicho: {prediction[0]}')

    # =========================
    # INTERPRETACIÓN
    # =========================

    interpretations = {
        0: 'Frijoles grandes y compactos',
        1: 'Frijoles pequeños y alargados',
        2: 'Frijoles medianos con forma uniforme',
        3: 'Frijoles con geometría irregular',
        4: 'Frijoles densos y redondos',
        5: 'Grupo especializado de frijoles'
    }

    if prediction[0] in interpretations:

        st.info(interpretations[prediction[0]])

    # =========================
    # VISUALIZACIÓN PCA
    # =========================

    st.subheader('Visualización de Clusters')

    fig, ax = plt.subplots(figsize=(10,7))

    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=kmeans.labels_,
        alpha=0.5
    )

    # Nuevo punto
    ax.scatter(
        new_pca[:,0],
        new_pca[:,1],
        s=300,
        marker='X'
    )

    ax.set_title('Clusters con PCA')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')

    st.pyplot(fig)

# =========================
# INFORMACIÓN DEL MODELO
# =========================

st.subheader('Información del Modelo')

st.write(f'Número de clusters detectados: {kmeans.n_clusters}')

st.write('Modelo utilizado: K-Means Clustering')

st.write('Preprocesamiento: StandardScaler')