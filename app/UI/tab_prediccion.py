import streamlit as st
import numpy as np

from src.predict import predict_cluster
from config import CLUSTER_NAMES, BEAN_DESCRIPTIONS, BEAN_IMAGES, SLIDER_CONFIG

def render(kmeans, scaler, pca):
    """Renderiza el tab de predicción."""

    st.markdown("""
### Analiza tu frijol
Ingresa las medidas físicas del frijol que deseas clasificar.
Puedes obtener estos valores con herramientas de análisis de imagen o medición digital.
""")

    st.info("""
**¿Cómo medir tu frijol?**
- **Área**: superficie total del frijol en píxeles (10 000 – 200 000)
- **Perímetro**: longitud del contorno del frijol (400 – 1 700)
- **MajorAxisLength**: largo del eje más largo del frijol (150 – 650)
- **MinorAxisLength**: largo del eje más corto del frijol (80 – 400)
- **Compactness**: qué tan redondo es el frijol, donde 1.0 es perfectamente redondo (0.60 – 0.95)
""")

    col1, col2 = st.columns(2)

    with col1:
        area = st.slider(
            "Área (px²)", *SLIDER_CONFIG["area"],
            help="Superficie total del frijol"
        )
        perimeter = st.slider(
            "Perímetro (px)", *SLIDER_CONFIG["perimeter"],
            help="Longitud del contorno"
        )
        major_axis = st.slider(
            "MajorAxisLength (px)", *SLIDER_CONFIG["major_axis"],
            help="Eje más largo del frijol"
        )

    with col2:
        minor_axis = st.slider(
            "MinorAxisLength (px)", *SLIDER_CONFIG["minor_axis"],
            help="Eje más corto del frijol"
        )
        compactness = st.slider(
            "Compactness", *SLIDER_CONFIG["compactness"],
            help="Qué tan redondo es el frijol"
        )

    if st.button("Predecir Frijol"):
        _mostrar_resultado(kmeans, scaler, pca, area, perimeter, major_axis, minor_axis, compactness)


def _mostrar_resultado(kmeans, scaler, pca, area, perimeter, major_axis, minor_axis, compactness):
    """Calcula y muestra el resultado de la predicción."""
    cluster, new_scaled = predict_cluster(kmeans, scaler, area, perimeter, major_axis, minor_axis, compactness)

    # Guardar en session_state para el gráfico PCA del tab2
    st.session_state["cluster"] = cluster
    st.session_state["new_pca"] = pca.transform(new_scaled)

    bean_name = CLUSTER_NAMES.get(cluster, "Desconocido")
    st.success(f"Frijol detectado: {bean_name}")

    # Score de similitud
    distance = np.min(kmeans.transform(new_scaled))
    score = max(0, 100 - (distance * 10))

    # Imagen + descripción
    image_path = BEAN_IMAGES.get(bean_name)
    if image_path and image_path.exists():
        img_col, text_col = st.columns([1, 2])
        with img_col:
            st.image(str(image_path), width=260)
        with text_col:
            st.markdown(
                f"### {bean_name}\n\n"
                f"{BEAN_DESCRIPTIONS.get(bean_name)}\n\n"
                f"---\n\n"
                f"### Nivel de similitud\n\n{score:.1f}%"
            )

    st.progress(int(score))