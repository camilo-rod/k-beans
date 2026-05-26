import streamlit as st


def render():
    """Renderiza el tab de información del proyecto."""
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
        st.write(
            "K-Beans es un sistema inteligente para clasificación automática de frijoles "
            "usando Machine Learning.\n\nAutor: Victor Daniel"
        )