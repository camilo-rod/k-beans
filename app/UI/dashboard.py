import streamlit as st


def render(kmeans, df):
    """Renderiza las métricas del modelo y la descripción general."""
    st.subheader("Resumen del modelo")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", kmeans.n_clusters)
    c2.metric("Modelo", "K-Means")
    c3.metric("Registros", len(df))

    st.markdown(
        "K-Beans utiliza K-Means para detectar patrones geométricos y clasificar frijoles según sus características físicas."
    )