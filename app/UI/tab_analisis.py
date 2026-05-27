import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import PALETTE


def render(kmeans, X_pca):
    """Renderiza el tab de análisis con el gráfico PCA."""
    labels = kmeans.labels_

    pca_df = pd.DataFrame({
        'PCA1':    X_pca[:, 0],
        'PCA2':    X_pca[:, 1],
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

    # Punto del usuario si existe
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