import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import re
import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.preprocessing import load_data,get_features
from src.predict import load_model,load_scaler,predict_cluster


# CONFIGURACIÓN

st.set_page_config(
    page_title="K-Beans",
    page_icon="🌱",
    layout="wide"
)


# CSS

css_path=Path(__file__).parent/'static'/'style.css'

with open(css_path,'r',encoding='utf-8') as f:
    css_content=f.read()

st.markdown(
f"<style>{css_content}</style>",
unsafe_allow_html=True
)


# HEADER

header_path=Path(__file__).parent/'templates'/'header.html'

with open(header_path,'r',encoding='utf-8') as f:
    raw_html=f.read()

body_match=re.search(
r'<body>(.*?)</body>',
raw_html,
re.DOTALL
)

header_html=(
body_match.group(1).strip()
if body_match
else raw_html
)

st.markdown(
header_html,
unsafe_allow_html=True
)


# CARGAR MODELO

kmeans=load_model(
'models/kmeans_model.pkl'
)

scaler=load_scaler(
'models/scaler.pkl'
)

df=load_data(
'data/processed/Dry_Bean_Dataset_clean.csv'
)

features=get_features()


missing=[
col
for col in features
if col not in df.columns
]

if missing:

    st.error(
    f"Faltan columnas: {missing}"
    )

    st.stop()



# DATOS PCA

X=df[features]

X_scaled=scaler.transform(
X
)

pca=PCA(
n_components=2
)

X_pca=pca.fit_transform(
X_scaled
)

labels=kmeans.labels_


# DASHBOARD

st.subheader(
"Resumen del modelo"
)

c1,c2,c3=st.columns(3)

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


# TABS

tab1,tab2,tab3=st.tabs([

"Predicción",
"Análisis",
"Información"

])


# TAB PREDICCIÓN

with tab1:

    st.subheader(
    "Ingresar datos del frijol"
    )

    col1,col2=st.columns(2)

    with col1:

        area=st.slider(
        "Área",
        10000,
        80000,
        42000
        )

        perimeter=st.slider(
        "Perímetro",
        400,
        1200,
        700
        )

        major_axis=st.slider(
        "MajorAxisLength",
        150,
        500,
        250
        )


    with col2:

        minor_axis=st.slider(
        "MinorAxisLength",
        80,
        250,
        150
        )

        compactness=st.slider(
        "Compactness",
        0.60,
        0.95,
        0.80
        )


    if st.button(
    "Predecir Cluster"
    ):

        cluster,new_scaled=(

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


        new_pca=pca.transform(
        new_scaled
        )


        st.session_state[
        "new_pca"
        ]=new_pca


        st.subheader(
        "Resultado"
        )

        st.success(
        f"Cluster detectado: {cluster}"
        )


        interpretations={

        0:"Frijoles grandes y compactos",

        1:"Frijoles pequeños y alargados",

        2:"Frijoles medianos y uniformes",

        3:"Frijoles con geometría irregular",

        4:"Frijoles densos y redondos",

        5:"Grupo especializado",

        6:"Características mixtas"

        }

        st.info(

        interpretations.get(
        cluster,
        "Sin descripción"
        )

        )


        # SIMILITUD

        st.subheader(
        "Nivel de similitud"
        )

        distance=np.min(

        kmeans.transform(
        new_scaled
        )

        )

        score=max(
        0,
        100-(distance*10)
        )

        st.progress(
        int(score)
        )

        st.write(
        f"Similitud: {score:.1f}%"
        )


# IMAGEN

bean_images={

0: Path(__file__).parent/"static"/"bean0.png",
1: Path(__file__).parent/"static"/"bean1.png",
2: Path(__file__).parent/"static"/"bean2.png",
3: Path(__file__).parent/"static"/"bean3.png",
4: Path(__file__).parent/"static"/"bean4.png",
5: Path(__file__).parent/"static"/"bean5.png",
6: Path(__file__).parent/"static"/"bean6.png"

}

image_path=bean_images.get(cluster)

if image_path and image_path.exists():

    st.markdown(
    "<h3 style='text-align:center'>Tipo de frijol detectado</h3>",
    unsafe_allow_html=True
    )

    c1,c2,c3=st.columns([1,2,1])

    with c2:

        st.image(
        str(image_path),
        use_container_width=True
        )

else:

    st.warning(
    "No se encontró la imagen del frijol."
    )

# TAB ANÁLISIS

with tab2:

    st.subheader(
    "Visualización K-Means"
    )

    palette=[

'#4CAF50',  # verde hoja
'#FF9800',  # naranja cálido
'#00BCD4',  # turquesa
'#9C27B0',  # morado
'#E91E63',  # rosado
'#FFC107',  # amarillo
'#795548'   # café

]

    colors=palette[
    :kmeans.n_clusters
    ]

    # CENTROIDES PCA

    centroids_pca=np.array([

    X_pca[labels==i].mean(axis=0)

    for i in range(
    kmeans.n_clusters
    )

    ])


    fig,ax=plt.subplots(
figsize=(13,8)
)

fig.patch.set_facecolor('#f8fafc')

ax.set_facecolor('#ffffff')

ax.grid(
linestyle='--',
linewidth=.8,
alpha=.2
)


for i in range(kmeans.n_clusters):

    mask=labels==i

    ax.scatter(

    X_pca[mask,0],
    X_pca[mask,1],

    c=colors[i],
    alpha=.75,
    s=80,

    edgecolors='white',

    linewidths=1,

    label=f'Cluster {i}'
    )


ax.scatter(

centroids_pca[:,0],
centroids_pca[:,1],

s=500,

marker='o',

c=colors,

edgecolors='black',

linewidths=2,

zorder=5

)


for i,(cx,cy) in enumerate(
centroids_pca
):

    ax.text(

    cx,
    cy,

    str(i),

    fontsize=11,

    fontweight='bold',

    color='white',

    ha='center',

    va='center'

    )


if "new_pca" in st.session_state:

    point=st.session_state["new_pca"]

    ax.scatter(

    point[:,0],
    point[:,1],

    s=700,

    marker='*',

    color='#ff1744',

    edgecolors='black',

    linewidths=2,

    label='Nuevo frijol'

    )


ax.set_title(

"Mapa inteligente de clusters K-Beans",

fontsize=22,

fontweight='bold',

pad=20

)

ax.set_xlabel(
"Componente Principal 1"
)

ax.set_ylabel(
"Componente Principal 2"
)

legend=ax.legend(
fontsize=10,
frameon=True
)

legend.get_frame().set_alpha(.95)

for spine in ax.spines.values():

    spine.set_visible(False)

plt.tight_layout()

st.pyplot(fig)


# TAB INFORMACIÓN

with tab3:

    st.subheader(
    "Información del modelo"
    )

    st.write("""

### Dataset
Dry Bean Dataset

### Variables

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- Compactness

### Algoritmo

K-Means Clustering

### Preprocesamiento

StandardScaler

""")


with st.expander(
"Información del proyecto"
):

    st.write("""

Dataset: Dry Bean Dataset

Modelo: K-Means

Autor: Victor Daniel

K-Beans es un sistema de clasificación
y análisis de frijoles basado en
Machine Learning.

""")
