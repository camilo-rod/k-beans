# K-Beans

## Descripción
K-Beans es una aplicación de machine learning que segmenta tipos de frijoles secos en grupos homogéneos según sus características morfológicas (forma, tamaño y compactness), utilizando el algoritmo de clustering K-Means. El objetivo es automatizar la clasificación de variedades de frijoles, reduciendo la dependencia de clasificación manual.

## Demostración
[Enlace a la aplicación desplegada](https://k-beans-molzphfd6yfgwcrt3bzijh.streamlit.app/)

## Algoritmo utilizado
- **Nombre del algoritmo:** K-Means Clustering
- **Por qué es apropiado para este problema:** Es un algoritmo no supervisado ideal para segmentar datos morfológicos continuos sin etiquetas previas, agrupando frijoles similares en clusters según su distancia euclidiana.
- **Métricas de desempeño obtenidas:** Silhouette Score: 0.519

## Dataset
- **Fuente:** [Dry Bean Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Tamaño:** 13,611 muestras
- **Features utilizadas:**
  - `Area`: Área de la zona del frijol
  - `Perimeter`: Longitud del perímetro
  - `MajorAxisLength`: Longitud del eje mayor de la elipse equivalente
  - `MinorAxisLength`: Longitud del eje menor de la elipse equivalente
  - `Compactness`: Relación entre el área y el perímetro al cuadrado

## Instalación local

### Requisitos
- Python 3.11+
- pip

### Pasos
```bash
git clone https://github.com/camilo-rod/k-beans.git
cd k-beans
pip install -r requirements.txt
streamlit run app/app.py
```

### Ejecutar aplicación
```bash
streamlit run app/app.py
# o
python -m streamlit run app/app.py
```

## Uso
1. Ingresa los valores morfológicos del frijol (Area, Perimeter, MajorAxisLength, MinorAxisLength, Compactness) dentro de los rangos indicados.
2. Haz clic en **Predecir cluster**.
3. La app mostrará el cluster asignado, una descripción del perfil del frijol y una visualización PCA con todos los clusters.

## Estructura del proyecto
```
k-beans/
├── app/
│   ├── static/          # Estilos CSS
│   ├── templates/       # Header HTML
│   └── app.py           # Aplicación Streamlit
├── data/
│   ├── raw/             # Dataset original
│   └── processed/       # Dataset limpio
├── models/
│   ├── kmeans_model.pkl # Modelo entrenado
│   └── scaler.pkl       # Scaler serializado
├── notebook/
│   └── drybean.ipynb    # Entrenamiento y análisis
├── src/
│   ├── __init__.py
│   ├── preprocessing.py # Funciones de preprocesamiento
│   └── predict.py       # Funciones de predicción
├── .gitignore
├── README.md
└── requirements.txt
```

## Autores
- **Camilo Andrés Rodriguez Herrera** — Desarrollo del modelo y entrenamiento
- **Victor Daniel Oviedo Fino** — Desarrollo de la aplicación y despliegue

## Licencia
MIT License
