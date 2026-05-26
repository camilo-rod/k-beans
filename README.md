# K-Beans

## Descripción
K-Beans es una aplicación de machine learning que segmenta tipos de frijoles secos en grupos homogéneos según sus características morfológicas (forma, tamaño y compactness), utilizando el algoritmo de clustering K-Means. El objetivo es automatizar la clasificación de variedades de frijoles, reduciendo la dependencia de clasificación manual.

## Demostración
[Enlace a la aplicación desplegada](https://k-beans-molzphfd6yfgwcrt3bzijh.streamlit.app/)

## Motivación
La clasificación manual de variedades de frijoles es un proceso lento, costoso y propenso a errores humanos, especialmente en contextos agroindustriales donde se manejan grandes volúmenes. Este proyecto nació del interés por aplicar técnicas de machine learning a un problema concreto del agro, explorando cómo características morfológicas medibles (como el área, el perímetro y la compactness) pueden ser suficientes para agrupar variedades de forma automatizada, sin necesidad de etiquetas previas.

También fue una oportunidad para integrar conocimientos de preprocesamiento de datos, reducción de dimensionalidad (PCA) y despliegue de modelos en una aplicación web funcional.

## Industrias de aplicación
El enfoque de este proyecto es aplicable en múltiples sectores:

- **Agroindustria y procesamiento de alimentos:** clasificación automatizada de granos, semillas y legumbres en líneas de producción.
- **Control de calidad:** detección de productos fuera de estándar morfológico en plantas de empaque.
- **Investigación agrícola:** caracterización de variedades de cultivos para estudios de biodiversidad o mejoramiento genético.
- **Retail y comercio:** categorización de productos agrícolas para optimizar inventarios o precios diferenciados por variedad.
- **Logística agrícola:** separación automatizada de lotes según características físicas antes del almacenamiento o exportación.

## Proyectos similares
Este tipo de solución no es nueva; otros equipos e instituciones han abordado problemas parecidos:

- **UCI ML Repository — Dry Bean Dataset (Koklu & Ozkan, 2020):** el dataset que usamos proviene de un estudio original donde se aplicaron algoritmos de clasificación supervisada (SVM, k-NN, MLP) para identificar 7 variedades de frijoles con alta precisión. Nuestro enfoque es no supervisado, lo que lo diferencia al no requerir etiquetas.
- **Kaggle notebooks sobre clustering de granos:** existen múltiples notebooks públicos que aplican K-Means o clustering jerárquico sobre este mismo dataset, generalmente como ejercicios académicos.
- **AgroVision y startups de visión computacional agrícola:** empresas como Taranis o Gamaya aplican modelos similares (aunque más sofisticados) para análisis de cultivos usando imágenes satelitales y de drones.

Lo que diferencia a K-Beans es la integración completa: desde el preprocesamiento hasta una aplicación web desplegada y usable por alguien sin conocimientos de ML.

## Qué suele faltar en proyectos como este
Muchos proyectos académicos de clustering se quedan en el notebook. Lo que frecuentemente falta, y que intentamos abordar, incluye:

- **Interpretabilidad de los clusters:** decir "cluster 2" no comunica nada útil. Añadir descripciones del perfil morfológico de cada grupo hace la salida accionable.
- **Validación más allá del Silhouette Score:** métricas como el índice Davies-Bouldin o la inercia con el método del codo suelen omitirse.
- **Manejo de datos fuera de rango en producción:** los modelos entrenados en rangos específicos fallan silenciosamente con entradas atípicas si no hay validación en la interfaz.
- **Documentación del proceso de decisión:** por qué K-Means y no DBSCAN, por qué 4 clusters y no 5; estas decisiones rara vez se explican.
- **Pruebas con usuarios reales:** la mayoría de los proyectos no validan si la interfaz es comprensible para alguien del dominio agrícola.

## Falencias en el desarrollo
El proyecto tuvo varias limitaciones a la hora de su desarrollo como:
- **Primer proyecto colaborativo en GitHub:** al no estar familiarizados con el flujo de ramas, durante las primeras semanas trabajamos directamente sobre `main`, lo que generó conflictos de merge que tuvimos que resolver manualmente y con pérdida de algunos cambios intermedios. Hacia el final adoptamos un flujo más ordenado con ramas por feature.
- **Interfaz funcional pero sin pruebas de usabilidad:** la app cumple su función, pero no fue probada con personas ajenas al proyecto, por lo que no sabemos qué tan intuitiva resulta para un usuario del contexto agrícola.

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
├── docs/
│   ├── presentacion.pdf/             # Presentacion final
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
