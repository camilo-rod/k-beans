import streamlit as st
import numpy as np
import cv2

from src.predict import predict_cluster
from config import CLUSTER_NAMES, BEAN_DESCRIPTIONS, BEAN_IMAGES


def _compacidad(c):
    """Calcula qué tan compacto (redondo) es un contorno."""
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    if p == 0:
        return 0
    return (4 * np.pi * a) / (p ** 2)


def extraer_features(imagen_bytes):
    """Procesa la imagen y extrae las features morfológicas del frijol."""
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Escala de grises y desenfoque para reducir ruido
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Umbral de Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Operaciones morfológicas para separar sombra del frijol
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel_big,   iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    thresh = cv2.erode(thresh, kernel_small, iterations=2)

    # Buscar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Filtrar contornos: descartar ruido pequeño y sombras grandes
    alto, ancho = img.shape[:2]
    area_total  = alto * ancho

    candidatos = [
        c for c in contours
        if 1000 < cv2.contourArea(c) < area_total * 0.5
    ]

    if not candidatos:
        return None, None

    # Elegir el contorno más compacto (más redondo = más frijol, menos sombra)
    contour = max(candidatos, key=_compacidad)
    area    = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    # Elipse equivalente para ejes
    if len(contour) >= 5:
        ellipse    = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        major_axis = max(w, h)
        minor_axis = min(w, h)

    # Compactness = 4π * Area / Perimeter²
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    compactness = float(np.clip(compactness, 0.60, 0.95))

    features = {
        "Area":            float(area),
        "Perimeter":       float(perimeter),
        "MajorAxisLength": float(major_axis),
        "MinorAxisLength": float(minor_axis),
        "Compactness":     compactness,
    }

    # Dibujar contorno sobre imagen original para mostrar al usuario
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_rgb, [contour], -1, (74, 103, 65), 3)

    return features, img_rgb


def render(kmeans, scaler, pca):
    """Renderiza el tab de cámara."""

    st.markdown("""
### Fotografía tu frijol
Toma una foto con tu cámara y la app extraerá automáticamente las medidas del frijol.
""")

    st.warning("""
**Para mejores resultados:**
- Coloca el frijol sobre un **fondo blanco o muy claro**
- Asegúrate de que haya buena iluminación
- Evita sombras sobre el frijol — inclina el celular si es necesario
- Centra el frijol y acércate lo suficiente
- Un solo frijol por foto
""")

    foto = st.camera_input("Tomar foto del frijol")

    if foto is not None:
        with st.spinner("Analizando imagen..."):
            features, img_contorno = extraer_features(foto.getvalue())

        if features is None:
            st.error("No se detectó ningún frijol. Asegúrate de usar fondo claro, buena iluminación y evitar sombras grandes.")
            return

        # Mostrar imagen con contorno detectado
        st.image(img_contorno, caption="Contorno detectado", use_container_width=True)

        # Mostrar features extraídas
        st.success("Frijol detectado. Features extraídas:")

        col1, col2, col3 = st.columns(3)
        col1.metric("Área (px²)",      f"{features['Area']:.0f}")
        col1.metric("Perímetro (px)",   f"{features['Perimeter']:.1f}")
        col2.metric("MajorAxisLength",  f"{features['MajorAxisLength']:.1f}")
        col2.metric("MinorAxisLength",  f"{features['MinorAxisLength']:.1f}")
        col3.metric("Compactness",      f"{features['Compactness']:.3f}")

        st.markdown("---")

        # Validar rangos antes de predecir
        advertencias = []
        if not (10000 <= features["Area"] <= 200000):
            advertencias.append(f"Área fuera de rango ({features['Area']:.0f}px²) — acércate más o aléjate del frijol.")
        if not (400 <= features["Perimeter"] <= 1700):
            advertencias.append(f"Perímetro fuera de rango ({features['Perimeter']:.1f}px).")
        if not (150 <= features["MajorAxisLength"] <= 650):
            advertencias.append(f"MajorAxisLength fuera de rango ({features['MajorAxisLength']:.1f}px).")
        if not (80 <= features["MinorAxisLength"] <= 400):
            advertencias.append(f"MinorAxisLength fuera de rango ({features['MinorAxisLength']:.1f}px).")

        if advertencias:
            for a in advertencias:
                st.warning(a)
            st.info("Ajusta la distancia a la cámara y vuelve a intentarlo, o ve al tab **Predicción** para ingresar los valores manualmente.")
            return

        # Predecir
        cluster, new_scaled = predict_cluster(
            kmeans, scaler,
            features["Area"],
            features["Perimeter"],
            features["MajorAxisLength"],
            features["MinorAxisLength"],
            features["Compactness"],
        )

        st.session_state["cluster"] = cluster
        st.session_state["new_pca"] = pca.transform(new_scaled)

        bean_name = CLUSTER_NAMES.get(cluster, "Desconocido")
        st.success(f"🌱 Frijol clasificado como: **{bean_name}**")

        distance = np.min(kmeans.transform(new_scaled))
        score    = max(0, 100 - (distance * 10))

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