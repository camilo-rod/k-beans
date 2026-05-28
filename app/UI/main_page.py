"""
main_page.py
Flujo completo en una sola página — totalmente responsive (móvil y desktop).
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.predict import predict_cluster
from config import CLUSTER_NAMES, BEAN_DESCRIPTIONS, BEAN_IMAGES, PALETTE, SLIDER_CONFIG


# ─────────────────────────────────────────────
# CSS responsive
# ─────────────────────────────────────────────

def _inject_css():
    st.markdown("""
    <style>
    /* Solo fix para imagen que no se desborde en móvil */
    [data-testid="stImage"] img { max-width: 100% !important; height: auto !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Visión computacional
# ─────────────────────────────────────────────

def _compacidad(c):
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    if p == 0:
        return 0
    return (4 * np.pi * a) / (p ** 2)


def _mascara_hsv(img_blur):
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    _, mask_sat = cv2.threshold(S, 30, 255, cv2.THRESH_BINARY)

    V_inv = cv2.bitwise_not(V)
    _, mask_val = cv2.threshold(V_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_sat_exp = cv2.dilate(mask_sat, k_dilate, iterations=1)

    mask_bean = cv2.bitwise_and(mask_val, mask_sat_exp)

    if cv2.countNonZero(mask_bean) < cv2.countNonZero(mask_val) * 0.15:
        mask_bean = mask_val

    return mask_bean


def extraer_features(imagen_bytes):
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    thresh  = _mascara_hsv(blurred)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel_big,   iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    alto, ancho = img.shape[:2]
    area_total  = alto * ancho

    candidatos = [c for c in contours
                  if cv2.contourArea(c) > 500 and cv2.contourArea(c) < area_total * 0.5]
    if not candidatos:
        return None, None

    contour   = max(candidatos, key=_compacidad)
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if len(contour) >= 5:
        ellipse    = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        major_axis = max(w, h)
        minor_axis = min(w, h)

    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.75

    features = {
        "Area":            float(area),
        "Perimeter":       float(perimeter),
        "MajorAxisLength": float(major_axis),
        "MinorAxisLength": float(minor_axis),
        "Compactness":     float(np.clip(compactness, 0.10, 1.0)),
    }

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_rgb, [contour], -1, (74, 103, 65), 3)
    return features, img_rgb


# ─────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────

def _mostrar_resultado(kmeans, scaler, pca, X_pca,
                       area, perimeter, major_axis, minor_axis, compactness):
    cluster, new_scaled = predict_cluster(
        kmeans, scaler, area, perimeter, major_axis, minor_axis, compactness
    )

    st.session_state["cluster"]   = cluster
    st.session_state["new_pca"]   = pca.transform(new_scaled)
    st.session_state["resultado"] = {
        "area": area, "perimeter": perimeter,
        "major_axis": major_axis, "minor_axis": minor_axis,
        "compactness": compactness,
    }

    bean_name = CLUSTER_NAMES.get(cluster, "Desconocido")
    distance  = np.min(kmeans.transform(new_scaled))
    score     = max(0, 100 - (distance * 10))

    st.markdown("---")
    st.markdown("### Resultado")
    st.success(f"Frijol clasificado como: **{bean_name}**")

    # ── Imagen + descripción ──────────────────
    # En móvil: imagen arriba, texto abajo (1 columna efectiva)
    # En desktop: imagen izquierda, texto derecha
    image_path = BEAN_IMAGES.get(bean_name)
    if image_path and image_path.exists():
        img_col, text_col = st.columns([1, 2])
        with img_col:
            st.image(str(image_path), use_container_width=True)
        with text_col:
            st.markdown(f"#### {bean_name}")
            st.write(BEAN_DESCRIPTIONS.get(bean_name, ""))
            st.markdown("---")
            st.markdown("#### Nivel de similitud")
            st.markdown(f"**{score:.1f}%**")
            st.progress(int(score))
    else:
        st.markdown(f"#### Nivel de similitud: **{score:.1f}%**")
        st.progress(int(score))

    # ── Gráfica PCA ───────────────────────────
    st.markdown("---")
    st.markdown("### Distribución de clusters (PCA)")

    labels = kmeans.labels_
    pca_df = pd.DataFrame({
        "PCA1":    X_pca[:, 0],
        "PCA2":    X_pca[:, 1],
        "Cluster": labels,
    })

    # dpi=150 → nítido en pantallas de alta densidad (móviles Retina/AMOLED)
    # figsize pequeño → use_container_width lo estira al ancho disponible
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)

    sns.scatterplot(
        data=pca_df, x="PCA1", y="PCA2", hue="Cluster",
        palette=PALETTE[:kmeans.n_clusters], alpha=0.6,
        s=18, edgecolor="white", linewidth=0.2, ax=ax,
    )

    point = st.session_state["new_pca"]
    ax.scatter(point[:, 0], point[:, 1], s=280, marker="*",
               color="red", label=f"Tu frijol ({bean_name})", zorder=10)

    ax.set_title("Clusters de frijoles", fontsize=9, pad=6)
    ax.set_xlabel("Componente principal 1", fontsize=7)
    ax.set_ylabel("Componente principal 2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.6, color="gray")
    ax.set_axisbelow(True)

    legend = ax.get_legend()
    legend.set_title("Cluster", prop={"size": 6})
    for text in legend.get_texts():
        text.set_fontsize(5)

    for spine in ax.spines.values():
        spine.set_edgecolor("#4a90c4")
        spine.set_linewidth(0.7)

    plt.tight_layout(pad=0.5)
    # use_container_width=True hace que ocupe todo el ancho en móvil y desktop
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────

def _sliders_manuales(v_area, v_perimeter, v_major_axis, v_minor_axis, v_compactness):
    """Sliders en una sola columna en móvil (Streamlit los apila automáticamente)."""

    def dyn_range(val, lo, hi, pct=0.5):
        margin = max((hi - lo) * pct, abs(val) * pct, 1.0)
        return min(lo, val - margin), max(hi, val + margin)

    a_lo,  a_hi  = dyn_range(v_area,      10000, 200000)
    p_lo,  p_hi  = dyn_range(v_perimeter,   400,   1700)
    mj_lo, mj_hi = dyn_range(v_major_axis,  150,    650)
    mn_lo, mn_hi = dyn_range(v_minor_axis,   80,    400)

    # 1 columna en móvil, 2 en desktop — Streamlit colapsa columnas estrechas solo
    col1, col2 = st.columns([1, 1])
    with col1:
        area       = st.slider("Área (px²)",          float(a_lo),  float(a_hi),  v_area,       help="Superficie total del frijol")
        perimeter  = st.slider("Perímetro (px)",       float(p_lo),  float(p_hi),  v_perimeter,  help="Longitud del contorno")
        major_axis = st.slider("MajorAxisLength (px)", float(mj_lo), float(mj_hi), v_major_axis, help="Eje más largo del frijol")
    with col2:
        minor_axis  = st.slider("MinorAxisLength (px)", float(mn_lo), float(mn_hi), v_minor_axis,  help="Eje más corto del frijol")
        compactness = st.slider("Compactness",          0.10, 1.0, v_compactness, step=0.01,       help="Qué tan redondo es el frijol")

    return area, perimeter, major_axis, minor_axis, compactness


# ─────────────────────────────────────────────
# Render principal
# ─────────────────────────────────────────────

def render(kmeans, scaler, pca, X_pca):

    _inject_css()

    # ── PASO 1: Info ──────────────────────────
    st.markdown("### Analiza tu frijol")

    st.info(
        "**Para mejores resultados:** fondo blanco, buena iluminación, "
        "sin sombras, un solo frijol centrado en el encuadre."
    )

    with st.expander("¿Cómo se miden las features?"):
        st.markdown("""
| Feature | Descripción |
|---|---|
| Área | Superficie total en píxeles |
| Perímetro | Longitud del contorno |
| MajorAxisLength | Eje más largo |
| MinorAxisLength | Eje más corto |
| Compactness | Redondez (1.0 = esfera perfecta) |
""")

    # ── PASO 2: Cámara ────────────────────────
    foto = st.camera_input("Tomar foto del frijol", key="camara_nativa")

    foto_id = id(foto) if foto is not None else None
    if foto is not None and foto_id != st.session_state.get("_foto_id_procesado"):
        st.session_state["_foto_id_procesado"] = foto_id
        for key in ("foto_feats", "img_contorno", "cluster", "new_pca", "resultado"):
            st.session_state.pop(key, None)

        with st.spinner("Analizando imagen…"):
            feats, img_contorno = extraer_features(foto.getvalue())

        if feats is None:
            st.error("No se detectó ningún frijol. Usa fondo claro y buena iluminación.")
        else:
            st.session_state["foto_feats"]   = feats
            st.session_state["img_contorno"] = img_contorno

    if foto is None:
        for key in ("_foto_id_procesado", "foto_feats", "img_contorno",
                    "cluster", "new_pca", "resultado"):
            st.session_state.pop(key, None)

    # ── PASO 3: Lógica de datos ───────────────
    tiene_foto = bool(st.session_state.get("foto_feats"))

    defaults = {
        "area":        float(SLIDER_CONFIG["area"][2]),
        "perimeter":   float(SLIDER_CONFIG["perimeter"][2]),
        "major_axis":  float(SLIDER_CONFIG["major_axis"][2]),
        "minor_axis":  float(SLIDER_CONFIG["minor_axis"][2]),
        "compactness": float(SLIDER_CONFIG["compactness"][2]),
    }

    if tiene_foto:
        feats = st.session_state["foto_feats"]

        # Contorno detectado
        st.image(st.session_state["img_contorno"],
                 caption="Contorno detectado", use_container_width=True)
        st.success("Frijol detectado — features extraídas automáticamente:")

        # Métricas: 2 columnas en móvil (más legible que 3)
        c1, c2 = st.columns(2)
        c1.metric("Área (px²)",      f"{feats['Area']:.0f}")
        c1.metric("Perímetro (px)",  f"{feats['Perimeter']:.1f}")
        c1.metric("Compactness",     f"{feats['Compactness']:.3f}")
        c2.metric("MajorAxisLength", f"{feats['MajorAxisLength']:.1f}")
        c2.metric("MinorAxisLength", f"{feats['MinorAxisLength']:.1f}")

        st.markdown("")

        # Toggle ajuste manual — visible ANTES de predecir
        ajuste_manual = st.toggle("Ajustar valores manualmente", value=False,
                                  help="Modifica las medidas antes de predecir")
        if ajuste_manual:
            st.caption("Valores pre-rellenados desde la foto. Ajústalos y presiona el botón.")
            area, perimeter, major_axis, minor_axis, compactness = _sliders_manuales(
                float(feats["Area"]), float(feats["Perimeter"]),
                float(feats["MajorAxisLength"]), float(feats["MinorAxisLength"]),
                float(feats["Compactness"]),
            )
            st.markdown("")
            if st.button("Predecir con valores ajustados", use_container_width=True, type="primary"):
                _mostrar_resultado(kmeans, scaler, pca, X_pca,
                                   area, perimeter, major_axis, minor_axis, compactness)
        else:
            # Botón principal
            if st.button("Predecir frijol", use_container_width=True, type="primary"):
                _mostrar_resultado(
                    kmeans, scaler, pca, X_pca,
                    feats["Area"], feats["Perimeter"],
                    feats["MajorAxisLength"], feats["MinorAxisLength"],
                    feats["Compactness"],
                )
            elif "resultado" in st.session_state:
                res = st.session_state["resultado"]
                _mostrar_resultado(kmeans, scaler, pca, X_pca,
                                   res["area"], res["perimeter"],
                                   res["major_axis"], res["minor_axis"],
                                   res["compactness"])

    else:
        # Rama manual
        st.markdown("---")
        st.markdown("### Ingresa las medidas manualmente")
        st.caption("Ingresa las medidas físicas del frijol que deseas clasificar.")

        area, perimeter, major_axis, minor_axis, compactness = _sliders_manuales(
            defaults["area"], defaults["perimeter"],
            defaults["major_axis"], defaults["minor_axis"],
            defaults["compactness"],
        )

        st.markdown("")
        if st.button("Predecir frijol", use_container_width=True, type="primary"):
            _mostrar_resultado(kmeans, scaler, pca, X_pca,
                               area, perimeter, major_axis, minor_axis, compactness)
        elif "resultado" in st.session_state:
            res = st.session_state["resultado"]
            _mostrar_resultado(kmeans, scaler, pca, X_pca,
                               res["area"], res["perimeter"],
                               res["major_axis"], res["minor_axis"],
                               res["compactness"])