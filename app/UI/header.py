import base64
import streamlit as st
from config import APP_DIR


def render():
    """Renderiza las fuentes, el header con logo y título."""

    # Fuentes (debe ir en st.markdown, el <link> del HTML se pierde en Streamlit)
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    # Logo en base64
    with open(APP_DIR / "static" / "logo.png", "rb") as f:
        logo_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div class="kb-header">
      <div class="kb-bg-texture"></div>
      <div class="kb-header-content" style="display:flex; align-items:center; gap:24px;">
        <img src="{logo_b64}" width="100" style="border-radius:50%; background:transparent; padding:0; flex-shrink:0; filter: drop-shadow(0 6px 18px rgba(0,0,0,.35));"/>
        <div>
          <h1 class="kb-title">
            <span class="kb-title-k">K</span><span class="kb-title-dash">—</span><span class="kb-title-beans">Beans</span>
          </h1>
          <p class="kb-subtitle">Agrupación inteligente de frijoles mediante K-Means y análisis de componentes principales</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)