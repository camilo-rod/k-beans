"""
camera_widget.py  —  UI/camera_widget.py

Custom component de cámara para K-Beans.
Sirve templates/index.html como iframe bidireccional.

Flujo:
  1. index.html arranca, envía 'streamlit:componentReady' (apiVersion:1)
  2. Usuario presiona el botón circular → JS envía 'streamlit:setComponentValue'
  3. Python recibe el base64 JPEG como valor de retorno de camera_widget()

Devuelve:
  str  — base64 JPEG de la foto (cambia con cada nueva captura)
  None — hasta que se tome la primera foto
"""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# templates/ está un nivel arriba de UI/
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

_camera_component = components.declare_component(
    name="camera_widget",
    path=str(_TEMPLATES_DIR),   # Streamlit busca index.html aquí
)


def camera_widget(height: int = 520, key: str = "camera_widget") -> str | None:
    """
    Renderiza la cámara a pantalla completa y devuelve el base64 JPEG
    de la foto capturada, o None si aún no se ha tomado ninguna.
    """
    # Parchear todos los iframes del componente para que tengan allow="camera"
    # Streamlit no agrega este atributo por defecto y Chrome lo requiere.
    st.markdown("""
    <script>
    (function patchIframes() {
        document.querySelectorAll('iframe').forEach(function(f) {
            if (!f.allow || !f.allow.includes('camera')) {
                f.allow = 'camera; microphone';
            }
        });
    })();
    // Repetir por si el iframe aún no existe al cargar
    setTimeout(function patchIframes() {
        document.querySelectorAll('iframe').forEach(function(f) {
            if (!f.allow || !f.allow.includes('camera')) {
                f.allow = 'camera; microphone';
            }
        });
    }, 800);
    </script>
    """, unsafe_allow_html=True)

    return _camera_component(height=height, key=key, default=None)