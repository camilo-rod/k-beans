from pathlib import Path

# PATHS
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent

# CLUSTERS
CLUSTER_NAMES = {
    0: "Seker",
    1: "Barbunya",
    2: "Bombay",
    3: "Sira",
    4: "Dermason",
    5: "Cali",
    6: "Horoz"
}

BEAN_DESCRIPTIONS = {
    "Seker":    "Frijol pequeño y compacto con forma redondeada.",
    "Barbunya": "Frijol irregular y ancho con manchas características.",
    "Bombay":   "Frijol robusto y compacto de gran tamaño.",
    "Sira":     "Frijol equilibrado y uniforme de tamaño medio.",
    "Dermason": "Frijol pequeño y ligeramente alargado.",
    "Cali":     "Frijol grande y claro con textura uniforme.",
    "Horoz":    "Frijol denso con estructura alargada y fuerte."
}

BEAN_IMAGES = {
    "Seker":    APP_DIR / "static" / "bean5.png",
    "Barbunya": APP_DIR / "static" / "bean0.png",
    "Bombay":   APP_DIR / "static" / "bean1.png",
    "Sira":     APP_DIR / "static" / "bean6.png",
    "Dermason": APP_DIR / "static" / "bean3.png",
    "Cali":     APP_DIR / "static" / "bean2.png",
    "Horoz":    APP_DIR / "static" / "bean4.png"
}

PALETTE = ['#4a90c4', '#d85a30', '#3b9e75', '#e8a020', '#9b59b6', '#e74c3c', '#1abc9c']

# SLIDERS: (min, max, default)
SLIDER_CONFIG = {
    "area":        (10000, 200000, 42000),
    "perimeter":   (400,   1700,   700),
    "major_axis":  (150,   650,    250),
    "minor_axis":  (80,    400,    150),
    "compactness": (0.60,  0.95,   0.80),
}