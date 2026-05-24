import pandas as pd
from sklearn.preprocessing import StandardScaler


# Variables morfológicas usadas para el clustering
FEATURES = [
    'Area',
    'Perimeter',
    'MajorAxisLength',
    'MinorAxisLength',
    'Compactness'
]


def load_data(path='data/processed/Dry_Bean_Dataset_clean.csv'):
    """Carga el dataset y limpia los nombres de columnas."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def get_features():
    """Retorna la lista de features usadas en el modelo."""
    return FEATURES


def scale_features(X):
    """
    Ajusta un StandardScaler y escala X.
    Retorna (X_scaled, scaler).
    Usar solo durante el entrenamiento en el notebook.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
