import joblib
import pandas as pd
from .preprocessing import FEATURES


def load_model(path='models/kmeans_model.pkl'):
    """Carga el modelo K-Means serializado."""
    return joblib.load(path)


def load_scaler(path='models/scaler.pkl'):
    """Carga el StandardScaler serializado."""
    return joblib.load(path)


def predict_cluster(model, scaler, area, perimeter, major_axis, minor_axis, compactness):
    """
    Recibe los valores individuales, los escala y retorna el cluster predicho.
    """
    new_data = pd.DataFrame({
        'Area':            [area],
        'Perimeter':       [perimeter],
        'MajorAxisLength': [major_axis],
        'MinorAxisLength': [minor_axis],
        'Compactness':     [compactness]
    })
    new_scaled = scaler.transform(new_data)
    prediction = model.predict(new_scaled)
    return int(prediction[0]), new_scaled
