import joblib
import numpy as np

def load_model(path):

    return joblib.load(path)

def load_scaler(path):

    return joblib.load(path)

def predict_cluster(
    model,
    scaler,
    area,
    perimeter,
    major_axis,
    minor_axis,
    compactness
):

    data = np.array([[
        area,
        perimeter,
        major_axis,
        minor_axis,
        compactness
    ]])

    scaled = scaler.transform(data)

    cluster = model.predict(scaled)[0]

    return cluster, scaled
