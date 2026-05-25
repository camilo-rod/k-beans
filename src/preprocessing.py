import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):

    return pd.read_csv(path)

def get_features():

    return [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "Compactness"
    ]

def scale_features(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
