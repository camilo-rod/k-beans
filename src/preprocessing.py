import pandas as pd

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
