# pipelines/prediction_pipeline.py

import pandas as pd
import pickle

def predict(data):

    """python pipelines/prediction_pipeline.py"""

    model_path = "models/model.pkl"

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    prediction = model.predict(data)

    return prediction
