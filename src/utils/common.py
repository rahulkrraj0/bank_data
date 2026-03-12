# src/utils/common.py

import pickle

def save_object(file_path, obj):
    """
    file_path: str
    obj: Any sort of object

    from src.utils.common import save_object
    save_object("models/model.pkl", model)

    """

    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path):

    """
    file_path: str

    from src.utils.common import load_object
    model = load_object("models/model.pkl")

    """

    with open(file_path, "rb") as file:
        return pickle.load(file)