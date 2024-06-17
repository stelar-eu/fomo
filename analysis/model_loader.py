import pickle
from prophet.serialize import model_from_json

def load_prophet_model(filepath: str):

    with open(filepath, "r") as f:
        return model_from_json(f.read())
    
def loaf_sktime_model(filepath: str):

    with open(filepath, "rb") as f:
        return pickle.load(f)
    
def load_model(filepath):

    try:
        return load_prophet_model(filepath)
    except UnicodeDecodeError:
        return loaf_sktime_model(filepath)