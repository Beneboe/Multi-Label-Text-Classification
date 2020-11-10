import utils.metrics as mt
import numpy as np
import json

def get_name(id, type_name):
    return f'{id}_{type_name}'

# Metrics storage
def get_prediction_path(name) -> str:
    return f'results/predict/{name}.npz'

def get_metrics_path(name) -> str:
    return f'results/metrics/{name}.json'

def get_confusions_path(name) -> str:
    return f'results/confusions/{name}.json'

def load_prediction(id, type_name):
    with open(get_prediction_path(get_name(id, type_name)), 'rb') as fp:
        return np.load(fp)

def save_prediction(id, type_name, y_predict):
    with open(get_prediction_path(get_name(id, type_name)), 'wb') as fp:
        np.save(fp, y_predict)

def load_metrics(id, type_name):
    with open(get_metrics_path(get_name(id, type_name)), 'r') as fp:
        return json.load(fp)

def save_metrics(id, type_name, metrics):
    with open(get_metrics_path(get_name(id, type_name)), 'w') as fp:
        json.dump(metrics, fp)

def load_confusion(id, type_name):
    with open(get_confusions_path(get_name(id, type_name)), 'r') as fp:
        return json.load(fp)

def save_confusion(id, type_name, cm):
    with open(get_confusions_path(get_name(id, type_name)), 'w') as fp:
        json.dump(cm, fp)

# Model storage
def get_weights_path(name) -> str:
    return f'results/weights/{name}'

def get_history_path(name) -> str:
    return f'results/history/{name}.json'

def load_history(id, type_name):
    with open(get_history_path(get_name(id, type_name)), 'r') as fp:
        return json.load(fp)

def save_history(id, type_name, history):
    with open(get_history_path(get_name(id, type_name)), 'w') as fp:
        json.dump(history, fp)