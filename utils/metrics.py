from typing import Tuple, List
import numpy as np

# Helper functions
def get_confusion(y_predict, y_expected) -> Tuple[int, int, int, int]:
    tp_items = np.logical_and(y_predict == 1, y_expected == 1)
    fp_items = np.logical_and(y_predict == 1, y_expected == 0)
    fn_items = np.logical_and(y_predict == 0, y_expected == 1)
    tn_items = np.logical_and(y_predict == 0, y_expected == 0)

    tp = np.count_nonzero(tp_items, axis=0)
    fp = np.count_nonzero(fp_items, axis=0)
    fn = np.count_nonzero(fn_items, axis=0)
    tn = np.count_nonzero(tn_items, axis=0)

    return (tp, fp, fn, tn)

def get_prediction(model, X):
    y_predict = model.predict(X).flatten()
    # step function
    y_predict[y_predict < 0.5] = 0.0
    y_predict[y_predict >= 0.5] = 1.0
    return y_predict

def get_all_predictions(models, X):
    y_predict = np.zeros((X.shape[0],len(models)))
    for i in range(len(models)):
        y_predict[:,i] = get_prediction(models[i], X)
    return y_predict

# Metric functions
def count(y_predict, y_expected) -> int:
    tp, fp, fn, tn = get_confusion(y_predict, y_expected)
    return int(tp + fp + fn + tn)

def accuracy(y_predict, y_expected) -> float:
    tp, fp, fn, tn = get_confusion(y_predict, y_expected)
    return (tp + tn) / (tp + fp + fn + tn)

def recall(y_predict, y_expected) -> float:
    tp, _, fn, _ = get_confusion(y_predict, y_expected)
    return tp / (tp + fn)

def precision(y_predict, y_expected) -> float:
    tp, fp, _, _ = get_confusion(y_predict, y_expected)
    return tp / (tp + fp)

def f1measure(y_predict, y_expected) -> float:
    pr = precision(y_predict, y_expected)
    rc = recall(y_predict, y_expected)
    return 2 * (pr * rc) / (pr + rc)

def macro_f1measure(y_predict, y_expected) -> float:
    class_count = y_expected.T.shape[0]
    # Collect the F1 Measures for each class
    f1s = [f1measure(y_predict.T[i], y_expected.T[i]) for i in range(class_count)]
    return np.average(f1s)

def micro_f1measure(y_predict, y_expected) -> float:
    # Same as accuracy over the entire array
    true_count = np.count_nonzero((y_predict == y_expected).all(axis=1))
    return true_count / (y_predict.shape[0])

# Evaluation functions
def evaluate(model, X, y_expected, metric) -> List[float]:
    y_predict = get_prediction(model, X)
    return metric(y_predict, y_expected)

def evaluate_multiple(models, X, y_expected, metric) -> float:
    y_predict = np.zeros((X.shape[0],len(models)))
    for i in range(len(models)):
        y_predict[:,i] = get_prediction(models[i], X)

    return metric(y_predict, y_expected)
