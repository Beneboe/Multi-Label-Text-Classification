from typing import Tuple
import numpy as np

def get_confusion(y_predict, y_expected) -> Tuple[int, int, int, int]:
    tp_items = np.logical_and(y_predict == 1, y_expected == 1)
    fp_items = np.logical_and(y_predict == 1, y_expected == 0)
    fn_items = np.logical_and(y_predict == 0, y_expected == 1)
    tn_items = np.logical_and(y_predict == 0, y_expected == 0)

    tp = np.sum(tp_items)
    fp = np.sum(fp_items)
    fn = np.sum(fn_items)
    tn = np.sum(tn_items)

    return (tp, fp, fn, tn)

def count(y_predict, y_expected) -> float:
    tp, fp, fn, tn = get_confusion(y_predict, y_expected)
    return tp + fp + fn + tn

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
    f1s = [f1measure(y_predict.T[i], y_expected.T[i]) for i in range(class_count)]
    return np.average(f1s)

def evaluate(model, X, y_expected, metric) -> float:
    y_predict = model.predict(X).flatten()
    # step function
    y_predict[y_predict < 0.5] = 0.0
    y_predict[y_predict >= 0.5] = 1.0
    return metric(y_predict, y_expected)
