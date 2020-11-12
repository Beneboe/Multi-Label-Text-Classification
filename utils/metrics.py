from typing import Tuple, List, Dict, Any
import numpy as np

# Helper functions
def binarize(y_predict, threshold = 0.5):
    return (y_predict > threshold).astype('int8')

def get_confusion(y_predict, y_expected, threshold = 0.5) -> Tuple[int, int, int, int]:
    y_predict = binarize(y_predict, threshold)

    tp_items = np.logical_and(y_predict == 1, y_expected == 1)
    fp_items = np.logical_and(y_predict == 1, y_expected == 0)
    fn_items = np.logical_and(y_predict == 0, y_expected == 1)
    tn_items = np.logical_and(y_predict == 0, y_expected == 0)

    tp = np.count_nonzero(tp_items, axis=0)
    fp = np.count_nonzero(fp_items, axis=0)
    fn = np.count_nonzero(fn_items, axis=0)
    tn = np.count_nonzero(tn_items, axis=0)

    return (tp, fp, fn, tn)

# Metric functions
def count(y_predict, y_expected, threshold = 0.5) -> int:
    tp, fp, fn, tn = get_confusion(y_predict, y_expected, threshold)
    return int(tp + fp + fn + tn)

def accuracy(y_predict, y_expected, threshold = 0.5) -> float:
    tp, fp, fn, tn = get_confusion(y_predict, y_expected, threshold)
    return (tp + tn) / (tp + fp + fn + tn)

def recall(y_predict, y_expected, threshold = 0.5) -> float:
    tp, _, fn, _ = get_confusion(y_predict, y_expected, threshold)
    return tp / (tp + fn)

def precision(y_predict, y_expected, threshold = 0.5) -> float:
    tp, fp, _, _ = get_confusion(y_predict, y_expected, threshold)
    return tp / (tp + fp)

def f1measure(y_predict, y_expected, threshold = 0.5) -> float:
    pr = precision(y_predict, y_expected, threshold)
    rc = recall(y_predict, y_expected, threshold)
    return 2 * (pr * rc) / (pr + rc)

def all_metrics(y_predict, y_expected, threshold = 0.5) -> Dict[str, Any]:
    return {
        'accuracy': accuracy(y_predict, y_expected),
        'recall': recall(y_predict, y_expected),
        'precision': precision(y_predict, y_expected),
        'f1 measure': f1measure(y_predict, y_expected),
    }

def macro_f1measure(y_predict, y_expected) -> float:
    class_count = y_expected.T.shape[0]
    # Collect the F1 Measures for each class
    f1s = [f1measure(y_predict.T[i], y_expected.T[i]) for i in range(class_count)]
    return np.average(f1s)

def micro_f1measure(y_predict, y_expected) -> float:
    # Same as accuracy over the entire array
    true_count = np.count_nonzero((y_predict == y_expected).all(axis=1))
    return true_count / (y_predict.shape[0])
