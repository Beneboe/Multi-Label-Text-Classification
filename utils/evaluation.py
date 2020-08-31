import numpy as np

def get_confusion(y_predict, y_expected):
    tp_items = np.logical_and(y_predict == 1, y_expected == 1)
    fp_items = np.logical_and(y_predict == 1, y_expected == 0)
    fn_items = np.logical_and(y_predict == 0, y_expected == 1)
    tn_items = np.logical_and(y_predict == 0, y_expected == 0)

    tp = np.sum(tp_items)
    fp = np.sum(fp_items)
    fn = np.sum(fn_items)
    tn = np.sum(tn_items)

    return (tp, fp, fn, tn)

def count(confusion):
    tp, fp, fn, tn = confusion
    return tp + fp + fn + tn

def accuracy(confusion):
    tp, fp, fn, tn = confusion
    return (tp + tn) / (tp + fp + fn + tn)

def recall(confusion):
    tp, fp, fn, tn = confusion
    return tp / (tp + fn)

def precision(confusion):
    tp, fp, fn, tn = confusion
    return tp / (tp + fp)

def f1measure(confusion):
    tp, fp, fn, tn = confusion
    return 2 * (precision(confusion) * recall(confusion)) / (precision(confusion) + recall(confusion))

def evaluate(model, X, y_expected, metric):
    y_predict = model.predict(X).flatten()
    # step function
    y_predict[y_predict < 0.5] = 0.0
    y_predict[y_predict >= 0.5] = 1.0
    confusion = get_confusion(y_predict, y_expected)
    return metric(confusion)
