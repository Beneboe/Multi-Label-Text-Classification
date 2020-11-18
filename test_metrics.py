import utils.metrics as mt
import numpy as np

def test_get_confusion():
    y_predict = np.array([1, 1, 0, 0])
    y_expected = np.array([1, 0, 1, 0])
    tp, fp, fn, tn = mt.get_confusion(y_predict, y_expected)
    assert tp == 1
    assert fp == 1
    assert fn == 1
    assert tn == 1

def test_count():
    y_predict = np.array([1, 1, 0, 0])
    y_expected = np.array([1, 0, 1, 0])
    assert mt.count(y_predict, y_expected) == 4

def test_accuracy():
    y_predict = np.array([1, 1, 0, 0])
    y_expected = np.array([1, 0, 1, 0])
    assert mt.accuracy(y_predict, y_expected) == 0.5

def test_recall():
    y_predict = np.array([1, 1, 0, 0])
    y_expected = np.array([1, 0, 1, 0])
    assert mt.recall(y_predict, y_expected) == 0.5

def test_precision():
    y_predict = np.array([1, 1, 0, 0])
    y_expected = np.array([1, 0, 1, 0])
    assert mt.precision(y_predict, y_expected) == 0.5

def test_f1score():
    y_predict = np.array([1, 1, 0, 0])
    y_expected = np.array([1, 0, 1, 0])
    assert mt.f1score(y_predict, y_expected) == 0.5