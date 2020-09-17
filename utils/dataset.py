import numpy as np
import pandas as pd

def var_stats(a):
    return {
        'max': a.max(),
        'min': a.min(),
        'mean': a.mean(),
        'max count': np.count_nonzero(a == a.max()),
        'min count': np.count_nonzero(a == a.min()),
        'mean count': np.count_nonzero(a == np.round(a.mean())),
        'max arg': a.argmax(),
        'min arg': a.argmin(),
    }

def class_frequencies(count, labels_array):
    freqs = np.zeros((count,), dtype='int32')
    for labels in labels_array:
        label_ids = np.array(labels)
        freqs[label_ids] += 1
    return freqs
