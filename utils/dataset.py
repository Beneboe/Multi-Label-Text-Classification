import numpy as np
import pandas as pd
import gensim
from keras.preprocessing.sequence import pad_sequences
from numpy.random import default_rng

def get_stats(a):
    amax = a.max()
    amin = a.min()
    amean = a.mean()
    amedian = np.median(a)

    return {
        'max': amax,
        'min': amin,
        'mean': amean,
        'median': amedian,
        'max count': np.count_nonzero(a == amax),
        'min count': np.count_nonzero(a == amin),
        'mean count': np.count_nonzero(a == np.round(amean)),
        'median count': np.count_nonzero(a == amedian),
        'max arg': a.argmax(),
        'min arg': a.argmin(),
        'deviation': np.std(a),
    }

def class_frequencies(count, labels_array):
    freqs = np.zeros((count,), dtype='int32')
    for labels in labels_array:
        label_ids = np.array(labels)
        freqs[label_ids] += 1
    return freqs

def import_dataset(path, length):
    ds_frame = pd.read_json(path, lines=True)
    # Make sequences same length
    X = pad_sequences(ds_frame['X'], maxlen=length)
    y = ds_frame['y']
    return X, y

def import_embedding_layer():
    model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin", binary=True)
    return model.get_keras_embedding(train_embeddings=False)

def is_positive(i):
    return lambda y: i in y

def is_negative(i):
    return lambda y: i not in y

def get_dataset(X, y, i, balanced=True):
    rng = default_rng(42)

    X_positive = X[y.map(is_positive(i))]
    X_negative = X[y.map(is_negative(i))]

    # Subsample negative indices
    if balanced:
        X_negative = rng.choice(X_negative, X_positive.shape[0], replace=False)

    y_positive = np.ones(X_positive.shape[0], dtype='int8')
    y_negative = np.zeros(X_negative.shape[0], dtype='int8')

    X = np.concatenate((X_positive,X_negative))
    y = np.concatenate((y_positive,y_negative))

    # Shuffle the data
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y