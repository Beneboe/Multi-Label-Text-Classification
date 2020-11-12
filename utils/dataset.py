import numpy as np
import scipy.sparse as sp
import pandas as pd
import gensim
from keras.preprocessing.sequence import pad_sequences
from numpy.random import default_rng


amazoncat13k_top10_label_data = [
    (1471,355211),
    (7961,194561),
    (7892,128026),
    (9237,120090),
    (7083,97803),
    (7891,88967),
    (4038,76277),
    (10063,75035),
    (12630,71667),
    (8108,71667),
]

amazoncat13k_threshold_label_data = [
    (50,6554,50),
    (100,4949,100),
    (1000,7393,996),
    (10000,84,9976),
    (50000,9202,48521),
    (100000,7083,96012),
]

amazoncat13k_top10_labels, _ = zip(*amazoncat13k_top10_label_data)
amazoncat13k_thresholds, amazoncat13k_threshold_labels, _ = zip(*amazoncat13k_threshold_label_data)


# Statistical functions
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

# Import functions
def import_amazoncat13k(dataset, length):
    X = np.load(f'datasets/AmazonCat-13K/X.{dataset}.npy', allow_pickle=True)
    y = sp.load_npz(f'datasets/AmazonCat-13K/Y.{dataset}.npz')

    # Make sequences same length
    X = pad_sequences(X, maxlen=length)

    return X, y

def import_embedding_layer():
    model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin", binary=True)
    return model.get_keras_embedding(train_embeddings=False)

def get_dataset(X, y, i, p_weight=None):
    rng = default_rng(42)

    yi = y[:, i].toarray().flatten()

    ind = np.arange(X.shape[0])
    mask = yi == 1
    pos_ind = ind[mask]
    neg_ind = ind[~mask]

    # Subsample negative indices
    if p_weight is not None:
        p_count = min(int(X.shape[0] * p_weight), pos_ind.shape[0])
        n_count = int(p_count * ((1 - p_weight) / p_weight))

        neg_ind = rng.choice(neg_ind, n_count, replace=False)

    new_ind = np.concatenate((pos_ind,neg_ind))
    rng.shuffle(new_ind)

    X = X[new_ind]
    yi = yi[new_ind]

    return X, yi
