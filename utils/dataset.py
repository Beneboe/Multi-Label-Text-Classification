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
    (6853,10,10),
    (9971,100,100),
    (3175,1000,1000),
    (1370,9954,10000),
    (7083,97803,100000),
]

amazoncat13k_top10_labels, _ = zip(*amazoncat13k_top10_label_data)
amazoncat13k_threshold_labels, _, amazoncat13k_thresholds = zip(*amazoncat13k_threshold_label_data)

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
