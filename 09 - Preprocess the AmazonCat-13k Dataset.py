from utils.text_preprocessing import preprocess
import pandas as pd
import scipy.sparse as sp
import numpy as np

CLASS_COUNT = 13_330

def is_positive(i):
    return lambda y: i in y

def preprocess_amazoncat13k(dataset, append_content=False):
    dataset_path = f'datasets/AmazonCat-13K/{dataset}.json'
    df = pd.read_json(dataset_path, lines=True)

    if append_content:
        df['title'] = df['title'].str.cat(df['content'], sep=' ')

    row_count = df.shape[0]

    # Calculate compressed sparse row data
    (row_inds, col_inds) = ([], [])
    for row_ind, row_data in enumerate(df['target_ind']):
        for col in row_data:
            row_inds.append(row_ind)
            col_inds.append(col)
    Y_csc_data = np.ones(len(row_inds), dtype='int32')

    X = preprocess(df['title'])
    Y = sp.csc_matrix((Y_csc_data, (row_inds, col_inds)), shape=(row_count, CLASS_COUNT))
    return X, Y

def remove_tokens_under(X, y, token_threshold=0):
    token_lens = X.map(len)
    remove_indices = np.arange(X.shape[0])[token_lens > token_threshold]
    remove_count = token_lens.shape[0] - remove_indices.shape[0]

    print('Instances with less than or equal to {0} tokens will be removed.'.format(token_threshold))
    print('This amounts to {0} instances ({1:.2%}).'.format(remove_count, remove_count / token_lens.shape[0]))

    # Remove instances under token_threshold
    X = X[remove_indices]
    y = y[remove_indices]
    return X, y

X, Y = preprocess_amazoncat13k('trn')
# X, Y = preprocess_amazoncat13k('trn', append_content=True)
np.save('datasets/AmazonCat-13K/X.trn.raw.npy', X)
sp.save_npz('datasets/AmazonCat-13K/Y.trn.raw.npz', Y)
X, Y = remove_tokens_under(X, Y)
np.save('datasets/AmazonCat-13K/X.trn.processed.npy', X)
sp.save_npz(f'datasets/AmazonCat-13K/Y.trn.processed.npz', Y)

X, Y = preprocess_amazoncat13k('tst')
# X, Y = preprocess_amazoncat13k('tst', append_content=True)
np.save('datasets/AmazonCat-13K/X.tst.npy', X)
sp.save_npz(f'datasets/AmazonCat-13K/Y.tst.npz', Y)
