# %% [markdown]
# # Preprocess the AmazonCat-13k Dataset
# Setup the hyperparameters

# %%
CLASS_COUNT = 13330
DATASET_TYPE = 'trn'

# %% [markdown]
# First, load the dataset.

# %%
import pandas as pd
from nltk import word_tokenize

df = pd.read_json(
    f'datasets/AmazonCat13K/{DATASET_TYPE}.json',
    lines=True,
)
df

# %% [markdown]
# The dataset has the fields: *uid*, *title*, *content*, *target_ind*, *target_rel*.
# Next, we can calculate the maximum and minimum inds.

# %%
import numpy as np

max_ind = 0
min_ind = 2_147_483_647
for inds in df['target_ind']:
    a = np.array(inds)
    ma = a.max()
    if ma > max_ind:
        max_ind = ma
    mi = a.min()
    if mi < min_ind:
        min_ind = mi

print("Max ind:", max_ind)
print("Min ind:", min_ind)
print("Count (= difference + 1):", max_ind - min_ind + 1)
print("Count (expected):", CLASS_COUNT)

# %% [markdown]
# Next, we can calculate the class frequencies.

# %%

freqs = np.zeros((CLASS_COUNT,), dtype='int32')
for inds in df['target_ind']:
    ii = np.array(inds)
    freqs[ii] += 1

ma = freqs.argmax()
print("Most frequent ind:", ma)
print("Most frequent ind (count):", freqs[ma])

print()

mi = freqs.argmin()
print("Least frequent ind:", mi)
print("Least frequent ind (count):", freqs[mi])

print()

print("Mean frequency:", freqs.mean())

del freqs

# %% [markdown]
# Next, we can calculate the text lengths.

# %%
vlen = np.vectorize(len)
title_lens = vlen(df['title'].to_numpy())
print("Shortest title length:", title_lens.min())
print("Longest title length:", title_lens.max())
print("Average title length:", title_lens.mean())

del title_lens
print()

content_lens = vlen(df['content'].to_numpy())
print("Shortest content length:", content_lens.min())
print("Longest content length:", content_lens.max())
print("Average content length:", content_lens.mean())

del content_lens

# %% [markdown]
# Next, we can calculate the number of classes for each instance.

# %%
lens = df['target_ind'].map(len)

print("Average number of classes:", lens.mean())
print("Maximum number of classes:", lens.max())
print("Minimum number of classes:", lens.min())

del lens

# %% [markdown]
# ## Preprocess the Dataset

# %%
from utils.text_preprocessing import preprocess
from keras.preprocessing.sequence import pad_sequences

df_processed = pd.DataFrame({
    'X': preprocess(df['title']),
    'y': df['target_ind']
})
df_processed

# %% [markdown]
# Save the dataset

# %%
df_processed.to_json(f'datasets/AmazonCat13K.{DATASET_TYPE}.json', orient='records', lines=True)

# %% [markdown]
# Next, we can calculate the text lengths (again).

# %%
vlen = np.vectorize(len)
lens = vlen(df_processed['X'])
print("Shortest text length:", lens.min())
c = np.count_nonzero(lens == lens.min())
print("Shortest text length (count):", c)
print("Shortest text length (portion):", c / lens.shape[0])
print("Longest text length:", lens.max())
print("Longest text length (count):", np.count_nonzero(lens == lens.max()))
print("Average text length:", lens.mean())
print("Average text length (count):", np.count_nonzero(lens == np.round(lens.mean())))

del lens

# %%
