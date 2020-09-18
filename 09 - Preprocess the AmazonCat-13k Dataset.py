# %% [markdown]
# # Preprocess the AmazonCat-13k Dataset
# Setup the hyperparameters

# %%
CLASS_COUNT = 13330
DATASET_TYPE = 'trn'
# CUTOFF is inclusive
CUTOFF = 2 if DATASET_TYPE == 'trn' else 0
INPUT_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.json'
OUTPUT_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.processed.json'

# %% [markdown]
# First, load the dataset.

# %%
import pandas as pd
from nltk import word_tokenize

df = pd.read_json(INPUT_PATH, lines=True)
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
# ## Preprocess the Dataset

# %%
from utils.text_preprocessing import preprocess
from keras.preprocessing.sequence import pad_sequences

X = preprocess(df['title'])
y = df['target_ind']

# %% [markdown]
# Next, we can calculate the statistics for class frequencies, title char lengths, content char lengths, and instance class counts.

# %%
from utils.dataset import var_stats, class_frequencies

vlen = np.vectorize(len)
token_lens = vlen(X)

ds_stats_index = []
ds_stats = []

ds_stats_index.append('class frequencies')
ds_stats.append(var_stats(class_frequencies(CLASS_COUNT, df['target_ind'])))

ds_stats_index.append('title char lengths')
ds_stats.append(var_stats(vlen(df['title'])))

ds_stats_index.append('content char lengths')
ds_stats.append(var_stats(vlen(df['content'])))

ds_stats_index.append('instance class count')
ds_stats.append(var_stats(df['target_ind'].map(len)))

ds_stats_index.append('token lengths')
ds_stats.append(var_stats(token_lens))

pd.DataFrame(ds_stats, index=ds_stats_index)

# %% [markdown]
# Cut off instances with not enough tokens.

# %%
print('Instances with less than or equal to {0} tokens get cut off.'.format(CUTOFF))
count = np.count_nonzero(token_lens <= CUTOFF)
print('This amounts to {0} instances ({1:.2%}).'.format(
    count, count / token_lens.shape[0]))

# Cutoff
indices = np.arange(X.shape[0])[token_lens > CUTOFF]
X = X[indices]
y = y[indices]

# %% [markdown]
# Calculate the metrics after the cutoff.

# %%
token_lens = vlen(X)

ds_stats_index.append('class frequencies (after cutoff)')
ds_stats.append(var_stats(class_frequencies(CLASS_COUNT, y)))

ds_stats_index.append('instance class count (after cutoff)')
ds_stats.append(var_stats(y.map(len)))

ds_stats_index.append('token lengths (after cutoff)')
ds_stats.append(var_stats(token_lens))

pd.DataFrame(ds_stats, index=ds_stats_index)

# %%
df_processed = pd.DataFrame({ 'X': X, 'y': y })
df_processed

# %% [markdown]
# Save the dataset

# %%
df_processed.to_json(OUTPUT_PATH, orient='records', lines=True)