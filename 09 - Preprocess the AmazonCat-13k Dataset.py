# %%
from utils.text_preprocessing import preprocess
import pandas as pd
import numpy as np

# %% [markdown]
# # Preprocess the AmazonCat-13k Dataset

# %%
CLASS_COUNT = 13330
DATASET_TYPE = 'trn'
ADD_CONTENT = False
# CUTOFF is inclusive
CUTOFF = 0 if DATASET_TYPE == 'trn' else 0
DS_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.json'
SUFFIX = '.content' if ADD_CONTENT else ''
BEFORE_CUTOFF_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}{SUFFIX}.before.json'
AFTER_CUTOFF_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}{SUFFIX}.processed.json'

# %% [markdown]
# First, load the dataset.

# %%
df = pd.read_json(DS_PATH, lines=True)
df

# %% [markdown]
# ## Preprocess the Dataset

# %%
if ADD_CONTENT:
    df['title'] = df['title'].str.cat(df['content'], sep=' ')
X = preprocess(df['title'])
y = df['target_ind']

# %% [markdown]
# Save the dataset before the cutoff

df_processed = pd.DataFrame({ 'X': X, 'y': y })
df_processed.to_json(BEFORE_CUTOFF_PATH, orient='records', lines=True)

# %%
vlen = np.vectorize(len)
token_lens = vlen(X)

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
# Save the dataset

# %%
df_processed = pd.DataFrame({ 'X': X, 'y': y })
df_processed

# %%
df_processed.to_json(AFTER_CUTOFF_PATH, orient='records', lines=True)