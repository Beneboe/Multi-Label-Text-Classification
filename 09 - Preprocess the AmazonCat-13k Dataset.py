# %%
from utils.dataset import get_stats, class_frequencies
from utils.text_preprocessing import preprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Preprocess the AmazonCat-13k Dataset

# %%
CLASS_COUNT = 13330
DATASET_TYPE = 'trn'
# CUTOFF is inclusive
CUTOFF = 0 if DATASET_TYPE == 'trn' else 0
INPUT_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.json'
OUTPUT_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.processed.json'

# %% [markdown]
# First, load the dataset.

# %%
df = pd.read_json(INPUT_PATH, lines=True)
df

# %% [markdown]
# The dataset has the fields: *uid*, *title*, *content*, *target_ind*, *target_rel*.
# Next, we can calculate the maximum and minimum inds.

# %%
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
# Next, we can calculate the statistics for class frequencies, title char lengths, content char lengths, and instance class counts.

# %%
vlen = np.vectorize(len)

ds_stats_index = []
ds_stats = []

freqs = class_frequencies(CLASS_COUNT, df['target_ind'])
ds_stats_index.append('class frequencies')
ds_stats.append(get_stats(freqs))

ds_stats_index.append('title char lengths')
ds_stats.append(get_stats(vlen(df['title'])))

ds_stats_index.append('content char lengths')
ds_stats.append(get_stats(vlen(df['content'])))

ds_stats_index.append('instance class count')
ds_stats.append(get_stats(df['target_ind'].map(len)))

pd.DataFrame(ds_stats, index=ds_stats_index)

# %% [markdown]
# Create a boxplot for the frequencies.

# %%
fig1, ax1 = plt.subplots()
ax1.set_title('Class frequencies')
ax1.boxplot(freqs)
plt.savefig(f'datasets/imgs/AmazonCat-13K_{DATASET_TYPE}_boxplot.png', dpi=163)

# %% [markdown]
# Create a historgram for the frequencies.

# %%
sorted = np.sort(freqs)

fig2, ax2 = plt.subplots()
ax2.set_title('Class frequencies')
plt.yscale('log')
ax2.plot(np.arange(sorted.shape[0]), sorted)
plt.savefig(f'datasets/imgs/AmazonCat-13K_{DATASET_TYPE}_histogram.png', dpi=163)

# %% [markdown]
# ## Preprocess the Dataset

# %%
X = preprocess(df['title'])
y = df['target_ind']

# %% [markdown]
# Next, we can calculate the statistics for the token lengths.

# %%
token_lens = vlen(X)

ds_stats_index.append('token lengths')
ds_stats.append(get_stats(token_lens))

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
ds_stats.append(get_stats(class_frequencies(CLASS_COUNT, y)))

ds_stats_index.append('instance class count (after cutoff)')
ds_stats.append(get_stats(y.map(len)))

ds_stats_index.append('token lengths (after cutoff)')
ds_stats.append(get_stats(token_lens))

stats_df = pd.DataFrame(ds_stats, index=ds_stats_index)
stats_df.to_csv(f'datasets/stats/AmazonCat-13K_{DATASET_TYPE}.csv')

# %%
df_processed = pd.DataFrame({ 'X': X, 'y': y })
df_processed

# %% [markdown]
# Save the dataset

# %%
df_processed.to_json(OUTPUT_PATH, orient='records', lines=True)