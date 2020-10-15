# %%
from utils.dataset import get_stats, class_frequencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Gather Statistics for the AmazonCat-13k Dataset

# %%
CLASS_COUNT = 13330
DATASET_TYPE = 'trn'
ADD_CONTENT = False
RAW_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.json'
BEFORE_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.before.json'
AFTER_PATH = f'datasets/AmazonCat-13K/{DATASET_TYPE}.processed.json'

# %% [markdown]
# First, load the dataset.

# %%
df_raw = pd.read_json(RAW_PATH, lines=True)
df_raw

# %%
df_before = pd.read_json(BEFORE_PATH, lines=True)
df_after = pd.read_json(AFTER_PATH, lines=True)

# %% [markdown]
# The dataset has the fields: *uid*, *title*, *content*, *target_ind*, *target_rel*.
# Next, we can calculate the maximum and minimum inds.

# %%
max_ind = 0
min_ind = 2_147_483_647
for inds in df_before['y']:
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

# %%
freqs_before = class_frequencies(CLASS_COUNT, df_before['y'])
freqs_after = class_frequencies(CLASS_COUNT, df_after['y'])

# %% [markdown]
# Create a boxplot for the frequencies.

# %%
fig1, ax1 = plt.subplots()
ax1.set_title('Class frequencies')
ax1.boxplot(freqs_before)
plt.savefig(f'datasets/imgs/AmazonCat-13K_{DATASET_TYPE}_boxplot.png', dpi=163)

# %% [markdown]
# Create a historgram for the frequencies.

# %%
sorted = np.sort(freqs_before)
freqs_mean = np.mean(sorted)
freqs_median = np.median(sorted)

fig2, ax2 = plt.subplots()
ax2.set_title('Label frequencies')
ax2.set_yscale('log')
ax2.set_xlabel('Labels (sorted by occurence)')
ax2.set_ylabel('Occurences')
ax2.hlines(freqs_mean, 0, 1, transform=ax2.get_yaxis_transform(), color='tab:brown', linestyles='dashed')
ax2.hlines(freqs_median, 0, 1, transform=ax2.get_yaxis_transform(), color='tab:olive', linestyles='dashed')
# ax2.text(0, freqs_median, 'median', ha='left', va='bottom')
ax2.legend(['mean', 'median'], loc='lower right')
ax2.plot(np.arange(sorted.shape[0]), sorted)
fig2.savefig(f'datasets/imgs/AmazonCat-13K_{DATASET_TYPE}_histogram.png', dpi=163)

# %% [markdown]
# Next, we can calculate the statistics for class frequencies, title char lengths, content char lengths, and instance class counts.

# %%
vlen = np.vectorize(len)

stats = [
    ('title char lengths', get_stats(vlen(df_raw['title']))),
    ('content char lengths', get_stats(vlen(df_raw['content']))),

    ('class frequencies', get_stats(freqs_before)),
    ('instance class count', get_stats(df_before['y'].map(len))),
    ('token lengths', get_stats(vlen(df_before['X']))),

    ('class frequencies (after cutoff)', get_stats(freqs_after)),
    ('instance class count (after cutoff)', get_stats(df_after['y'].map(len))),
    ('token lengths (after cutoff)', get_stats(vlen(df_after['X']))),
]

ds_stats_index, ds_stats = zip(*stats)
stats_df = pd.DataFrame(ds_stats, index=ds_stats_index)
stats_df.to_csv(f'datasets/stats/AmazonCat-13K_{DATASET_TYPE}.csv')


# %% [markdown]
# Calculate the labels below a set of frequency thresholds

# %%
freqs_args = np.argsort(freqs_after)

def freqs_args_below(threshold):
    # Index before which all indexes point to frequences below the threshold
    i = np.searchsorted(freqs_after, threshold, side='right', sorter=freqs_args)
    # return freqs_args[i-1:0:-1]
    return freqs_args[i-1]

vfreqs_args_below = np.vectorize(freqs_args_below)

# %%
thresholds = [50, 100, 1_000, 10_000, 50_000, 100_000]
labels = vfreqs_args_below(thresholds)

threshold_labels = pd.DataFrame(
    { 'label': labels, 'frequency': freqs_after[labels] },
    index=pd.Index(thresholds, name='threshold'))
threshold_labels.to_csv(f'datasets/stats/AmazonCat-13K_{DATASET_TYPE}_thresholds.csv')
