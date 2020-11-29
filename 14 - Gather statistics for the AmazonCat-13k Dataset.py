# %%
from matplotlib import cm
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# %% [markdown]
# # Gather Statistics for the AmazonCat-13k Dataset

# %%
CLASS_COUNT = 13330
ADD_CONTENT = False

# %% [markdown]
# First, load the dataset.

# %%
df_raw = pd.read_json('datasets/AmazonCat-13K/trn.json', lines=True)
df_raw

# %%
X_raw = np.load(f'datasets/AmazonCat-13K/X.trn.raw.npy', allow_pickle=True)
Y_raw = sp.load_npz(f'datasets/AmazonCat-13K/Y.trn.raw.npz')

X_processed = np.load(f'datasets/AmazonCat-13K/X.trn.processed.npy', allow_pickle=True)
Y_processed = sp.load_npz(f'datasets/AmazonCat-13K/Y.trn.processed.npz')

# %%
label_text = [None] * 13_330
with open('datasets/AmazonCat-13K/Yf.txt', 'r') as f:
    for id, line in enumerate(f):
        line = line.strip()
        label_text[id] = line

# %% [markdown]
# The dataset has the fields: *uid*, *title*, *content*, *target_ind*, *target_rel*.
# Next, we can calculate the maximum and minimum inds.

# %%
max_ind = 0
min_ind = 2_147_483_647
for inds in df_raw['target_ind']:
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
vlen = np.vectorize(len)

# Samples per label
samples_per_label_raw = np.asarray(Y_raw.sum(0)).flatten()
samples_per_label_processed = np.asarray(Y_processed.sum(0)).flatten()

# Labels per sample
labels_per_sample_raw = np.asarray(Y_raw.sum(1)).flatten()
labels_per_sample_processed = np.asarray(Y_processed.sum(1)).flatten()

# Tokens per sample
tokens_raw = vlen(X_raw)
tokens_processed = vlen(X_processed)

# %% [markdown]
# Create a boxplot for the frequencies.

# %%
fig1, ax1 = plt.subplots()
ax1.set_title('Samples per Label Boxplot')
ax1.set_yscale('log')
ax1.set_xlabel('Label (sorted by decreasing occurence)')
ax1.set_ylabel('Number of Samples')
ax1.boxplot(samples_per_label_raw)
fig1.savefig(f'datasets/AmazonCat-13K/stats/boxplot.png', dpi=163)
fig1.savefig(f'datasets/AmazonCat-13K/stats/boxplot.pdf')

# %% [markdown]
# Create a histogram for the frequencies.

# %%
# Constants
sorted = np.sort(samples_per_label_raw)
samples_per_label_raw_mean = np.mean(sorted)
samples_per_label_raw_median = np.median(sorted)

fig2, ax2 = plt.subplots()
ax2.set_title('Samples per Label Histogram')
ax2.set_yscale('log')
ax2.set_xlabel('Label (sorted by decreasing occurence)')
ax2.set_ylabel('Number of Samples')
ax2.hlines(samples_per_label_raw_mean, 0, 1, transform=ax2.get_yaxis_transform(), color='blue', linestyles='dashed')
ax2.hlines(samples_per_label_raw_median, 0, 1, transform=ax2.get_yaxis_transform(), color='green', linestyles='dashed')
ax2.legend(['mean', 'median'], loc='upper right')
ax2.plot(np.arange(sorted.shape[0]), sorted[::-1], color='k')

plt.grid()
fig2.savefig(f'datasets/AmazonCat-13K/stats/histogram.png', dpi=163)
fig2.savefig(f'datasets/AmazonCat-13K/stats/histogram.pdf')

# %% [markdown]
# Next, we can calculate the statistics for class frequencies, title char lengths, content char lengths, and instance class counts.

# %%
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

# %%
stats = [
    ('title char lengths', get_stats(vlen(df_raw['title']))),
    ('content char lengths', get_stats(vlen(df_raw['content']))),

    ('raw samples per label', get_stats(samples_per_label_raw)),
    ('raw labels per sample', get_stats(labels_per_sample_raw)),
    ('raw token lengths', get_stats(tokens_raw)),

    ('processed samples per label', get_stats(samples_per_label_processed)),
    ('processed labels per sample', get_stats(labels_per_sample_processed)),
    ('processed token lengths', get_stats(tokens_processed)),
]

ds_stats_index, ds_stats = zip(*stats)
stats_df = pd.DataFrame(ds_stats, index=ds_stats_index)
stats_df.to_csv(f'datasets/AmazonCat-13K/stats/all_stats.csv')

# %% [markdown]
# Calculate the labels below a set of frequency thresholds

# %%
id_text_map = [None] * 13_330
with open('datasets/AmazonCat-13K/Yf.txt', 'r') as f:
    for id, line in enumerate(f):
        line = line.strip()
        id_text_map[id] = line

def label_text(id):
    return id_text_map[id]

# %%
freqs_args = np.argsort(samples_per_label_raw)

def freqs_args_below(threshold):
    # Index before which all indexes point to frequences below the threshold
    i = np.searchsorted(samples_per_label_raw, threshold, side='right', sorter=freqs_args)
    # return freqs_args[i-1:0:-1]
    return freqs_args[i-1]

vfreqs_args_below = np.vectorize(freqs_args_below)

# %%
thresholds = [10, 100, 1_000, 10_000, 100_000]
labels = vfreqs_args_below(thresholds)

threshold_labels = pd.DataFrame(
    { 'label': labels, 'text': [label_text(label) for label in labels], 'threshold': thresholds, 'frequency': samples_per_label_raw[labels] }
)
threshold_labels.to_csv(f'datasets/AmazonCat-13K/stats/topbelowk.csv')

# %% [markdown]
# Calculate the top 10 most frequent labels
freqs_args_before = np.argsort(samples_per_label_raw)

top10freq_id = freqs_args_before[:-(10 + 1):-1]

top10freqs = pd.DataFrame(
    { 'label': top10freq_id, 'text': [label_text(label) for label in top10freq_id], 'frequency': samples_per_label_raw[top10freq_id] }
)
top10freqs.to_csv(f'datasets/AmazonCat-13K/stats/top10.csv')

#%%
def imbalance_ratio(label):
    y = Y_raw[:, label].toarray().flatten()

    assert len(y.shape) == 1

    y_neg = np.count_nonzero(y == 0)
    y_pos = np.count_nonzero(y == 1)

    assert (y_neg + y_pos) == y.shape[0]

    return float(max(y_neg, y_pos))/float(min(y_neg, y_pos))

# %% Find out the imbalance ratio for 'natural history'
imbalance_ratio(8035)

# %% Find out the imbalance ratio for 'books'
imbalance_ratio(1471)

# %%
coo = Y_raw.astype('float32')
coo = coo.transpose().dot(coo)
dia = coo.diagonal()

# row: label, col: probability distribution of row
ncoo = (coo.transpose() / dia).transpose()

# %%
def sub_classes(label):
    m = np.squeeze(np.asarray(ncoo[:, label] == 1.0))
    labels = np.arange(ncoo.shape[0])
    return [label_text[l] for l in labels[m]]

def co_occurrences(label):
    m = np.squeeze(np.asarray(ncoo[label] == 1.0))
    labels = np.arange(ncoo.shape[0])
    return [label_text[l] for l in labels[m]]

# %% Write the subclasses of books
books_sub_classes = sub_classes(1471)
with open('datasets/AmazonCat-13K/subclasses/books.txt', 'w') as f:
    f.writelines((label + "\n" for label in books_sub_classes))

# %% Write the subclasses of music
music_sub_classes = sub_classes(7961)

with open('datasets/AmazonCat-13K/subclasses/music.txt', 'w') as f:
    f.writelines((label + "\n" for label in music_sub_classes))

# %% Write the subclasses of movies & tv
moviestv_sub_classes = sub_classes(7892)

with open('datasets/AmazonCat-13K/subclasses/moviestv.txt', 'w') as f:
    f.writelines((label + "\n" for label in moviestv_sub_classes))

# %% Is 'natural history' a subclass of 'books'
ncoo[8035, 1471] == 1.0

# %% Get the co-occurrences of natural history
co_occurrences(8035)

# %%
ncoo_sum = ncoo.sum(1)
ncoo_sum = np.squeeze(np.asarray(ncoo_sum))

# %%
plt.title('Co-Occurrences by Label Frequency')
plt.xlabel('# of samples per label')
plt.ylabel('summed co-occurences')
plt.scatter(samples_per_label_raw, ncoo_sum)
plt.xscale('log')
plt.show()

# %%
plt.hexbin(np.log10(samples_per_label_raw), ncoo_sum, gridsize=20)
plt.show()
