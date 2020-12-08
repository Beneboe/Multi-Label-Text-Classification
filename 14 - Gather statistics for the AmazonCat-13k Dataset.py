# %%
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
with open('datasets/AmazonCat-13K/Yf.txt', 'r') as sf:
    for id, line in enumerate(sf):
        line = line.strip()
        label_text[id] = line

# %% [markdown]
# The dataset has the fields: *uid*, *title*, *content*, *target_ind*, *target_rel*.

# %%
vlen = np.vectorize(len)

# Samples per label (sf - sample frequency)
sf = np.asarray(Y_raw.sum(0)).flatten()
sf_processed = np.asarray(Y_processed.sum(0)).flatten()

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
ax1.boxplot(sf)
fig1.savefig(f'datasets/AmazonCat-13K/stats/boxplot.png', dpi=163)
fig1.savefig(f'datasets/AmazonCat-13K/stats/boxplot.pdf')

# %% [markdown]
# Create a histogram for the frequencies.

# %%
# Constants
sf_sorted = np.sort(sf)
samples_per_label_raw_mean = np.mean(sf_sorted)
samples_per_label_raw_median = np.median(sf_sorted)

plt.title('Samples per Label Histogram')
plt.yscale('log')
plt.xlabel('Label (sorted by decreasing occurrence)')
plt.ylabel('Number of Samples')
plt.axhline(samples_per_label_raw_mean, color='blue', linestyle='dashed', label='mean')
plt.axhline(samples_per_label_raw_median,color='green', linestyle='dashed', label='median')
plt.legend(loc='lower right')
plt.plot(np.arange(sf_sorted.shape[0]), sf_sorted, color='k')
plt.grid()

plt.savefig(f'datasets/AmazonCat-13K/stats/histogram.png', dpi=163)
plt.savefig(f'datasets/AmazonCat-13K/stats/histogram.pdf')
plt.show()

# %%
hist, bins = np.histogram(sf, bins=50)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
plt.hist(sf, bins=logbins)
plt.xscale('log')
plt.title('Samples per Label Distribution')
plt.xlabel('Number of Samples')
plt.ylabel('Number of Number of Samples')
plt.savefig(f'datasets/AmazonCat-13K/stats/distribution.png', dpi=163)
plt.savefig(f'datasets/AmazonCat-13K/stats/distribution.pdf')
plt.show()

# %% Combined
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(6.4 * 1.5, 4.8))
ax1.set_title('Samples per Label Histogram')
ax1.set_yscale('log')
ax1.set_xlabel('Label (sorted by decreasing occurrence)')
ax1.set_ylabel('Number of Samples')
ax1.axhline(samples_per_label_raw_mean, color='blue', linestyle='dashed', label='mean')
ax1.axhline(samples_per_label_raw_median,color='green', linestyle='dashed', label='median')
ax1.legend(loc='lower right')
ax1.plot(np.arange(sf_sorted.shape[0]), sf_sorted, color='k')
ax1.grid()


hist, bins = np.histogram(sf, bins=50)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
ax2.hist(sf, bins=logbins)
ax2.set_xscale('log')
ax2.set_title('Samples per Label Distribution')
ax2.set_xlabel('Number of Samples')
ax2.set_ylabel('Number of Number of Samples')

fig.savefig(f'datasets/AmazonCat-13K/stats/histdist.png', dpi=163)
fig.savefig(f'datasets/AmazonCat-13K/stats/histdist.pdf')
fig.show()


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

    ('raw samples per label', get_stats(sf)),
    ('raw labels per sample', get_stats(labels_per_sample_raw)),
    ('raw token lengths', get_stats(tokens_raw)),

    ('processed samples per label', get_stats(sf_processed)),
    ('processed labels per sample', get_stats(labels_per_sample_processed)),
    ('processed token lengths', get_stats(tokens_processed)),
]

ds_stats_index, ds_stats = zip(*stats)
stats_df = pd.DataFrame(ds_stats, index=ds_stats_index)
stats_df.to_csv(f'datasets/AmazonCat-13K/stats/all_stats.csv')

# %% [markdown]
# Calculate the top 10 most frequent labels
sf_args = np.argsort(sf)

top10 = sf_args[:-(10 + 1):-1]

# %% Save top 10 most frequent labels
top10freqs = pd.DataFrame(
    { 'label': top10, 'text': [label_text[label] for label in top10], 'frequency': sf[top10] }
)
top10freqs.to_csv(f'datasets/AmazonCat-13K/stats/top10.csv')

# %% In how many samples are top 10 labels?
rows = Y_raw[:, top10].nonzero()[0]
row_count = np.unique(rows).shape[0]
row_count / Y_raw.shape[0]

# %% [markdown]
# Calculate the labels below a set of frequency thresholds
sf_args = np.argsort(sf)

def freqs_args_below(threshold):
    # Index before which all indexes point to frequences below the threshold
    i = np.searchsorted(sf, threshold, side='right', sorter=sf_args)
    # return freqs_args[i-1:0:-1]
    return sf_args[i-1]

vfreqs_args_below = np.vectorize(freqs_args_below)
thresholds = [10, 100, 1_000, 10_000, 100_000]
l2 = vfreqs_args_below(thresholds)

# %% Save the labels
threshold_labels = pd.DataFrame(
    { 'label': l2, 'text': [label_text[label] for label in l2], 'threshold': thresholds, 'frequency': sf[l2] }
)
threshold_labels.to_csv(f'datasets/AmazonCat-13K/stats/topbelowk.csv')

# %% In how many samples the labels?
rows = Y_raw[:, l2].nonzero()[0]
row_count = np.unique(rows).shape[0]
row_count / Y_raw.shape[0]

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

# %% Find out the imbalance ratio for 'personal care'
imbalance_ratio(8842)

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
with open('datasets/AmazonCat-13K/stats/books.txt', 'w') as sf:
    sf.writelines((label + "\n" for label in books_sub_classes))

# %% Write the subclasses of music
music_sub_classes = sub_classes(7961)

with open('datasets/AmazonCat-13K/stats/music.txt', 'w') as sf:
    sf.writelines((label + "\n" for label in music_sub_classes))

# %% Write the subclasses of movies & tv
moviestv_sub_classes = sub_classes(7892)

with open('datasets/AmazonCat-13K/stats/moviestv.txt', 'w') as sf:
    sf.writelines((label + "\n" for label in moviestv_sub_classes))

# %% Is 'natural history' a subclass of 'books'
ncoo[8035, 1471] == 1.0

# %% Get the co-occurrences of natural history
co_occurrences(8035)

# %%
ones_vec = np.ones(coo.shape[0])
ncoo_sum = ncoo.dot(ones_vec)
ncoo_sum = np.squeeze(np.asarray(ncoo_sum))

# %%
plt.title('Co-Occurrences by Label Frequency')
plt.xlabel('# of samples per label')
plt.ylabel('summed co-occurences')
plt.scatter(sf, ncoo_sum)
plt.xscale('log')
plt.show()

# %%
plt.hexbin(np.log10(sf), ncoo_sum, gridsize=30)
plt.show()

# %%
hist, xbins, ybins = np.histogram2d(sf, ncoo_sum, bins=50)
logxbins = np.logspace(np.log10(xbins[0]), np.log10(xbins[-1]), len(xbins))
plt.hist2d(sf, ncoo_sum, [logxbins,ybins])
plt.xscale('log')
plt.show()

# %%
