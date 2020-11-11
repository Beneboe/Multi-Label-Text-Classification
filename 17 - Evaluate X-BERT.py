# %%
from utils.dataset import get_dataset, import_amazoncat13k
import scipy.sparse as smat
import utils.metrics as mt
import numpy as np

# load the test set
INPUT_LENGTH = 10
X_test, y_test = import_amazoncat13k('tst', INPUT_LENGTH)

# Top 10 most frequent labels ordered from most to least frequent
# Order: label, frequency
top10_label_data = [
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
top10_labels, _ = zip(*top10_label_data)

# Labels just below certain frequency thresholds
# Order: threshold, label, frequency
threshold_data = [
    (50,6554,50),
    (100,4949,100),
    (1000,7393,996),
    (10000,84,9976),
    (50000,9202,48521),
    # (100000,7083,96012), # duplicate
]
_, threshold_labels, _ = zip(*threshold_data)

# %% [markdown]
# ## Load the prediction file

# %%
y_predict = smat.load_npz('results/xbert/elmo-a0-s0/tst.pred.xbert.npz')

# %% [markdown]
# ##  Map from original label index to X-BERT label index

# %%

# Maps label text -> X-BERT label id
label_map_xbert = {}
with open('datasets/AmazonCat-13K/xbert_mapping.txt', 'r') as f:
    for id, line in enumerate(f):
        line = line.strip()
        tpos = line.index('\t')
        id_text = line[0:tpos]
        label_text = line[(tpos+1):]
        label_map_xbert[label_text] = id

# Maps original label id -> label text
label_id_map_original = [None] * 13_330
with open('datasets/AmazonCat-13K/Yf.txt', 'r') as f:
    for id, line in enumerate(f):
        line = line.strip()
        label_id_map_original[id] = line

def xbert_map(og_label_id):
    label_text = label_id_map_original[og_label_id]
    return label_map_xbert[label_text]

# %% [markdown]
# ##  Calculate the metrics per label

# %%
def metrics(label):
    _, yi_expected = get_dataset(X_test, y_test, label)
    yi_predict = y_predict[:, xbert_map(label)]
    return mt.all_metrics(yi_predict, yi_expected)

# %% [markdown]
# ## Calculate the micro and macro f1 measure

# %%
lbs = [xbert_map(label) for label in top10_labels]
ys_predict = y_predict[:, lbs]
ys_expected = y_test[:, top10_labels]
macro = mt.macro_f1measure(ys_predict, ys_expected)
micro = mt.micro_f1measure(ys_predict, ys_expected)

# %%
