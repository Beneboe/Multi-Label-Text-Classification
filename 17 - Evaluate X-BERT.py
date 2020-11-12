# %%
import matplotlib.pyplot as plt
import scipy.sparse as sp
import utils.metrics as mt
import utils.dataset as ds
import numpy as np
from tensorflow.python.framework import sparse_tensor
from sklearn.metrics import roc_curve

# %%

# load the test set
INPUT_LENGTH = 10
X_test, y_test = ds.import_amazoncat13k('tst', INPUT_LENGTH)

# %% [markdown]
# ## Load the prediction file

# %%
y_predict = sp.load_npz('results/xbert/elmo-a0-s0/tst.pred.xbert.npz')

# %% [markdown]
# ##  Map from original label index to X-BERT label index

# %%

# Maps label text -> X-BERT label id
xbert_label_map = {}
with open('datasets/AmazonCat-13K/xbert_mapping.txt', 'r') as f:
    for id, line in enumerate(f):
        line = line.strip()
        tpos = line.index('\t')
        id_text = line[0:tpos]
        label_text = line[(tpos+1):]
        xbert_label_map[label_text] = id

# Maps original label id -> label text
original_id_map = [None] * 13_330
with open('datasets/AmazonCat-13K/Yf.txt', 'r') as f:
    for id, line in enumerate(f):
        line = line.strip()
        original_id_map[id] = line

# Maps original label id -> X-BERT label id
def to_xbert_id(og_label_id):
    label_text = original_id_map[og_label_id]
    return xbert_label_map[label_text]

# %% [markdown]
# ##  Calculate the metrics per label

# %%
def metrics(label):
    _, yi_expected = ds.get_dataset(X_test, y_test, label)
    yi_predict = y_predict[:, to_xbert_id(label)]
    return mt.all_metrics(yi_predict, yi_expected)

# %% [markdown]
# ## Calculate the micro and macro f1 measure

# %%
mapped_labels = [to_xbert_id(label) for label in ds.amazoncat13k_top10_labels]

ys_predict = y_predict[:, mapped_labels].toarray()
ys_expected = y_test[:, ds.amazoncat13k_top10_labels].toarray()

#%%
# The macro and micro metrics need the threshold applied
ys_predict_bin = mt.apply_threshold(ys_predict)
macro = mt.macro_f1measure(ys_predict_bin, ys_expected)
micro = mt.micro_f1measure(ys_predict_bin, ys_expected)

print(f'Macro f1 measure {macro}')
print(f'Micro f1 measure {micro}')

# %% [markdown]
# ## Create the ROC curve

# %%
fpr, tpr, _ = roc_curve(ys_expected.ravel(), ys_predict.ravel())

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# %%
