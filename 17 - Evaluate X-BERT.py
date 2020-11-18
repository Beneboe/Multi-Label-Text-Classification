# %%
import matplotlib.pyplot as plt
import scipy.sparse as sp
import utils.metrics as mt
import utils.dataset as ds
import utils.storage as st
import numpy as np
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
# ## Calculate the micro and macro f1score

# %%
labels = ds.amazoncat13k_top10_labels
mapped_labels = [to_xbert_id(label) for label in labels]

xbert_y_predict = y_predict[:, mapped_labels].toarray()
xbert_y_expect = y_test[:, labels].toarray()

# %%
# The macro and micro metrics need the threshold applied
xbert_y_predict_bin = mt.apply_threshold(xbert_y_predict)
macro = mt.macro_f1score(xbert_y_predict_bin, xbert_y_expect)
micro = mt.micro_f1score(xbert_y_predict_bin, xbert_y_expect)

print(f'Macro f1score {macro}')
print(f'Micro f1score {micro}')

# %%
ova_y_predict = np.array([st.load_prediction(label, '50%positive') for label in labels]).T
# Cannot use the same y_expect for xbert and ova because the test set was shuffled when the prediction for ova was
# calculated and therefore the same shuffling needs to apply to y_expect
ova_y_expect = np.array([ds.get_dataset(X_test, y_test, label)[1] for label in labels]).T

# %% [markdown]
# ## Create the ROC curve

# %%
fpr, tpr = dict(), dict()
fpr['xbert'], tpr['xbert'], _ = roc_curve(xbert_y_expect.ravel(), xbert_y_predict.ravel())
fpr['ova'], tpr['ova'], _ = roc_curve(ova_y_expect.ravel(), ova_y_predict.ravel())

plt.figure()
lw = 2
plt.plot(fpr['xbert'], tpr['xbert'], color='darkorange',
         lw=lw, label='ROC curve for X-BERT prediction')
plt.plot(fpr['ova'], tpr['ova'], color='aqua',
         lw=lw, label='ROC curve for OvA prediction')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.savefig(f'results/imgs/roc_curve.png', dpi=163)
plt.savefig(f'results/imgs/roc_curve.pdf')
plt.show()

# %%
