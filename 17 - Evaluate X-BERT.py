# %%
import matplotlib.pyplot as plt
import scipy.sparse as sp
import utils.metrics as mt
import utils.dataset as ds
import utils.storage as st
import numpy as np
from sklearn.metrics import roc_curve, auc

# %%
# load the test set
INPUT_LENGTH = 10
X_test, y_test = ds.import_amazoncat13k('tst', INPUT_LENGTH)

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
# ## Load the prediction file

# %%
xbert_yp = sp.load_npz('results/xbert/elmo-a0-s0/tst.pred.xbert.npz')

# %% X-BERT predictions
l1 = ds.amazoncat13k_top10_labels
l2 = ds.amazoncat13k_threshold_labels

l1_xbert = [to_xbert_id(label) for label in l1]
l2_xbert = [to_xbert_id(label) for label in l2]

l1_xbert_yp = xbert_yp[:, l1_xbert].toarray()
l1_xbert_ye = y_test[:, l1].toarray()

l2_xbert_yp = xbert_yp[:, l2_xbert].toarray()
l2_xbert_ye = y_test[:, l2].toarray()

# %% CGA predictions

# Cannot use the same y_expect for xbert and ova because the test set was shuffled when the prediction for ova was
# calculated and therefore the same shuffling needs to apply to y_expect
l1_cga_yp = np.array([st.load_prediction(label, 'unbalanced') for label in l1]).T
l2_cga_yp = np.array([st.load_prediction(label, 'unbalanced') for label in l2]).T

l1_cga_ye = np.array([ds.get_dataset(X_test, y_test, label)[1] for label in l1]).T
l2_cga_ye = np.array([ds.get_dataset(X_test, y_test, label)[1] for label in l2]).T

# %% [markdown]
# ## Create the ROC curve

# %%
curve_names = ['L1 X-BERT', 'L1 Coarse', 'L2 X-BERT', 'L2 Coarse']
curves = [
    roc_curve(l1_xbert_ye.ravel(), l1_xbert_yp.ravel()),
    roc_curve(l1_cga_ye.ravel(), l1_cga_yp.ravel()),
    roc_curve(l2_xbert_ye.ravel(), l2_xbert_yp.ravel()),
    roc_curve(l2_cga_ye.ravel(), l2_cga_yp.ravel()),
]
areas = [auc(fpr, tpr) for fpr, tpr, _ in curves]

plt.figure()
lw = 2
for name, (fpr, tpr, _), area in zip(curve_names, curves, areas):
    plt.plot(fpr, tpr, lw=lw, label=f'%s (area = %0.2f)' % (name, area))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(f'results/imgs/xbert_curve.png', dpi=163)
plt.savefig(f'results/imgs/xbert_curve.pdf')
plt.show()

# %% [markdown]
# ## Calculate the micro and macro f1score

# %%
# The macro and micro metrics need the threshold applied
xbert_yp_bin = mt.apply_threshold(l1_xbert_yp)
macro = mt.macro_f1score(xbert_yp_bin, l1_xbert_ye)
micro = mt.micro_f1score(xbert_yp_bin, l1_xbert_ye)

print(f'Macro f1score {macro}')
print(f'Micro f1score {micro}')
