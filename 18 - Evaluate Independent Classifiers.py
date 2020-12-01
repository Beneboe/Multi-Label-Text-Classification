# %%
import utils.metrics as mt
import utils.storage as st
import utils.dataset as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it
from sklearn.metrics import roc_curve, auc

# %%
# load the test set
INPUT_LENGTH = 10
X_test, y_test = ds.import_amazoncat13k('tst', INPUT_LENGTH)

# %% [markdown]
# ## Load the prediction files


# %%
# Get label sets
l1 = ds.amazoncat13k_top10_labels
l2 = ds.amazoncat13k_threshold_labels

# Load expected
l1_ye = np.array([ds.get_dataset(X_test, y_test, label)[1] for label in l1]).T
l2_ye = np.array([ds.get_dataset(X_test, y_test, label)[1] for label in l2]).T

# Load predictions
l1_b_yp = np.array([st.load_prediction(label, '50%positive') for label in l1]).T
l1_ub_yp = np.array([st.load_prediction(label, 'unbalanced') for label in l1]).T

l2_b_yp = np.array([st.load_prediction(label, '50%positive') for label in l2]).T
l2_ub_yp = np.array([st.load_prediction(label, 'unbalanced') for label in l2]).T

# %% [markdown]
# ## Calculate the micro and macro f1score

# %%
# macro = mt.macro(mt.f1score)
# micro = mt.macro(mt.f1score)

macro = mt.macro_f1score
micro = mt.micro_f1score

f1scores = [
    (macro(l1_b_yp, l1_ye), micro(l1_b_yp, l1_ye)),
    (macro(l1_ub_yp, l1_ye), micro(l1_ub_yp, l1_ye)),
    (macro(l2_b_yp, l2_ye), micro(l2_b_yp, l2_ye)),
    (macro(l2_ub_yp, l2_ye), micro(l2_ub_yp, l2_ye)),
]

f1score_table = zip(it.product(['L1', 'L2'], ['balanced', 'unbalanced']), f1scores)
f1score_table = [u + v for u, v in f1score_table]

f1score_df = pd.DataFrame(f1score_table)
f1score_df.to_csv(f'results/imgs_data/f1scores.csv')
f1score_df

# %%
print('Labels in l1_b that always predict false')
print([l2[label] for label in range(len(l2)) if np.all(mt.apply_threshold(l1_b_yp[:, label]) == 0)])

print('Labels in l1_ub that always predict false')
print([l2[label] for label in range(len(l2)) if np.all(mt.apply_threshold(l1_ub_yp[:, label]) == 0)])

print('Labels in l2_b that always predict false')
print([l2[label] for label in range(len(l2)) if np.all(mt.apply_threshold(l2_b_yp[:, label]) == 0)])

print('Labels in l2_ub that always predict false')
print([l2[label] for label in range(len(l2)) if np.all(mt.apply_threshold(l2_ub_yp[:, label]) == 0)])

# %%
roc_curves = [
    roc_curve(l1_ye.ravel(), l1_b_yp.ravel()),
    roc_curve(l1_ye.ravel(), l1_ub_yp.ravel()),
    roc_curve(l2_ye.ravel(), l2_b_yp.ravel()),
    roc_curve(l2_ye.ravel(), l2_ub_yp.ravel()),
]

txts = [
    'L1 balanced',
    'L1 unbalanced',
    'L2 balanced',
    'L2 unbalanced',
]

aucs = [auc(fpr, tpr) for fpr, tpr, _ in roc_curves]

# fpr, tpr, _ = roc_curve(y_expected.ravel(), y_predict.ravel())

plt.figure()
lw = 2
for (fpr, tpr, _), txt, a in zip(roc_curves, txts, aucs):
    plt.plot(fpr, tpr, lw=lw, label=f'%s (area = %0.2f)' % (txt, a))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'results/imgs/micro_roc.png', dpi=163)
plt.savefig(f'results/imgs/micro_roc.pdf')
plt.show()

# %%
