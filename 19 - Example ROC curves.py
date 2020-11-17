# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from numpy.random import default_rng

# %%
rng = default_rng(42)

good_expect_mask = rng.uniform(size=(100000,)) > 0.9
good_expect = good_expect_mask.astype('int8')
good_expect_nonzero_count = np.count_nonzero(good_expect)
good_expect_zero_count = good_expect.shape[0] - good_expect_nonzero_count

good_predict = np.ones(good_expect.shape)
good_predict[good_expect_mask] = rng.normal(0.68, 0.1, good_expect_nonzero_count)
good_predict[~good_expect_mask] = rng.normal(0.42, 0.1, good_expect_zero_count)

# %%
fpr, tpr = dict(), dict()
# fpr['best'], tpr['best'], _ = roc_curve()
fpr['good'], tpr['good'], _ = roc_curve(good_expect, good_predict)
# fpr['bad'], tpr['bad'], _ = roc_curve()

plt.figure()
lw = 2
# plt.plot(fpr['best'], tpr['best'], color='darkorange', lw=lw, label='')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random classifier')
plt.plot(fpr['good'], tpr['good'], color='darkgreen', lw=lw, label='Good classifier', clip_on=False, zorder = 1000)
plt.plot([0, 0, 1], [0, 1, 1], color='darkorange', lw=lw, label='Ideal classifier', clip_on=False, zorder = 1000)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.savefig(f'results/imgs/roc_curve_example.png', dpi=163)
plt.savefig(f'results/imgs/roc_curve_example.pdf')
plt.show()

# %%
