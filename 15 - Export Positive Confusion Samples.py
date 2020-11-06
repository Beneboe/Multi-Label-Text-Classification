from utils.models import Weighted50Classifier
from utils.text_preprocessing import from_token_ids
from utils.dataset import import_dataset, get_dataset
import numpy as np

# %%
INPUT_LENGTH = 10
# CLASS_COUNT = 30
CLASS = 8842

# %%
X_train, y_train = import_dataset('datasets/AmazonCat-13K/trn.processed.json', INPUT_LENGTH)
X_test, y_test = import_dataset('datasets/AmazonCat-13K/tst.processed.json', INPUT_LENGTH)

# %%
Xi, yi_expected = get_dataset(X_test, y_test, CLASS)

# %% [markdown]
# Retrieve and save the false positives

# %%
balanced = Weighted50Classifier(CLASS)
balanced.load_weights()
yi_predict = balanced.get_prediction(Xi)
fp_mask = np.logical_and(yi_predict == 1, yi_expected == 0)
Xi_fp = np.apply_along_axis(from_token_ids, 1, Xi[fp_mask])
np.savetxt(f'results/{CLASS}_balanced_false_positives.csv', Xi_fp, delimiter=',', fmt='%s')

# %% [markdown]
# Retrieve and save the true positives

# %%
balanced = Weighted50Classifier(CLASS)
balanced.load_weights()
yi_predict = balanced.get_prediction(Xi)
fp_mask = np.logical_and(yi_predict == 1, yi_expected == 1)
Xi_fp = np.apply_along_axis(from_token_ids, 1, Xi[fp_mask])
np.savetxt(f'results/{CLASS}_balanced_true_positives.csv', Xi_fp, delimiter=',', fmt='%s')
