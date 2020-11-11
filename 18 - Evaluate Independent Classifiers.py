# %%
from utils.dataset import get_dataset, import_dataset
import utils.metrics as mt
import utils.storage as st
import numpy as np

# load the test set
INPUT_LENGTH = 10
X_test, y_test = import_dataset('datasets/AmazonCat-13K/tst.processed.json', INPUT_LENGTH)

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
# ## Load the prediction files

# %%
labels = top10_labels
y_predict = np.array([st.load_prediction(label, '50%positive') for label in labels]).T
y_expected = np.array([get_dataset(X_test, y_test, label)[1] for label in labels]).T

# %% [markdown]
# ##  Calculate the metrics per label

# %%
def metrics(label_id):
    yi_expected = y_expected[:, label_id]
    yi_predict = y_predict[:, label_id]
    return mt.all_metrics(yi_predict, yi_expected)

# %% [markdown]
# ## Calculate the micro and macro f1 measure

# %%
macro = mt.macro_f1measure(y_predict, y_expected)
micro = mt.micro_f1measure(y_predict, y_expected)
print(f'Macro f1 measure {macro}')
print(f'Micro f1 measure {micro}')

# %%
