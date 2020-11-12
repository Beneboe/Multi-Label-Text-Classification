# %%
import utils.metrics as mt
import utils.storage as st
import utils.dataset as ds
import numpy as np

# load the test set
INPUT_LENGTH = 10
X_test, y_test = ds.import_amazoncat13k('tst', INPUT_LENGTH)

# %% [markdown]
# ## Load the prediction files

# %%
labels = ds.amazoncat13k_top10_labels
y_predict = np.array([st.load_prediction(label, '50%positive') for label in labels]).T
y_expected = np.array([ds.get_dataset(X_test, y_test, label)[1] for label in labels]).T

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
