# %%
import utils.metrics as mt
import utils.storage as st
import utils.dataset as ds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

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
# ## Calculate the micro and macro f1score

# %%
macro = mt.macro_f1score(y_predict, y_expected)
micro = mt.micro_f1score(y_predict, y_expected)
print(f'Macro f1score {macro}')
print(f'Micro f1score {micro}')


# %%
fpr, tpr, _ = roc_curve(y_expected.ravel(), y_predict.ravel())

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
