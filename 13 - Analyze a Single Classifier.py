# %%
from utils.dataset import import_amazoncat13k, get_dataset
from utils.plots import plot_confusion
from utils.models import random_prediction, prediction_threshold
import utils.metrics as mt
import matplotlib.pyplot as plt
import utils.storage as st
import numpy as np
import pandas as pd

# %%
INPUT_LENGTH = 10
CLASS = 8842

X_test, y_test = import_amazoncat13k('tst', INPUT_LENGTH)
Xi, yi_expected = get_dataset(X_test, y_test, CLASS)

# %% [markdown]
# Create random predictions

# %%
yi_predict = random_prediction(Xi, 0.5)
st.save_prediction(8842, '50%positive_random', yi_predict)

yi_predict = random_prediction(Xi, prediction_threshold(yi_expected))
st.save_prediction(8842, 'unbalanced_random', yi_predict)

# %%
keras_type_names = ['50%positive', '20%positive', '10%positive', 'unbalanced']
random_type_names = ['50%positive_random', 'unbalanced_random']
class_types = keras_type_names + random_type_names
class_names = [st.get_name(CLASS, n) for n in class_types]

def predictions():
    for name in class_types:
        yield st.load_prediction(CLASS, name)

def confusions():
    for yi_predict in predictions():
        yield mt.get_confusion(yi_predict, yi_expected)

def metrics():
    for yi_predict in predictions():
        yield mt.all_metrics(yi_predict, yi_expected)

# %% [markdown]
# Create the confusion matrix for each classifier

# %%
for name, cm in zip(class_names, confusions()):
    plot_confusion(cm, name)
    plt.tight_layout()
    plt.savefig(f'results/imgs/classifier_{name}_confusion.png', dpi=163)
    plt.savefig(f'results/imgs/classifier_{name}_confusion.pdf')
    plt.show()

# %%
nrows, ncols = 2, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
for i, (name, cm) in enumerate(zip(class_names, confusions())):
    ax = axs[i // ncols, i % ncols]
    plot_confusion(cm, name, ax)

plt.tight_layout()
plt.savefig(f'results/imgs/classifier_{CLASS}_all_confusion.png', dpi=163)
plt.savefig(f'results/imgs/classifier_{CLASS}_all_confusion.pdf')

# %% [markdown]
# Create metrics comparison
metric_comparison = pd.DataFrame(
    list(metrics()),
    index=pd.Index([name for name in class_types]))

metric_comparison.to_csv(f'results/imgs_data/classifier_{CLASS}_all_metrics.csv')
metric_comparison

# %%
