# %%
from utils.dataset import import_dataset, get_dataset
from utils.plots import plot_confusion, plot_history
from utils.models import Weighted50Classifier, UnbalancedClassifier, Weighted10Classifier, Weighted20Classifier, Weighted50RandomClassifier, UnbalancedRandomClassifier, classifiers, keras_classifiers, random_classifiers
from utils.text_preprocessing import from_token_ids
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# %%
INPUT_LENGTH = 10
# CLASS_COUNT = 30
CLASS = 8842

# %%
X_train, y_train = import_dataset('datasets/AmazonCat-13K/trn.processed.json', INPUT_LENGTH)
X_test, y_test = import_dataset('datasets/AmazonCat-13K/tst.processed.json', INPUT_LENGTH)

# %%
Xi, yi_expected = get_dataset(X_test, y_test, CLASS)

# %%
def get_title(classifier):
    return ' '.join(word.capitalize() for word in classifier.get_name().split('_'))

# %% [markdown]
# Plot the history diagrams

# %%
history_names = ['balanced', 'unbalanced', 'p20', 'p10']
for c in keras_classifiers(CLASS, skip_model=True):
# for name in history_names:
    history = None
    with open(f'results/history/{c.get_name()}.json', 'r') as fp:
        history = json.load(fp)
    plot_history(history, get_title(c))
    plt.tight_layout()
    plt.savefig(f'datasets/imgs/classifier_{c.get_name()}_history.png', dpi=163)
    plt.savefig(f'datasets/imgs/classifier_{c.get_name()}_history.svg')
    plt.show()

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

# %% [markdown]
# Create the confusion matrix for each classifier

# %%
for c in classifiers(CLASS):
    cm = c.get_confusion(Xi, yi_expected)
    plot_confusion(cm)
    plt.tight_layout()
    plt.savefig(f'datasets/imgs/classifier_{c.get_name()}_confusion.png', dpi=163)
    plt.savefig(f'datasets/imgs/classifier_{c.get_name()}_confusion.svg')
    plt.show()

# %%
nrows, ncols = 2, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(14,10))
for i, c in enumerate(classifiers(CLASS)):
    cm = c.get_confusion(Xi, yi_expected)
    ax = axs[i // ncols, i % ncols]
    plot_confusion(cm, ax)
    ax.set_title(get_title(c))

plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_all_confusion.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_all_confusion.svg')

# %% [markdown]
# Create metrics comparison

metric_comparison = pd.DataFrame(
    ([c.load_metrics() for c in keras_classifiers(CLASS, skip_model=True)]
    + [c.get_metrics(Xi, yi_expected) for c in random_classifiers(CLASS)]),
    index=pd.Index([get_title(c) for c in classifiers(skip_model=True)]))

metric_comparison.to_csv(f'results/{CLASS}_metric_comparison.csv')
metric_comparison
