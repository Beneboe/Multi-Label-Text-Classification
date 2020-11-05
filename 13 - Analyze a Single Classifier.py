# %%
from utils.dataset import import_dataset, get_dataset
from utils.plots import plot_confusion, plot_history
from utils.models import BalancedClassifier, UnbalancedClassifier, Weighted10Classifier, Weighted20Classifier, BalancedRandomClassifier, UnbalancedRandomClassifier
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

# %% [markdown]
# Plot the history diagrams

# %%
history_names = ['balanced', 'unbalanced', 'p20', 'p10']
for name in history_names:
    history = None
    with open(f'results/history/{CLASS}_{name}.json', 'r') as fp:
        history = json.load(fp)
    plot_history(history, CLASS)
    plt.tight_layout()
    plt.savefig(f'datasets/imgs/classifier_{CLASS}_{name}_history.png', dpi=163)
    plt.savefig(f'datasets/imgs/classifier_{CLASS}_{name}_history.svg')
    plt.show()

# %% [markdown]
# Retrieve and save the false positives

# %%
balanced = BalancedClassifier(CLASS)
balanced.load_weights()
yi_predict = balanced.get_prediction(Xi)
fp_mask = np.logical_and(yi_predict == 1, yi_expected == 0)
Xi_fp = np.apply_along_axis(from_token_ids, 1, Xi[fp_mask])
np.savetxt(f'results/{CLASS}_balanced_false_positives.csv', Xi_fp, delimiter=',', fmt='%s')

# %% [markdown]
# Retrieve and save the true positives

# %%
balanced = BalancedClassifier(CLASS)
balanced.load_weights()
yi_predict = balanced.get_prediction(Xi)
fp_mask = np.logical_and(yi_predict == 1, yi_expected == 1)
Xi_fp = np.apply_along_axis(from_token_ids, 1, Xi[fp_mask])
np.savetxt(f'results/{CLASS}_balanced_true_positives.csv', Xi_fp, delimiter=',', fmt='%s')

# %% [markdown]
# Create the confusion matrix for each classifier

# %%
classifiers = [
    (BalancedClassifier, 'balanced'),
    (UnbalancedClassifier, 'unbalanced'),
    (Weighted10Classifier, 'p10'),
    (Weighted20Classifier, 'p20'),
    (BalancedRandomClassifier, 'balanced_random'),
    (UnbalancedRandomClassifier, 'unbalanced_random'),
]

# %%
for (classifier_type, name) in classifiers:
    classifier = classifier_type(CLASS)
    classifier.load_weights()
    cm = classifier.get_confusion(Xi, yi_expected)
    plot_confusion(cm)
    plt.tight_layout()
    plt.savefig(f'datasets/imgs/classifier_{CLASS}_{name}_confusion.png', dpi=163)
    plt.savefig(f'datasets/imgs/classifier_{CLASS}_{name}_confusion.svg')
    plt.show()

# %%
fig, axs = plt.subplots(3, 2, figsize=(14,10))
for i, (classifier_type, name) in enumerate(classifiers):
    classifier = classifier_type(CLASS)
    classifier.load_weights()
    cm = classifier.get_confusion(Xi, yi_expected)
    ax = axs[i // 2, i % 2]
    plot_confusion(cm, ax)
    title = ' '.join(word.capitalize() for word in name.split('_'))
    ax.set_title(title)


plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_all_confusion.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_all_confusion.svg')

# %% [markdown]
# Create metrics comparison

metric_comparison = pd.DataFrame(
    [
        BalancedClassifier(CLASS).load_metrics(),
        UnbalancedClassifier(CLASS).load_metrics(),
        Weighted10Classifier(CLASS).load_metrics(),
        Weighted20Classifier(CLASS).load_metrics(),
        BalancedRandomClassifier(CLASS).get_metrics(Xi, yi_expected),
        UnbalancedRandomClassifier(CLASS).get_metrics(Xi, yi_expected),
    ],
    index=pd.Index(['Balanced', 'Unbalanced', 'Positive 10', 'Positive 20', 'BalancedRandom', 'UnbalancedRandom']))

metric_comparison.to_csv(f'results/{CLASS}_metric_comparison.csv')
metric_comparison
