# %%
from utils.dataset import import_amazoncat13k, get_dataset
from utils.plots import plot_confusion
from utils.models import random_prediction, prediction_threshold
import utils.metrics as mt
import matplotlib.pyplot as plt
import utils.storage as st
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib.colors import LogNorm
from itertools import product

# %%
INPUT_LENGTH = 10
CLASS = 8842
CLASS_NAME = 'personal care'

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
keras_classes = ['50%positive', '20%positive', '10%positive', 'unbalanced']
random_classes = ['50%positive_random', 'unbalanced_random']
all_classes = keras_classes + random_classes
all_class_names = ['1:1', '10:2', '10:1', 'unbalanced', 'random 1:1', 'random 222:1']

def predictions(classifiers):
    for name in classifiers:
        yield st.load_prediction(CLASS, name)

def confusions(classifiers, normalize=False):
    for yi_predict in predictions(classifiers):
        cm = mt.get_confusion(yi_predict, yi_expected)
        if normalize:
            length = yi_predict.shape[0]
            tp, fp, fn, tn = cm
            cm = (tp / length, fp / length, fn / length, tn / length)
        yield cm

def metrics(classifiers):
    for yi_predict in predictions(classifiers):
        yield mt.all_metrics(yi_predict, yi_expected)

# %% [markdown]
# Create the confusion matrix for each classifier

# %%
for name, cm in zip(all_class_names, confusions(all_classes)):
    plot_confusion(cm, name)
    plt.tight_layout()
    plt.savefig(f'results/imgs/classifier_{name}_confusion.png', dpi=163)
    plt.savefig(f'results/imgs/classifier_{name}_confusion.pdf')
    plt.show()

# %%
nrows, ncols = 2, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
for i, (name, cm) in enumerate(zip(all_class_names, confusions(all_classes))):
    ax = axs[i // ncols, i % ncols]
    plot_confusion(cm, name, ax)

plt.tight_layout()
plt.savefig(f'results/imgs/classifier_{CLASS}_all_confusion.png', dpi=163)
plt.savefig(f'results/imgs/classifier_{CLASS}_all_confusion.pdf')

# %%
pd.DataFrame([cm[0]/(cm[0]+cm[1]) for cm in confusions(keras_classes, True)])

# %%
for name, cm in zip(all_class_names[0:4], confusions(keras_classes, True)):
    plot_confusion(cm, name)

# %%
def combine_confusions(cms):
    cmsl = list(cms)
    ccm = np.zeros((2 * len(cmsl), 2))
    for i in range(len(cmsl)):
        (tp, fp, fn, tn) = cmsl[i]
        ccm[2 * i + 0, 0] = tp
        ccm[2 * i + 0, 1] = fp
        ccm[2 * i + 1, 0] = fn
        ccm[2 * i + 1, 1] = tn
    return ccm

cm = combine_confusions(confusions(keras_classes, True))

df = pd.DataFrame(
    combine_confusions(confusions(keras_classes, True)),
    index=pd.Index(list(map(lambda x: ' '.join(x), product(all_class_names[0:4], ['True', 'False']))), name='Predicted'),
    columns=pd.Index(['True', 'False'], name='Actual'))

ax = sb.heatmap(df,
    fmt='.2%',
    annot=True,
    norm=LogNorm(vmin=cm.min().min(), vmax=cm.max().max()))

ax.set_title(f"Confusion Matrices for '{CLASS_NAME}'")

plt.tight_layout()
plt.savefig(f'results/imgs/{CLASS}_subsamp_confusion.png', dpi=163)
plt.savefig(f'results/imgs/{CLASS}_subsamp_confusion.pdf')
plt.show()

# %%
x_ticks = ['1:1 balanced', '10:2 balanced', '10:1 balanced', '(222:1) unbalanced']
# x_ticks = range(4)

recalls = []
precisions = []
for m in metrics(keras_classes):
    recalls.append(m['recall'])
    precisions.append(m['precision'])

line2, = plt.plot(x_ticks, recalls, 'b^', label = 'Recall')
line1, = plt.plot(x_ticks, precisions, 'bs', label = 'Precision')

plt.title('Sub-sampling strategy comparison')
plt.xlabel('Sub-sampling strategy')
plt.ylabel('Metric Performance')

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(f'results/imgs/{CLASS}_subsamp_plot.png', dpi=163)
plt.savefig(f'results/imgs/{CLASS}_subsamp_plot.pdf')
plt.show()

# %%
pd.DataFrame(metrics(keras_classes))

# %% [markdown]
# Create metrics comparison
metric_comparison = pd.DataFrame(
    list(metrics(all_classes)),
    index=pd.Index([name for name in all_class_names]))

metric_comparison.to_csv(f'results/imgs_data/classifier_{CLASS}_all_metrics.csv')
metric_comparison

# %%
