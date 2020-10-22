# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset
from utils.plots import plot_confusion, plot_history
from utils.models import BaseBalancedClassifier, BaseUnbalancedClassifier, BalancedRandomClassifier, UnbalancedRandomClassifier
from utils.text_preprocessing import from_token_ids
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
from numpy.random import default_rng
import utils.metrics as mt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# %%
INPUT_LENGTH = 10
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
# CLASS_COUNT = 30
TRAIN_PATH = 'datasets/AmazonCat-13K/trn.processed.json'
TEST_PATH = 'datasets/AmazonCat-13K/tst.processed.json'
EPOCHS = 30
TRAINING_THRESHOLD = 2
CLASS = 8842

# %%
X_train, y_train = import_dataset(TRAIN_PATH, INPUT_LENGTH)
X_test, y_test = import_dataset(TEST_PATH, INPUT_LENGTH)
embedding_layer = import_embedding_layer()

# %%
# Model 1
inner_model = Sequential([
    LSTM(units=128),
    Dense(units=32),
    Dense(units=1, activation='sigmoid'),
])

# Model 2
# inner_model = Sequential([
#     LSTM(units=128, return_sequences=True),
#     Dropout(0.5),
#     LSTM(units=64),
#     Dropout(0.5),
#     Dense(units=1, activation='sigmoid'),
# ])

# Model 3
# inner_model = Sequential([
#     Dense(units=8),
#     Dropout(0.5),
#     Flatten(),
#     Dense(units=1, activation='sigmoid'),
# ])

model = Sequential([
    InputLayer(input_shape=(INPUT_LENGTH,)),
    embedding_layer,
    inner_model
])

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
    'accuracy',
    'Recall',
    'Precision',
])

# %% [markdown]
# Define the models

# %%
class BalancedClassifier(BaseBalancedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id)

class UnbalancedClassifier(BaseUnbalancedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id)

# %%
Xi, yi_expected = get_dataset(X_test, y_test, CLASS)

# %% [markdown]
# Plot the history diagrams

# %%
history1 = None
with open(f'results/history/{CLASS}_balanced.json', 'r') as fp:
    history1 = json.load(fp)
plot_history(history1)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_balanced_history.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_balanced_history.svg')
plt.show()

history2 = None
with open(f'results/history/{CLASS}_unbalanced.json', 'r') as fp:
    history2 = json.load(fp)
plot_history(history2)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_unbalanced_history.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_unbalanced_history.svg')
plt.show()

# %% [markdown]
# Save the false positives

# %%
balanced = BalancedClassifier(CLASS)
balanced.load_weights()
yi_predict = balanced.get_prediction(Xi)
fp_mask = np.logical_and(yi_predict == 1, yi_expected == 0)
Xi_fp = np.apply_along_axis(from_token_ids, 1, Xi[fp_mask])
np.savetxt(f'results/{CLASS}_balanced_false_positives.csv', Xi_fp, delimiter=',', fmt='%s')

# %% [markdown]
# Create the confusion matrix for each classifier

# %%
classifiers = [
    (BalancedClassifier, 'balanced'),
    (UnbalancedClassifier, 'unbalanced'),
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
fig, axs = plt.subplots(2, 2, figsize=(14,10))
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
        BalancedRandomClassifier(CLASS).get_metrics(Xi, yi_expected),
        UnbalancedRandomClassifier(CLASS).get_metrics(Xi, yi_expected),
    ],
    index=pd.Index(['Balanced', 'Unbalanced', 'BalancedRandom', 'UnbalancedRandom']))

metric_comparison.to_csv(f'results/{CLASS}_metric_comparison.csv')
metric_comparison
