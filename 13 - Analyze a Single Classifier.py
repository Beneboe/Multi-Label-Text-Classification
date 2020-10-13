# %%
from numpy.core.fromnumeric import shape
from utils.dataset import import_dataset, import_embedding_layer, get_dataset
from utils.plots import plot_confusion, plot_history
from utils.models import BaseClassifier, Trainer
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
# Alternative 1
inner_model = Sequential([
    LSTM(units=128),
    Dense(units=32),
    Dense(units=1, activation='sigmoid'),
])

# Alternative 2
# inner_model = Sequential([
#     LSTM(units=128, return_sequences=True),
#     Dropout(0.5),
#     LSTM(units=64),
#     Dropout(0.5),
#     Dense(units=1, activation='sigmoid'),
# ])

# Alternative 3
# inner_model = Sequential([
#     Dense(units=4),
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
class ClassifierBalanced(BaseClassifier):
    def __init__(self, id):
        super(ClassifierBalanced, self).__init__(model, inner_model, id)

    def get_weights_path(self):
        return f'results/weights/{self.id}_balanced'

    def get_history_path(self):
        return f'results/history/{self.id}_balanced.json'

    def get_metrics_path(self):
        return f'results/metrics/{self.id}_balanced.json'

# %%
class ClassifierUnbalanced(BaseClassifier):
    def __init__(self, id):
        super(ClassifierUnbalanced, self).__init__(model, inner_model, id)

    def get_weights_path(self):
        return f'results/weights/{self.id}_unbalanced'

    def get_history_path(self):
        return f'results/history/{self.id}_unbalanced.json'

    def get_metrics_path(self):
        return f'results/metrics/{self.id}_unbalanced.json'

# %% [markdown]
# Train on balanced train set

# %%
trainer = Trainer(ClassifierBalanced, X_train, y_train, X_test, y_test)
trainer.train(CLASS, train_balance=True)

# %% [markdown]
# Train on unbalanced train set

# %%
trainer = Trainer(ClassifierUnbalanced, X_train, y_train, X_test, y_test)
trainer.train(CLASS, train_balance=False)

# %%
Xi, yi_expected = get_dataset(X_test, y_test, CLASS, False)

# %% [markdown]
# Create diagrams for the balanced classifier

# %%
balanced = ClassifierBalanced(CLASS)
balanced.load_weights()

# %%
cm1 = balanced.get_confusion(Xi, yi_expected)
plot_confusion(cm1)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_balanced_confusion.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_balanced_confusion.svg')
plt.show()

# %%
history1 = None
with open(f'results/history/{CLASS}_balanced.json', 'r') as fp:
    history1 = json.load(fp)
plot_history(history1)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_balanced_history.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_balanced_history.svg')
plt.show()

# %% [markdown]
# Save the false positives

# %%
yi_predict = balanced.get_prediction(Xi)
fp_mask = np.logical_and(yi_predict == 1, yi_expected == 0)
Xi_fp = np.apply_along_axis(from_token_ids, 1, Xi[fp_mask])
np.savetxt(f'results/{CLASS}_balanced_false_positives.csv', Xi_fp, delimiter=',', fmt='%s')

# %% [markdown]
# Create diagrams for the unbalanced classifier

# %%
unbalanced = ClassifierUnbalanced(CLASS)
unbalanced.load_weights()

# %%
cm2 = unbalanced.get_confusion(Xi, yi_expected)
plot_confusion(cm2)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_unbalanced_confusion.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_unbalanced_confusion.svg')
plt.show()

# %%
history2 = None
with open(f'results/history/{CLASS}_unbalanced.json', 'r') as fp:
    history2 = json.load(fp)
plot_history(history2)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_unbalanced_history.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_unbalanced_history.svg')
plt.show()

# %% [markdown]
# Create diagrams for the random classifier (0.5)

# %%
yi_predict_rand1 = default_rng(42).uniform(size=yi_expected.shape[0])
yi_predict_rand1 = (yi_predict_rand1 >= 0.5).astype('int32')
metrics3 = mt.all_metrics(yi_predict_rand1, yi_expected)

# %%
cm3 = mt.get_confusion(yi_predict_rand1, yi_expected)
plot_confusion(cm3)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_random(0.5)_confusion.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_random(0.5)_confusion.svg')
plt.show()

# %% [markdown]
# Create diagrams for the random classifier (frequency adjusted)

# %%
freq_threshold = np.count_nonzero(yi_expected < 0.5) / yi_expected.shape[0]

yi_predict_rand2 = default_rng(42).uniform(size=yi_expected.shape[0])
yi_predict_rand2 = (yi_predict_rand2 >= freq_threshold).astype('int32')

metrics4 = mt.all_metrics(yi_predict_rand2, yi_expected)

# %%
cm4 = mt.get_confusion(yi_predict_rand2, yi_expected)
plot_confusion(cm4)
plt.tight_layout()
plt.savefig(f'datasets/imgs/classifier_{CLASS}_random(adjusted)_confusion.png', dpi=163)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_random(adjusted)_confusion.svg')
plt.show()

# %% [markdown]
# Create metrics comparison

metric_comparison = pd.DataFrame([
    ClassifierBalanced(CLASS).load_metrics(),
    ClassifierUnbalanced(CLASS).load_metrics(),
    metrics3,
    metrics4,
], index=pd.Index(['Balanced', 'Unbalanced', 'Random(0.5)', 'Random(Adjusted)']))

metric_comparison.to_csv(f'results/{CLASS}_metric_comparison.csv')
metric_comparison

# %%
