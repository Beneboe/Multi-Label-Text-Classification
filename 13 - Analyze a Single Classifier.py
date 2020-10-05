# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset
from utils.plots import plot_confusion, plot_history
from utils.models import BaseClassifier, Trainer
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
import matplotlib.pyplot as plt
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
# Train on balanced train set

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

trainer = Trainer(ClassifierBalanced, X_train, y_train, X_test, y_test)
trainer.train(CLASS, train_balance=True)


# %% [markdown]
# Train on unbalanced train set

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

trainer = Trainer(ClassifierUnbalanced, X_train, y_train, X_test, y_test)
trainer.train(CLASS, train_balance=False)

# %% [markdown]
# Create confusion plot

# %%
classifier1 = ClassifierBalanced(CLASS, True)
classifier1.load_weights()

# %% [markdown]
# Create the confusion plot

# %%
Xi, y_expected = get_dataset(X_test, y_test, CLASS, False)
(tp, fp, fn, tn) = classifier1.get_confusion(Xi, y_expected)
plot_confusion(tp, fp, fn, tn)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_balanced_confusion.png', dpi=163)

# %%
classifier2 = ClassifierUnbalanced(CLASS)
classifier2.load_weights()

# %%
Xi, y_expected = get_dataset(X_test, y_test, CLASS, False)
(tp, fp, fn, tn) = classifier2.get_confusion(Xi, y_expected)
plot_confusion(tp, fp, fn, tn)
plt.savefig(f'datasets/imgs/classifier_{CLASS}_unbalanced_confusion.png', dpi=163)

# %% [markdown]
# Create the history plot

# %%
history1 = None
with open(f'results/history/{CLASS}_balanced.json', 'r') as fp:
    history1 = json.load(fp)
plot_history(history1)
plt.savefig('datasets/imgs/classifier_{CLASS}_balanced_history.png', dpi=163)

# %%
history2 = None
with open(f'results/history/{CLASS}_unbalanced.json', 'r') as fp:
    history2 = json.load(fp)
plot_history(history2)
plt.savefig('datasets/imgs/classifier_{CLASS}_unbalanced_history.png', dpi=163)
