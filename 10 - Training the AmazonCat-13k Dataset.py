# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset
from utils.models import BaseBalancedClassifier, BaseUnbalancedClassifier, Trainer
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives

# %% [markdown]
# # Training the AmazonCat-13k Dataset
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
# CLASS_COUNT = 30
TRAIN_PATH = 'datasets/AmazonCat-13K/trn.processed.json'
TEST_PATH = 'datasets/AmazonCat-13K/tst.processed.json'
EPOCHS = 10
TRAINING_THRESHOLD = 10

# %% [markdown]
# Import the dataset and the embedding layer

# %%
X_train, y_train = import_dataset(TRAIN_PATH, INPUT_LENGTH)
X_test, y_test = import_dataset(TEST_PATH, INPUT_LENGTH)
embedding_layer = import_embedding_layer()

# %% [markdown]
# Define the model

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
    Recall(),
    Precision(),
])

# %%
class BalancedClassifier(BaseBalancedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id)

class UnbalancedClassifier(BaseUnbalancedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id)

# %%
trainer_balanced = Trainer(
    BalancedClassifier,
    X_train, y_train,
    X_test, y_test,
    train_balance=True,
    threshold=TRAINING_THRESHOLD,
    epochs=EPOCHS)

trainer_unbalanced = Trainer(
    UnbalancedClassifier,
    X_train, y_train,
    X_test, y_test,
    train_balance=False,
    threshold=TRAINING_THRESHOLD,
    epochs=EPOCHS)

# %% [markdown]
# Actually train the classifiers.

# %%
# for i in range(CLASS_COUNT):
#     trainer_balanced.train(i)

# %%
CLASS = 8842

# %%
trainer_balanced.train(CLASS)

# %%
trainer_unbalanced.train(CLASS)