# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset
from utils.models import BaseBalancedClassifier, BaseUnbalancedClassifier, Trainer
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
import pandas as pd

# %% [markdown]
# # Training the AmazonCat-13k Dataset
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 100
CLASS_COUNT = 13330

# %% [markdown]
# Import the dataset and the embedding layer

# %%
X_train, y_train = import_dataset('datasets/AmazonCat-13K/trn.processed.json', INPUT_LENGTH)
X_test, y_test = import_dataset('datasets/AmazonCat-13K/tst.processed.json', INPUT_LENGTH)
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
    'Recall',
    'Precision',
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
    train_balance=True)

trainer_unbalanced = Trainer(
    UnbalancedClassifier,
    X_train, y_train,
    X_test, y_test,
    train_balance=False)

# %% [markdown]
# Actually train the classifiers.

# %%
# for i in range(CLASS_COUNT):
#     trainer_balanced.train(i)

# %%
trainer_balanced.train(8842)

# %%
trainer_unbalanced.train(8842)

# %%
# Labels just below certain thresholds
# Order: threshold, label, frequency
threshold_data = [
    (50,6554,50),
    (100,4949,100),
    (1000,7393,996),
    (10000,84,9976),
    (50000,9202,48521),
    (100000,7083,96012),
]

# %%
# Top 10 most frequent labels ordered from most to least frequent
# Order: label, frequency
top10_label_data = [
    (1471,355211)
    (7961,194561)
    (7892,128026)
    (9237,120090)
    # (7083,97803) # duplicate
    (7891,88967)
    (4038,76277)
    (10063,75035)
    (12630,71667)
]

# %%
# Train threshold labels
for _,label,_ in threshold_data:
    trainer_balanced.train(label)
    trainer_unbalanced.train(label)

# %%
# Train most frequent labels
for label,_ in top10_label_data:
    trainer_balanced.train(label)
    trainer_unbalanced.train(label)
