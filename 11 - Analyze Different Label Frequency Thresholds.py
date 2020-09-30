# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset
from utils.models import BaseClassifier, Trainer
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
from utils.dataset import class_frequencies
from utils.plots import confusion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # Analyze Different Label Frequency Thresholds
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 10
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
# CLASS_COUNT = 30
TRAIN_PATH = 'datasets/AmazonCat-13K/trn.processed.json'
TEST_PATH = 'datasets/AmazonCat-13K/tst.processed.json'
EPOCHS = 30
TRAINING_THRESHOLD = 2

# %% [markdown]
# Import the dataset and the embedding layer

# %%

X_train, y_train = import_dataset(TRAIN_PATH, INPUT_LENGTH)
X_test, y_test = import_dataset(TEST_PATH, INPUT_LENGTH)
embedding_layer = import_embedding_layer()

# %% [markdown]
# Define the model

# %%

# Alternative 1
inner_model = Sequential([
    LSTM(units=256),
    Dense(units=64),
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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
    'accuracy',
    Recall(),
    Precision(),
])

class Classifier(BaseClassifier):
    def __init__(self, id):
        super(Classifier, self).__init__(model, inner_model, id)

trainer = Trainer(Classifier, X_train, y_train, X_test, y_test, threshold=2, epochs=EPOCHS)

# %% [markdown]
# Calculate the frequencies

# %%
freqs = class_frequencies(CLASS_COUNT, y_train)
freqs_args = np.argsort(freqs)

def freqs_args_below(threshold):
    # Index before which all indexes point to frequences below the threshold
    i = np.searchsorted(freqs, threshold, side='right', sorter=freqs_args)
    return freqs_args[i-1:0:-1]

# %%
# thresholds = [50, 100, 1_000, 10_000, 50_000, 100_000]
thresholds = [50, 100, 500, 1_000, 5_000, 10_000]
labels = [freqs_args_below(threshold)[0] for threshold in thresholds]

# %% [markdown]
# Train the classifiers

# %%
for i in range(len(thresholds)):
    label = labels[i]
    print('The label {0} occurs {1} times'.format(label, freqs[label]))
    trainer.train(label)

# %% [markdown]
# Gather the metrics

# %%
precisions = [0.0] * len(thresholds)
recalls = [0.0] * len(thresholds)
for i in range(len(thresholds)):
    classifier = Classifier(labels[i])
    dict = classifier.get_metrics()
    precisions[i] = dict['precision']
    recalls[i] = dict['recall']

# %% [markdown]
# Graph the performance of each thresholds

# %%
plot_labels = ['{:,d}'.format(threshold) for threshold in thresholds]

plt.plot(plot_labels, precisions, 'bs')
plt.plot(plot_labels, recalls, 'y^')
plt.title('Classifier Frequency Performance')
plt.xlabel('Label Frequency Levels')
plt.legend(['Precision', 'Recall'])
plt.grid()
plt.show()
plt.savefig(f'datasets/imgs/AmazonCat-13K_classifier_frequency_performance.png', dpi=163)

# %%
# Create the table for each of the thresholds

pd.DataFrame({
    'precision': precisions,
    'recall': recalls,
}, index=plot_labels)
