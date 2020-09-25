# %% [markdown]
# # Analyze Different Label Frequency Thresholds
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 10
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
# CLASS_COUNT = 30
WEIGHTS_FILE_TEMPLATE = 'results/weights/cl_class={0}'
HISTORY_FILE_TEMPLATE = 'results/history/cl_class={0}.json'
METRICS_FILE_TEMPLATE = 'results/metrics/cl_class={0}.json'
TRAIN_PATH = 'datasets/AmazonCat-13K/trn.processed.json'
TEST_PATH = 'datasets/AmazonCat-13K/tst.processed.json'
EPOCHS = 30
TRAINING_THRESHOLD = 2

# %% [markdown]
# Import the dataset and the embedding layer

# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset

X_train, y_train = import_dataset(TRAIN_PATH, INPUT_LENGTH)
X_test, y_test = import_dataset(TEST_PATH, INPUT_LENGTH)
embedding_layer = import_embedding_layer()

# %% [markdown]
# Create, fit, and save the classifier Models

# %%
from utils.models import LSTMModel, get_embedding_layer, get_metrics
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
import numpy as np
import json

def process_classifier(i):
    print(f'Processing classifier {i}...')
    # Create the classifier
    classifier = LSTMModel(embedding_layer, INPUT_LENGTH)
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        Recall(),
        Precision(),
    ])
    classifier.summary()

    # Get the dataset
    Xi, yi = ds.get_dataset(X_train, y_train, i)

    # Train the classifier
    history = classifier.fit(Xi, yi, epochs=EPOCHS, verbose=1, batch_size=32)

    # Store the history
    with open(HISTORY_FILE_TEMPLATE.format(i), 'w') as fp:
        json.dump(history.history, fp)

    # Save the weights
    classifier.save_weights(WEIGHTS_FILE_TEMPLATE.format(i))

    # Calculate the metrics
    Xi_test, yi_test = get_dataset(X_test, y_test, i, balanced=False)
    metrics = get_metrics(classifier, Xi_test, yi_test)

    # Store the metrics
    with open(METRICS_FILE_TEMPLATE.format(i), 'w') as fp:
        json.dump(metrics, fp)

# %%
from utils.dataset import class_frequencies

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

# %%
# Train the classifiers
for i in range(len(thresholds)):
    label = labels[i]
    print('The label {0} occurs {1} times'.format(label, freqs[label]))
    process_classifier(label)

# %%
def load_metrics(i):
    with open(METRICS_FILE_TEMPLATE.format(i), 'r') as fp:
        return json.load(fp)

# %%
# Load the metrics
precisions = [0.0] * len(thresholds)
recalls = [0.0] * len(thresholds)
for i in range(len(thresholds)):
    dict = load_metrics(labels[i])
    precisions[i] = dict['precision']
    recalls[i] = dict['recall']

# %%
# Create the graph for each of the thresholds
import matplotlib.pyplot as plt

plot_labels = ['{:,d}'.format(threshold) for threshold in thresholds]

plt.plot(plot_labels, precisions, 'bs')
plt.plot(plot_labels, recalls, 'y^')
plt.title('Label Frequencies Metrics')
plt.xlabel('Label Frequency Levels')
plt.legend(['Precision', 'Recall'])
plt.grid()
plt.show()

# %%
# Create the table for each of the thresholds
import pandas as pd

pd.DataFrame({
    'precision': precisions,
    'recall': recalls,
}, index=plot_labels)
