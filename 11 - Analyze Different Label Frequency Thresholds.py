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
# Import the dataset

# %%
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

rng = np.random.default_rng(42)

def import_dataset(path):
    ds_frame = pd.read_json(path, lines=True)
    # Make sequences same length
    X = pad_sequences(ds_frame['X'], maxlen=INPUT_LENGTH)
    y = ds_frame['y']
    return X, y

X_train, y_train = import_dataset(TRAIN_PATH)
X_test, y_test = import_dataset(TEST_PATH)

# %%
def is_positive(i):
    return lambda y: i in y

def is_negative(i):
    return lambda y: i not in y

def get_dataset(X, y, i, balanced=True):
    X_positive = X[y.map(is_positive(i))]
    X_negative = X[y.map(is_negative(i))]

    # Subsample negative indices
    if balanced:
        X_negative = rng.choice(X_negative, X_positive.shape[0], replace=False)

    y_positive = np.ones(X_positive.shape[0], dtype='int8')
    y_negative = np.zeros(X_negative.shape[0], dtype='int8')

    X = np.concatenate((X_positive,X_negative))
    y = np.concatenate((y_positive,y_negative))

    # Shuffle the data
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y

# %% [markdown]
# ## Define the Classifier Model

# %%
import keras
from keras.layers import LSTM, Dense, InputLayer

class SimpleClassifier (keras.Sequential):
    def __init__(self, embedding_layer, d_units=8):
        super(SimpleClassifier, self).__init__()

        self.inner = keras.Sequential([
            LSTM(units=256),
            Dense(units=64),
            Dense(units=1, activation='sigmoid'),
        ])

        self.add(InputLayer(input_shape=(INPUT_LENGTH,)))
        self.add(embedding_layer)
        self.add(self.inner)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Do not save the weights of the embedding layer
        return self.inner.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        return self.inner.load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

# %% [markdown]
# ## Import the Embedding Layer

# %%
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)
embedding_layer = model.get_keras_embedding(train_embeddings=False)

# %% [markdown]
# ## Create, fit, and save the classifier Models
# Define the steps.

# %%
import utils.metrics as mt
import keras.metrics as kmt
import json

def get_metrics(classifier, X, y_expected):
    y_predict = mt.get_prediction(classifier, X)

    return {
        'count': mt.count(y_predict, y_expected),
        'accuracy': mt.accuracy(y_predict, y_expected),
        'recall': mt.recall(y_predict, y_expected),
        'precision': mt.precision(y_predict, y_expected),
        'f1 measure': mt.f1measure(y_predict, y_expected),
    }

def process_classifier(i):
    print(f'Processing classifier {i}...')
    # Create the classifier
    classifier = SimpleClassifier(embedding_layer)
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        kmt.Recall(),
        kmt.Precision(),
    ])
    classifier.summary()

    # Get the dataset
    Xi, yi = get_dataset(X_train, y_train, i)

    # Train the classifier
    history = classifier.fit(Xi, yi, epochs=EPOCHS, verbose=1, batch_size=20)

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
import utils.dataset as ds

freqs = ds.class_frequencies(CLASS_COUNT, y_train)
freqs_args = np.argsort(freqs)

def freqs_args_below(threshold):
    # Index before which all indexes point to frequences below the threshold
    i = np.searchsorted(freqs, threshold, side='right', sorter=freqs_args)
    return freqs_args[i-1:0:-1]

# %%
# thresholds = [50, 100, 1_000, 10_000, 50_000, 100_000]
thresholds = [50, 100, 500, 1_000, 5_000, 10_000, 50_000]
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
import matplotlib.pyplot as plt

plot_labels = ['{:,d}'.format(threshold) for threshold in thresholds]

plt.plot(plot_labels, precisions, 'bs')
plt.plot(plot_labels, recalls, 'y^')
plt.title('Label Frequencies Metrics')
plt.xlabel('Label Frequency Levels')
plt.legend(['Precision', 'Recall'])
plt.show()

# %%
pd.DataFrame({
    'precision': precisions,
    'recall': recalls,
}, index=plot_labels)

# %%