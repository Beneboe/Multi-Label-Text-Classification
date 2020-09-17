# %% [markdown]
# # Training the AmazonCat-13k Dataset
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
BALANCED = True
WEIGHTS_FILE_TEMPLATE = 'results/weights/cl_bal={0}_class={{0}}'.format('1' if BALANCED else '0')
HISTORY_FILE_TEMPLATE = 'results/history/cl_bal={0}_class={{0}}'.format('1' if BALANCED else '0')
METRICS_FILE_TEMPLATE = 'results/metrics/cl_bal={0}_class={{0}}'.format('1' if BALANCED else '0')

# %% [markdown]
# Import the dataset

# %%
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

rng = np.random.default_rng()

def import_dataset(path):
    ds_frame = pd.read_json(path, lines=True)
    # Make sequences same length
    X = pad_sequences(ds_frame['X'], maxlen=INPUT_LENGTH)
    y = ds_frame['y']
    return X, y

X_train, y_train = import_dataset(f'datasets/AmazonCat13K.trn.json')
X_test, y_test = import_dataset(f'datasets/AmazonCat13K.tst.json')

# %%
def is_positive(i):
    return lambda y: i in y

def is_negative(i):
    return lambda y: i not in y

def get_dataset(X, y, i):
    X_positive = X[y.map(is_positive(i))]
    X_negative = X[y.map(is_negative(i))]
    # Subsample negative indices
    if BALANCED:
        X_negative = rng.choice(X_negative, X_positive.shape[0], replace=False)

    y_positive = np.ones(X_positive.shape[0], dtype='int8')
    y_negative = np.zeros(X_negative.shape[0], dtype='int8')

    X = np.concatenate((X_positive,X_negative))
    y = np.concatenate((y_positive,y_negative))

    return X, y

# %% [markdown]
# ## Define the Classifier Model

# %%
import keras
class SimpleClassifier (keras.Sequential):
    def __init__(self, embedding_layer, d_units=8):
        super(SimpleClassifier, self).__init__()

        self.inner = keras.Sequential([
            keras.layers.Dense(units=d_units, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=1, activation='sigmoid'),
        ])

        self.add(keras.layers.InputLayer(input_shape=(INPUT_LENGTH,)))
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

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/first-steps/GoogleNews-vectors-negative300.bin.gz", binary=True)
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
        kmt.Accuracy(),
        kmt.Recall(),
        kmt.Precision(),
    ])
    classifier.summary()
    # Get the dataset
    Xi, yi = get_dataset(X_train, y_train, i)
    Xi_train, Xi_train_test, yi_train, yi_train_test = train_test_split(Xi, yi, test_size=VALIDATION_SPLIT, random_state=42)
    # TODO: Save the dataset
    # Train the classifier
    history = classifier.fit(Xi_train, yi_train, epochs=10, verbose=1, validation_data=(Xi_train_test, yi_train_test), batch_size=10)
    # Save the weights
    classifier.save_weights(WEIGHTS_FILE_TEMPLATE.format(i))
    # Calculate the metrics
    Xi_test, yi_test = get_dataset(X_test, y_test, i)
    # metrics = get_metrics(classifier, ???, ???)
    metrics = classifier.evaluate(Xi_test, yi_test, return_dict=True)
    # Store the history
    with open(HISTORY_FILE_TEMPLATE, 'w') as fp:
        json.dump(history.history, fp)
    # Store the metrics
    with open(METRICS_FILE_TEMPLATE, 'w') as fp:
        json.dump(metrics, fp)

# %% [markdown]
# Actually train the classifiers.

# %%
for i in range(CLASS_COUNT):
    process_classifier(i)