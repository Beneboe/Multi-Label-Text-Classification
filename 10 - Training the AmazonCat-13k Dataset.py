# %% [markdown]
# # Training the AmazonCat-13k Dataset
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
BALANCED = False
DATASET_TYPE = 'tst'

# %% [markdown]
# Import the dataset

# %%
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

rng = np.random.default_rng()

df = pd.read_json(f'datasets/AmazonCat13K.{DATASET_TYPE}.json', lines=True)

# Make sequences same length
data = pad_sequences(df['text'], maxlen=INPUT_LENGTH)

def get_dataset(i):
    positive_samples = data[df['class'] == i + 1]
    negative_samples = data[df['class'] != i + 1]
    # Subsample negative indices
    if BALANCED:
        negative_samples = rng.choice(negative_samples, positive_samples.shape[0], replace=False)

    X = np.concatenate((positive_samples,negative_samples))

    # TODO: Update text
    y_positive = np.ones(positive_samples.shape[0], dtype='int32')
    y_negative = np.zeros(negative_samples.shape[0], dtype='int32')
    y = np.concatenate((y_positive,y_negative))

    return train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42)

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

def weights_file_name(i):
    return f'models/amazon/mlmc_classifier{i}{"balanced" if BALANCED else "unbalanced"}'

def train_classifier(i):
    # Create the classifier
    classifier = SimpleClassifier(embedding_layer)
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.summary()
    # Get the dataset
    X_train, X_test, y_train, y_test = get_dataset(i)
    # TODO: Save the dataset
    # Train the classifier
    history = classifier.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test), batch_size=10)
    # TODO: Store the history
    # Save the weights
    classifier.save_weights(weights_file_name(i))
    # TODO: Calculate the metrics

# %% [markdown]
# Actually train the classifiers.

# %%
for i in range(100):
    train_classifier(i)