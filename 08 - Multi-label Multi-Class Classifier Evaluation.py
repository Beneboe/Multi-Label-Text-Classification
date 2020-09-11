# %% [markdown]
# # Multi-label Multi-Class Classifier Evaluation

# %% [markdown]
# ## Prepare the Data Set

# %%
# Set up hyper parameters
INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 4
BALANCED = False

# %%
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

rng = np.random.default_rng()

df = pd.read_csv("datasets/charcnn_keras_processed.csv",
    index_col=0,
    converters={"text": lambda x: x.strip("[]").replace("'","").split(", ")})

# Make sequences same length
data = pad_sequences(df['text'], maxlen=INPUT_LENGTH)

datasets = [None] * CLASS_COUNT
for i in range(CLASS_COUNT):
    positive_samples = data[df['class'] == i + 1]
    negative_samples = data[df['class'] != i + 1]
    # Subsample negative indices
    if BALANCED:
        negative_samples = rng.choice(negative_samples, positive_samples.shape[0], replace=False)

    X = np.concatenate((positive_samples,negative_samples))

    y_positive = np.ones(positive_samples.shape[0], dtype='int32')
    y_negative = np.zeros(negative_samples.shape[0], dtype='int32')
    y = np.concatenate((y_positive,y_negative))

    datasets[i] = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42)


# %% [markdown]
# ## Import the Embedding Layer

# %%
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/first-steps/GoogleNews-vectors-negative300.bin.gz", binary=True)

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
# ## Create the Classifier Models

# %%
embedding_layer = model.get_keras_embedding(train_embeddings=False)
classifiers = [None] * CLASS_COUNT
for i in range(CLASS_COUNT):
    classifiers[i] = SimpleClassifier(embedding_layer)
    classifiers[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifiers[i].load_weights(f'models/mlmc_classifier{i}{"balanced" if BALANCED else "unbalanced"}')
    classifiers[i].summary()

# %% [markdown]
# ## Calculate the metrics

# %%
from utils.evaluation import evaluate
from utils.evaluation import accuracy
from utils.evaluation import count
_, X_test, _, y_test = datasets[0]
evaluate(classifiers[0], X_test, y_test, accuracy)

# %%
loss, acc = classifiers[0].evaluate(X_test, y_test, verbose=0)
acc

# %%
