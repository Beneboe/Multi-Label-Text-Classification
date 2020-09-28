# %% [markdown]
# # Multi-label Multi-Class Classifier Evaluation
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 4
BALANCED = False


# %% [markdown]
# ## Prepare the Data Set

# %%
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

rng = np.random.default_rng()

df = pd.read_csv("datasets/ag_news_csv/train.processed.csv",
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

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)

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
# Keras's Metrics:

# %%

def calc_accuracy(i):
    _, X_test, _, y_test = datasets[i]
    loss, acc = classifiers[i].evaluate(X_test, y_test, verbose=0)
    return acc

report = pd.DataFrame({
    "accuracy": [calc_accuracy(i) for i in range(CLASS_COUNT)],
})
report

# %% [markdown]
# Library Metrics (for each classifier):

# %%
import utils.metrics as mt

report_data = np.zeros((CLASS_COUNT,), dtype=[
    ("count", "i4"),
    ("accuracy", "f4"),
    ("recall", "f4"),
    ("precision", "f4"),
    ("f1 measure", "f4"),
])

# Initialize report data
for i in range(CLASS_COUNT):
    _, X, _, y_expected = datasets[i]
    y_predict = mt.get_prediction(classifiers[i], X)

    report_data[i] = (
        mt.count(y_predict, y_expected),
        mt.accuracy(y_predict, y_expected),
        mt.recall(y_predict, y_expected),
        mt.precision(y_predict, y_expected),
        mt.f1measure(y_predict, y_expected),
    )

report = pd.DataFrame(report_data)
report

# %% [markdown]
# Next, compute a new test set for the metrics that measure the classification as a whole.

# %%

# Shape of dataset (unbalanced)
# all   120 000
# train  96 000
# test   24 000

# X_test of dataset[0] can overlap with X_test of dataset[1]
# Get new test set instead

indices = np.arange(data.shape[0])
indices = rng.choice(indices, int(indices.shape[0] * VALIDATION_SPLIT), replace=False)

X = data[indices]

y_data = df['class'].to_numpy()[indices]
y = np.zeros((X.shape[0], CLASS_COUNT), dtype='int32')
for i in range(CLASS_COUNT):
    ones_of_class = y_data == i + 1
    y[:,i][ones_of_class] = 1

# %% [markdown]
# Library Metrics:

# %%

y_predict = mt.get_all_predictions(classifiers, X)
y_expected = y

print("macro f1 measure:", mt.macro_f1measure(y_predict, y_expected))
print("micro f1 measure:", mt.micro_f1measure(y_predict, y_expected))
