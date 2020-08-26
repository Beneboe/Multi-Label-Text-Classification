# %% [markdown]
# # Create a Multi-Label Multi-Class Classifier

# %% [markdown]
# ## Import the Embedding Layer

# %%
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/first-steps/GoogleNews-vectors-negative300.bin.gz", binary=True)

# %% [markdown]
# ## Prepare the Data Set

# %%
# Set up hyper parameters
INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 4
BALANCED = True

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
    indices = np.arange(df.shape[0])
    positive_indices = indices[df['class'] == i + 1]
    negative_indices = indices[df['class'] != i + 1]
    # Subsample negative indices
    if BALANCED:
        negative_indices = rng.choice(negative_indices, positive_indices.shape[0], replace=False)

    X_positive = data[positive_indices]
    X_negative = data[negative_indices]
    X = np.concatenate((X_positive,X_negative))

    y_positive = np.ones(positive_indices.shape[0], dtype='int32')
    y_negative = np.zeros(negative_indices.shape[0], dtype='int32')
    y = np.concatenate((y_positive,y_negative))

    datasets[i] = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42)

# %% [markdown]
# ## Define the Classifier Model

# %%
import keras
class SimpleClassifier (keras.Sequential):
    def __init__(self, embedding_layer, d_units=8):
        super().__init__([
            keras.layers.InputLayer(input_shape=(INPUT_LENGTH,)),
            embedding_layer,
            keras.layers.Dense(units=d_units, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=1, activation='sigmoid'),
        ])

# %% [markdown]
# ## Create the Classifier Models

# %%
embedding_layer = model.get_keras_embedding(train_embeddings=False)
classifiers = [None] * CLASS_COUNT
for i in range(CLASS_COUNT):
    classifiers[i] = SimpleClassifier(embedding_layer)
    classifiers[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifiers[i].summary()

# %% [markdown]
# ## Fit the Classifier Models

# %%
histories = [None] * CLASS_COUNT
for i in range(CLASS_COUNT):
    X_train, X_test, y_train, y_test = datasets[i]
    histories[i] = classifiers[i].fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test), batch_size=10)

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)

for i in range(CLASS_COUNT):
    ax = axs[i // 2][i % 2]
    ax.plot(histories[i].history['accuracy'])
    ax.plot(histories[i].history['val_accuracy'])
    ax.set_title(f'Classifier {i} Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='lower right')
    ax.grid()

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Save the Classifier Models

# %%
for i in range(CLASS_COUNT):
    classifiers[i].save(f'models/mlmc_classifier{i}')