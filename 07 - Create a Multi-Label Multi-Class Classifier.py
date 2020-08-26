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
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 4

df = pd.read_csv("datasets/charcnn_keras_processed.csv",
    index_col=0,
    converters={"text": lambda x: x.strip("[]").replace("'","").split(", ")})

# Make sequences same length
data = pad_sequences(df['text'], maxlen=INPUT_LENGTH)

datasets = [None] * CLASS_COUNT
for i in range(CLASS_COUNT):
    indices = np.arange(df.shape[0])
    positive_indices = indices[df['class'] == i + 1]
    # negative_indices = indices[df['class'] != i + 1]

    X = data
    y = np.zeros(data.shape[0])
    y[positive_indices] = 1

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

for i in range(CLASS_COUNT):
    plt.plot(histories[i].history['accuracy'])
    plt.plot(histories[i].history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# %% [markdown]
# ## Save the Classifier Models

# %%
for i in range(CLASS_COUNT):
    classifiers[i].save(f'models/mlmc_classifier{i}')