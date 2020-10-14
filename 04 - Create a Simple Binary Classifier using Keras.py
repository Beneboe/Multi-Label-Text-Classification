# %% [markdown]
# # Create a Simple Binary Classifier using Keras

# %% [markdown]
# ## Import the Embedding Layer

# %%
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)

# %% [markdown]
# ## Prepare the Data Set
# Only use instances where the class is either 3 or 4.

# %%
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2

df = pd.read_csv('datasets/ag_news_csv/train.processed.csv',
    index_col=0,
    converters={"text": lambda x: x.strip("[]").replace("'","").split(", ")})

# Choose elements with class 3 or 4
df = df[(df['class'] == 3) | (df['class'] == 4)]

# Map classes 3 and 4 -> 0 and 1
df['class'] = df['class'] - 3

labels = df['class'].to_numpy()

# Make sequences the same length
data = pad_sequences(df['text'], maxlen=INPUT_LENGTH)

# Randomize indices
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# %% [markdown]
# Look at the shape of the data set

# %%
x_train.shape, y_train.shape, x_val.shape, y_val.shape

# %%
sum(y_train)

# %%
sum(y_val)

# %% [markdown]
# ## Train the Simple Classifier Model

# %%
import keras

embedding_layer = model.get_keras_embedding(train_embeddings=False)

classifier_keras_model = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=(INPUT_LENGTH,)),
        embedding_layer,
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1, activation='sigmoid'),
    ]
)

classifier_keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier_keras_model.summary()

# %%
history = classifier_keras_model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_val, y_val), batch_size=10)

# %%
# classifier_keras_model.save('models/first-steps/classifier(inputl 30, l1units 16, epochs 10, batch_s 10)')

# %%
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
text1 = df['text'].loc[0]
text1 = to_token_id(text1)
text1 = keras.preprocessing.sequence.pad_sequences([text1], maxlen=INPUT_LENGTH)
text1

# %%
classifier_keras_model.predict(text1)


# %% [markdown]
# ## Train Model with Updated Parameters

# %%
import keras

embedding_layer = model.get_keras_embedding(train_embeddings=False)

classifier = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(INPUT_LENGTH,)),
        embedding_layer,
        keras.layers.Dense(units=8, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1, activation='sigmoid'),
    ]
)
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()


# %%
history = classifier.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size=10)

# %%
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% [markdown]
# ## Train Model with Updated Parameters (Again)

# %%
import keras

classifier = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(INPUT_LENGTH,)),
        embedding_layer,
        keras.layers.Dense(units=4, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1, activation='sigmoid'),
    ]
)
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()

# %%
history = classifier.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_val, y_val), batch_size=20)

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
