# %% [markdown]
# # Setup a Simple Deep Learning Classifier using Keras

# %% [markdown]
# ## Import the Embedding Layer

# %%
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/first-steps/GoogleNews-vectors-negative300.bin.gz", binary=True)

# %% [markdown]
# ## Prepare the Data Set
# Only use instances where the class is either 3 or 4.

# %%
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2

# Import the dataset
colnames = ['class', 'text', 'description']

# The csv has no header row therefore header=None
df = pd.read_csv('datasets/first-steps/charcnn_keras.csv', header=None, names=colnames)
df['text'] = df['text'].astype(pd.StringDtype())

# Join the description column on the text column
df['text'] = df['text'].str.cat(df['description'], sep=" ")

# Remove the description column
df = df[['class', 'text']]

# Replace double slashes with a space
df['text'] = df['text'].str.replace("\\", " ")

# Choose elements with class 3 or 4
df = df[(df['class'] == 3) | (df['class'] == 4)]

# Map classes 3 and 4 -> 0 and 1
df['class'] = df['class'] - 3

labels = df['class'].to_numpy()

def to_token_id(tokens):
    return [model.vocab[token].index for token in tokens if token in model.vocab]

data = df['text'].apply(to_token_id)
# Make sequences the same length
data = pad_sequences(data, maxlen=INPUT_LENGTH)

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
# ## Train a Classifier Model

# %%
input_dim = x_train.shape[1]

embedding_layer = model.get_keras_embedding(train_embeddings=False)
embedding_layer.input_length = INPUT_LENGTH

classifier_keras_model = keras.models.Sequential(
    [
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
classifier_keras_model.save('models/first-steps/classifier(inputl 30, l1units 16, epochs 10, batch_s 10)')


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