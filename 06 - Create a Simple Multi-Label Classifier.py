# %% [markdown]
# # Create a Simple Multi-Label Classifier

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

INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 4

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

labels = df['class'].to_numpy()

# Get one-hot encoding for each label
labels = np.eye(CLASS_COUNT, dtype=int)[labels - 1]

def to_token_id(tokens):
    return [model.vocab[token].index for token in tokens if token in model.vocab]

data = df['text'].apply(to_token_id)

# Make sequences same length
data = pad_sequences(data, maxlen=INPUT_LENGTH)

# Randomize the dataset
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

train_count = int((1 - VALIDATION_SPLIT) * data.shape[0])
val_count = data.shape[0] - train_count

# Get the train and validation indices for the original dataset
train_indices = np.zeros((CLASS_COUNT, train_count // CLASS_COUNT), dtype=int)
val_indices = np.zeros((CLASS_COUNT, val_count // CLASS_COUNT), dtype=int)
for cls_id in range(0, CLASS_COUNT):
    class_indices = np.arange(data.shape[0])[labels[:, cls_id] == 1]
    train_indices[cls_id] = class_indices[:(train_count // 4)]
    val_indices[cls_id] = class_indices[(train_count // 4):]

train_indices = train_indices.flatten()
val_indices = val_indices.flatten()

# Set the data
# [x, x, x, x, x, x, x, x, o, o]
x_train = data[train_indices]
y_train = labels[train_indices]
# [o, o, o, o, o, o, o, o, x, x]
x_val = data[val_indices]
y_val = labels[val_indices]

# %% [markdown]
# Check the shape of the training and validation data set.

# %%
x_train.shape, y_train.shape, x_val.shape, y_val.shape

# %% [markdown]
# Check and see if the dataset is class-wise balanced.

# %%
y_train[y_train[:,0] == 1].shape[0],y_train[y_train[:,1] == 1].shape[0],y_train[y_train[:,2] == 1].shape[0],y_train[y_train[:,3] == 1].shape[0]

# %% [markdown]
# ## Create and train the classifier

# %%
import keras

embedding_layer = model.get_keras_embedding(train_embeddings=False)

classifier = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(INPUT_LENGTH,)),
        embedding_layer,
        keras.layers.Dense(units=8, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=4, activation='sigmoid'),
    ]
)
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()

# %%
history = classifier.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_val, y_val), batch_size=10)

# %%
classifier.predict(x_val)

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
index = np.random.choice(np.arange(y_val.shape[0]))

predicted = classifier.predict(np.array([x_val[index]])).reshape(-1)
expected = y_val[index]

w = 0.35
x = np.arange(CLASS_COUNT)
plt.bar(x - w/2, predicted, w, label='Predicted')
plt.bar(x + w/2, expected, w, label='Expected')
plt.xlabel('Class')
plt.xticks(x, (1,2,3,4))
plt.legend(loc='upper right')
plt.show()

# %%
np.sum(predicted)

# %%
predicted
