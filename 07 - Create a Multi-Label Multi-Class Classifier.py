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

INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 4

df = pd.read_csv("datasets/charcnn_keras_processed.csv",
    index_col=0,
    converters={"text": lambda x: x.strip("[]").replace("'","").split(", ")})

labels = df['class'].to_numpy()

# Get one-hot encoding for each label
labels = np.eye(CLASS_COUNT, dtype=int)[labels - 1]

# make sequences same length
data = pad_sequences(df['text'], maxlen=INPUT_LENGTH)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)

# randomize the dataset
data = data[indices]
labels = labels[indices]

train_count = int((1 - VALIDATION_SPLIT) * data.shape[0])
val_count = data.shape[0] - train_count

# get the train and validation indices for the original dataset
train_indices = np.zeros((CLASS_COUNT, train_count // CLASS_COUNT), dtype=int)
val_indices = np.zeros((CLASS_COUNT, val_count // CLASS_COUNT), dtype=int)
for cls_id in range(0, CLASS_COUNT):
    class_indices = np.arange(data.shape[0])[labels[:, cls_id] == 1]
    train_indices[cls_id] = class_indices[:(train_count // 4)]
    val_indices[cls_id] = class_indices[(train_count // 4):]

train_indices = train_indices.flatten()
val_indices = val_indices.flatten()

# set the data
# [x, x, x, x, x, x, x, x, o, o]
x_train = data[train_indices]
y_train = labels[train_indices]
# [o, o, o, o, o, o, o, o, x, x]
x_val = data[val_indices]
y_val = labels[val_indices]
