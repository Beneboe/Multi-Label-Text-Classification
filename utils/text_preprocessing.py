import gensim
import pandas as pd
from nltk import word_tokenize

def to_token_id(tokens):
    return [model.vocab[token].index for token in tokens if token in model.vocab]

# Load the model to get the vocabulary
model = gensim.models.KeyedVectors.load_word2vec_format("datasets/first-steps/GoogleNews-vectors-negative300.bin.gz", binary=True)

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

# Lowercase the words
df['text'] = df['text'].str.lower()

# Tokenize the words
df['text'] = df['text'].map(word_tokenize)

# Remove punctuation characters
punctuation = list('()-,.')
df['text'] = df['text'].map(lambda tokens: [token for token in tokens if token not in punctuation])

# Convert words to token ids
df['text'] = df['text'].apply(to_token_id)

df.to_csv("datasets/charcnn_keras_processed.csv")