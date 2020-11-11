import gensim
import pandas as pd
import numpy as np
from nltk import word_tokenize
from string import punctuation

# Load the model to get the vocabulary
model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin", binary=True)

def to_token_id(tokens):
    return [model.vocab[token].index for token in tokens if token in model.vocab]

def from_token_ids(ids):
    return [model.index2word[id] for id in ids]

def preprocess(series):
    # Lowercase the words
    series = series.str.lower()

    # Tokenize the words
    series = series.map(word_tokenize)

    # Remove punctuation characters
    series = series.map(lambda tokens: [token for token in tokens if token not in punctuation])

    # Convert words to token ids
    series = series.apply(to_token_id)

    return series

def preprocess_ag_news(dataset, dataset_path=None, output_path=None):
    if dataset_path is None:
        dataset_path = f'datasets/ag_news_csv/{dataset}.csv'

    if output_path is None:
        output_path = f'datasets/ag_news_csv/{dataset}.processed.csv'

    # Import the dataset
    colnames = ['class', 'text', 'description']

    # The csv has no header row therefore header=None
    df = pd.read_csv(dataset_path, header=None, names=colnames)
    df['text'] = df['text'].astype(pd.StringDtype())

    # Join the description column on the text column
    df['text'] = df['text'].str.cat(df['description'], sep=' ')

    # Remove the description column
    df = df[['class', 'text']]

    # Replace double slashes with a space
    df['text'] = df['text'].str.replace("\\", " ")

    df['text'] = preprocess(df['text'])

    df.to_csv(output_path)
