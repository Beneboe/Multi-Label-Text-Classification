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

def preprocess_amazoncat13k(
        dataset,
        dataset_path=None,
        before_path=None,
        after_path=None,
        token_threshold=0,
        append_content=False):
    
    suffix = ''
    if append_content:
        suffix = '.content'

    if dataset_path is None:
        dataset_path = f'datasets/AmazonCat-13K/{dataset}.json'
    
    if before_path is None:
        before_path = f'datasets/AmazonCat-13K/{dataset}{suffix}.before.json'
    
    if after_path is None:
        after_path = f'datasets/AmazonCat-13K/{dataset}{suffix}.processed.json'

    df = pd.read_json(dataset_path, lines=True)

    if append_content:
        df['title'] = df['title'].str.cat(df['content'], sep=' ')

    # Actual preprocessing
    X = preprocess(df['title'])
    y = df['target_ind']

    df_before = pd.DataFrame({ 'X': X, 'y': y })
    df_before.to_json(before_path, orient='records', lines=True)

    token_lens = X.map(len)
    remove_indices = np.arange(X.shape[0])[token_lens > token_threshold]
    remove_count = token_lens.shape[0] - remove_indices.shape[0]

    print('Instances with less than or equal to {0} tokens will be removed.'.format(token_threshold))
    print('This amounts to {0} instances ({1:.2%}).'.format(remove_count, remove_count / token_lens.shape[0]))

    # Remove instances under token_threshold
    X = X[remove_indices]
    y = y[remove_indices]

    df_after = pd.DataFrame({ 'X': X, 'y': y })
    df_after.to_json(after_path, orient='records', lines=True)



if __name__ == "__main__":

    # Import the dataset
    colnames = ['class', 'text', 'description']

    # The csv has no header row therefore header=None
    df = pd.read_csv('datasets/ag_news_csv/train.csv', header=None, names=colnames)
    df['text'] = df['text'].astype(pd.StringDtype())

    # Join the description column on the text column
    df['text'] = df['text'].str.cat(df['description'], sep=" ")

    # Remove the description column
    df = df[['class', 'text']]

    # Replace double slashes with a space
    df['text'] = df['text'].str.replace("\\", " ")

    df['text'] = preprocess(df['text'])

    df.to_csv("datasets/ag_news_csv/train.processed.csv")