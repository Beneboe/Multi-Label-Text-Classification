
# %% [markdown]
# # Creating a Word Embeddings Model
# Create a new word embeddings model using the CharCnn_Keras dataset.
# Importing the csv data with [panda's read_csv](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table).

# %% [markdown]
# ## Preparing the Input Data
# Accessing DataFrame's columns and combining them to create a new corpus.

# %%
import pandas as pd
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
df

# %%
df.loc[0]['text'] # get first item in pandas series

# %% [markdown]
# [Working with text data in pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html).

# %%
import nltk

df['text'] = df['text'].str.lower()
# tokenize the words
df['text'] = df['text'].map(nltk.word_tokenize)
# remove punctuation characters
punctuation = list('()-,.')
df['text'] = df['text'].map(lambda tokens: [token for token in tokens if token not in punctuation])
# TODO: single occurence words
# TODO: try different models with different word occurence thresholds
df

# %%
df.loc[0]['text'] # get first item in pandas series

# %% [markdown]
# ## Training a New Gensim Model

# %%
import gensim.models
model2 = gensim.models.Word2Vec(sentences=df['text'],size=150,window=3)


# %%
man = model2.wv['man']
man

# %%
len(man)

# %%
linreg2 = model2.wv['king'] - model2.wv['man'] + model2.wv['woman']
linreg2

# %%
model2.wv.similar_by_vector(linreg2, topn=5)

# %%
model2.wv.similarity('cat', 'dog')

# %%
model2.wv.similarity('laptop', 'pen')


# %%
model2.wv.most_similar(positive='computer', topn=10)

# %%
model2.wv.most_similar(positive='well', topn=10)

# %%
model2.wv['tree']
# %%
