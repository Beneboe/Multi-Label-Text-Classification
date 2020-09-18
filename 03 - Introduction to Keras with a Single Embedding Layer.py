# %% [markdown]
# # Introduction to Keras with a Single Embedding Layer

# %% [markdown]
# ## Import the Embedding Layer

# %%
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)

# %% [markdown]
# ## See How the Embedding Layer Works
# [Embedding layer](https://keras.io/api/layers/core_layers/embedding/)

# %%
import keras

embedding_layer = model.get_keras_embedding(train_embeddings=False)
basic_keras_model = keras.models.Sequential(
    [
        embedding_layer
    ]
)
basic_keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [markdown]
# Check if the keras predicted vector is equal to the real vector

# %%
my_words = "cat dog".split()
my_words_id = [model.vocab[word].index for word in my_words]

cat_id = my_words_id[my_words.index("cat")]

keras_predicted = basic_keras_model.predict([cat_id])[0,0] # id for cat
(model["cat"] == keras_predicted).all()


# %%
dog_id = my_words_id[my_words.index("dog")]

basic_keras_model.predict([cat_id, dog_id])[0,0]
