# %%
import gensim

# Load the pretrained word embeddings model
model = gensim.models.KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)

# %%
type(model)
# %%
len(model.index2entity)
# %%
model.vocab["cat"].index
# %%
len(model['cat'])
# %%
model["cat"]
# %%
linreg = model['king'] - model['man'] + model['woman']
linreg
# %%
model.similar_by_vector(linreg, topn=10)
# %%
model.similarity('cat', 'dog')
# %%
model.similarity('laptop', 'pen')
# %%
model.most_similar(positive='computer', topn=10)