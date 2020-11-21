# %%
import json
import utils.storage as st
import utils.dataset as ds
from utils.models import create_classifiers
from itertools import chain
from timeit import default_timer as timer

# %% [markdown]
# # Training on the AmazonCat-13k Dataset

# %%
INPUT_LENGTH = 10
CLASS_COUNT = 13330

# %% [markdown]
# Import the dataset and the embedding layer

# %%
X_train, y_train = ds.import_amazoncat13k('trn.processed', INPUT_LENGTH)
X_test, y_test = ds.import_amazoncat13k('tst', INPUT_LENGTH)

# %% [markdown]
# Actually train the classifiers.

# %%
# Get unique labels
labels = set(ds.amazoncat13k_top10_labels  + ds.amazoncat13k_threshold_labels)

def classifiers():
    return chain(
        create_classifiers(8842, ['50%positive', '20%positive', '10%positive', 'unbalanced']),
        chain(*[create_classifiers(label, ['50%positive', 'unbalanced']) for label in labels]),
    )

# %%
durations = {}

for c in classifiers():
    durations[c.name] = []

    print(f"Training classifier '{c.name}'.")

    start = timer()
    c.train(X_train, y_train)
    end = timer()

    duration = end - start
    durations[c.id].append(duration)
    print(f"Training took {duration} seconds.")

    Xi_test, yi_test = ds.get_dataset(X_test, y_test, c.id)
    yi_predict = c.get_prediction(Xi_test)
    st.save_prediction(c.id, c.type_name, yi_predict)

# %%
with open("results/durations.json", 'w') as fp:
    json.dump(durations, fp)

# %%
