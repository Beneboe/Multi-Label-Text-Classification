# %%
from utils.dataset import get_dataset, import_dataset
from utils.models import create_classifiers
from itertools import chain
from timeit import default_timer as timer
import utils.storage as st

# %% [markdown]
# # Training on the AmazonCat-13k Dataset
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 10
CLASS_COUNT = 13330

# %% [markdown]
# Import the dataset and the embedding layer

# %%
X_train, y_train = import_dataset('datasets/AmazonCat-13K/trn.processed.json', INPUT_LENGTH)
X_test, y_test = import_dataset('datasets/AmazonCat-13K/tst.processed.json', INPUT_LENGTH)

# %% [markdown]
# Define the model

# %%
# Top 10 most frequent labels ordered from most to least frequent
# Order: label, frequency
top10_label_data = [
    (1471,355211),
    (7961,194561),
    (7892,128026),
    (9237,120090),
    (7083,97803),
    (7891,88967),
    (4038,76277),
    (10063,75035),
    (12630,71667),
    (8108,71667),
]
top10_labels, _ = zip(*top10_label_data)

# %%
# Labels just below certain frequency thresholds
# Order: threshold, label, frequency
threshold_data = [
    (50,6554,50),
    (100,4949,100),
    (1000,7393,996),
    (10000,84,9976),
    (50000,9202,48521),
    # (100000,7083,96012), # duplicate
]
_, threshold_labels, _ = zip(*threshold_data)

# %% [markdown]
# Actually train the classifiers.

# %%
def classifiers():
    return chain(
        create_classifiers(8842, ['50%positive', '20%positive', '10%positive', 'unbalanced']),
        chain(*[create_classifiers(label, ['50%positive', 'unbalanced']) for label in top10_labels]),
        chain(*[create_classifiers(label, ['50%positive', 'unbalanced']) for label in threshold_labels]),
    )

# %%
durations = {}

for c in classifiers():
    durations[c.id] = []

    print(f"Training classifier '{c.name}'.")

    start = timer()
    c.train(X_train, y_train)
    end = timer()

    duration = end - start
    durations[c.id].append(duration)
    print(f"Training took {duration} seconds.")

    Xi_test, yi_test = get_dataset(X_test, y_test, c.id)
    yi_predict = c.get_prediction(Xi_test)
    st.save_prediction(c.id, c.type_name, yi_predict)

# %%
