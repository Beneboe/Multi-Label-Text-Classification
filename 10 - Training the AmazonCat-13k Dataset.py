# %%
from utils.dataset import import_dataset
from utils.models import BalancedClassifier, UnbalancedClassifier, Weighted10Classifier, Weighted20Classifier
from timeit import default_timer as timer

# %% [markdown]
# # Training the AmazonCat-13k Dataset
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 100
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
    # (7083,97803) # duplicate
    (7891,88967),
    (4038,76277),
    (10063,75035),
    (12630,71667),
]

# %%
# Labels just below certain thresholds
# Order: threshold, label, frequency
threshold_data = [
    (50,6554,50),
    (100,4949,100),
    (1000,7393,996),
    (10000,84,9976),
    (50000,9202,48521),
    (100000,7083,96012),
]

# %% [markdown]
# Actually train the classifiers.

# %%
configurations = dict(
      [(8842, [BalancedClassifier, Weighted20Classifier])]
    # + [(label, [BalancedClassifier, UnbalancedClassifier]) for label,_ in top10_label_data]
    # + [(label, [BalancedClassifier, UnbalancedClassifier]) for _,label,_ in threshold_data]
    )

# %%
training_types = {
    BalancedClassifier: "balanced",
    Weighted20Classifier: "20%positive",
    Weighted10Classifier: "10%positive",
    UnbalancedClassifier: "unbalanced",
}
durations = {}

for label, classifiers in configurations.items():
    durations[label] = []
    for classifier in classifiers:
        print(f"Training classifier for label '{label}' with '{training_types[classifier]}' training")

        start = timer()
        classifier(label).train(X_train, y_train, X_test, y_test)
        end = timer()

        duration = end - start
        durations[label].append(duration)
        print(f"Training took {duration} seconds.")

# %%
