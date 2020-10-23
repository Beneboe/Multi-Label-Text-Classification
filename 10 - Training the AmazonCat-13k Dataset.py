# %%
from utils.dataset import import_dataset
from utils.models import BaseBalancedClassifier, BaseUnbalancedClassifier, BaseWeightedClassifier, load_model

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
model, inner_model = load_model(INPUT_LENGTH)

# %%
class BalancedClassifier(BaseBalancedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id)

class Weighted10Classifier(BaseWeightedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id, 0.10)

class Weighted20Classifier(BaseWeightedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id, 0.20)

class UnbalancedClassifier(BaseUnbalancedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id)

# %% [markdown]
# Actually train the classifiers.

# %%
# for i in range(CLASS_COUNT):
#     trainer_balanced.train(i)

# %%
print("================================================================================")
print("Train the classifier for 8842")
print("================================================================================")
BalancedClassifier(8842).train(X_train, y_train, X_test, y_test)


# %%
Weighted10Classifier(8842).train(X_train, y_train, X_test, y_test)

# %%
Weighted20Classifier(8842).train(X_train, y_train, X_test, y_test)

# %%
UnbalancedClassifier(8842).train(X_train, y_train, X_test, y_test)

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

# %%
# Top 10 most frequent labels ordered from most to least frequent
# Order: label, frequency
top10_label_data = [
    (1471,355211)
    (7961,194561)
    (7892,128026)
    (9237,120090)
    # (7083,97803) # duplicate
    (7891,88967)
    (4038,76277)
    (10063,75035)
    (12630,71667)
]

# %%
print()
print("================================================================================")
print("Balanced training for different threshold labels")
print("================================================================================")
for _,label,_ in threshold_data:
    BalancedClassifier(label).train(X_train, y_train, X_test, y_test)

# %%
print()
print("================================================================================")
print("Unbalanced training for different threshold labels")
print("================================================================================")
for _,label,_ in threshold_data:
    UnbalancedClassifier(label).train(X_train, y_train, X_test, y_test)

# %%
print()
print("================================================================================")
print("Balanced training for most frequent labels")
print("================================================================================")
for label,_ in top10_label_data:
    BalancedClassifier(label).train(X_train, y_train, X_test, y_test)

# %%
print()
print("================================================================================")
print("Unbalanced training for most frequent labels")
print("================================================================================")
for label,_ in top10_label_data:
    UnbalancedClassifier(label).train(X_train, y_train, X_test, y_test)
