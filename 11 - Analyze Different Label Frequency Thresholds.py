# %%
from utils.models import BaseBalancedClassifier
from keras import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # Analyze Different Label Frequency Thresholds
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 10
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
# CLASS_COUNT = 30
TRAIN_PATH = 'datasets/AmazonCat-13K/trn.processed.json'
TEST_PATH = 'datasets/AmazonCat-13K/tst.processed.json'
EPOCHS = 30
TRAINING_THRESHOLD = 2

# %%
# Model 1
inner_model = Sequential([])
model = Sequential([])

# %% [markdown]
# Define the model

class Classifier(BaseBalancedClassifier):
    def __init__(self, id):
        super().__init__(model, inner_model, id)

# %%
threshold_data = [
    (50,6554,50),
    (100,4949,100),
    (1000,7393,996),
    (10000,84,9976),
    (50000,9202,48521),
    # (100000,7083,96012),
]

thresholds, labels, frequencies = zip(*threshold_data)

# %% [markdown]
# Gather the metrics

# %%
def get_metric_data(label):
    classifier = Classifier(label)
    metrics = classifier.load_metrics()

    return (metrics['precision'], metrics['recall'])

all_metric_data = [get_metric_data(label) for label in labels]
precisions, recalls = zip(*all_metric_data)

# %% [markdown]
# Graph the performance of each thresholds

# %%
formatted_thresholds = ['{:,d}'.format(threshold) for threshold in thresholds]

plt.plot(formatted_thresholds, precisions, 'bs')
plt.plot(formatted_thresholds, recalls, 'y^')
plt.title('Classifier Frequency Performance')
plt.xlabel('Label Frequency Levels')
plt.legend(['Precision', 'Recall'])
plt.grid()
plt.savefig(f'datasets/imgs/AmazonCat-13K_classifier_frequency_performance.png', dpi=163)
plt.show()

# %%
# Create the table for each of the thresholds

pd.DataFrame(all_metric_data, index=formatted_thresholds, columns=['precision', 'recall'])
