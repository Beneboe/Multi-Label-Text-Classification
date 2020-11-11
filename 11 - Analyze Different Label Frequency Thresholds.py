# %%
from utils.dataset import get_dataset, import_amazoncat13k
import pandas as pd
import matplotlib.pyplot as plt
import utils.storage as st
import utils.metrics as mt

# %% [markdown]
# # Analyze Different Label Frequency Thresholds

# %%
INPUT_LENGTH = 10
X_test, y_test = import_amazoncat13k('tst', INPUT_LENGTH)

# %%
threshold_data = [
    (50,6554,50),
    (100,4949,100),
    (1000,7393,996),
    (10000,84,9976),
    (50000,9202,48521),
    (100000,7083,96012),
]
thresholds, labels, frequencies = zip(*threshold_data)

# %% [markdown]
# Gather the metrics

# %%
def get_metric_data(label):
    _, yi_expt = get_dataset(X_test, y_test, label)
    yi_pred = st.load_prediction(label, '50%positive')
    return (mt.precision(yi_expt, yi_pred), mt.recall(yi_expt, yi_pred))

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
plt.savefig('results/imgs/label_frequency_performance.png', dpi=163)
plt.savefig('results/imgs/label_frequency_performance.pdf')
plt.show()

# %%
# Create the table for each of the thresholds

table = pd.DataFrame(all_metric_data, index=formatted_thresholds, columns=['precision', 'recall'])
table.to_csv('results/imgs_data/label_frequency_performance.csv')

# %%
