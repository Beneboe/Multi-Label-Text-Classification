# %%
import pandas as pd
import matplotlib.pyplot as plt
import utils.storage as st
import utils.metrics as mt
import utils.dataset as ds

# %% [markdown]
# # Analyze Different Label Frequency Thresholds

# %%
INPUT_LENGTH = 10
X_test, y_test = ds.import_amazoncat13k('tst', INPUT_LENGTH)

# %% [markdown]
# Gather the metrics

# %%
def get_metric_data(label):
    _, yi_expt = ds.get_dataset(X_test, y_test, label)
    yi_pred = st.load_prediction(label, '50%positive')
    return (mt.precision(yi_pred, yi_expt), mt.recall(yi_pred, yi_expt))

all_metric_data = [get_metric_data(label) for label in ds.amazoncat13k_threshold_labels]
precisions, recalls = zip(*all_metric_data)

# %% [markdown]
# Graph the performance of each thresholds

# %%
formatted_thresholds = ['{:,d}'.format(threshold) for threshold in ds.amazoncat13k_thresholds]

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
