# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.storage as st
import utils.metrics as mt
import utils.dataset as ds
from itertools import product

# %% [markdown]
# # Analyze Different Label Frequency Thresholds

# %%
INPUT_LENGTH = 10
X_test, y_test = ds.import_amazoncat13k('tst', INPUT_LENGTH)

# %% [markdown]
# Gather the metrics

# %%
labels = ds.amazoncat13k_threshold_labels
thresholds = ds.amazoncat13k_thresholds

ye = np.array([ds.get_dataset(X_test, y_test, label)[1] for label in labels]).T

b_yp = np.array([st.load_prediction(label, '50%positive') for label in labels]).T
ub_yp = np.array([st.load_prediction(label, 'unbalanced') for label in labels]).T

b_precisions = [mt.precision(b_yp[:, i] , ye[:, i]) for i in range(len(labels))]
b_recalls = [mt.recall(b_yp[:, i], ye[:, i]) for i in range(len(labels))]

ub_precisions = [mt.precision(ub_yp[:, i] , ye[:, i]) for i in range(len(labels))]
ub_recalls = [mt.recall(ub_yp[:, i], ye[:, i]) for i in range(len(labels))]

# %% [markdown]
# Graph the performance of each thresholds

# %%
# formatted_thresholds = ['{:,d}'.format(threshold) for threshold in thresholds]
formatted_thresholds = ['{:}\n({:,d})'.format(label, thresholds) for label, thresholds in zip(labels, thresholds)]

line1, = plt.plot(formatted_thresholds, b_precisions, 'bs', label = 'Balanced Precision')
line2, = plt.plot(formatted_thresholds, b_recalls, 'b^', label = 'Balanced Recall')
# line3, = plt.plot(formatted_thresholds, ub_precisions, 'ys', label = 'Unbalanced Precision')
# line4, = plt.plot(formatted_thresholds, ub_recalls, 'y^', label = 'Unbalanced Recall')

plt.title('L2 Deconstructed Performance')
plt.xlabel('Labels of L2')
plt.ylabel('Metric Performance')
plt.legend()
plt.grid()

plt.savefig('results/imgs/l2_deconstructed_performance.png', dpi=163)
plt.savefig('results/imgs/l2_deconstructed_performance.pdf')
plt.show()

# %%
# Create the table for each of the thresholds

table = pd.DataFrame({
    'label': labels,
    'threshold': thresholds,
    'balanced precision': b_precisions,
    'balanced recall': b_recalls,
    'unbalanced precision': ub_precisions,
    'unbalanced recall': ub_recalls,
})
table.to_csv('results/imgs_data/l2_deconstructed_performance.csv')
table

# %%
