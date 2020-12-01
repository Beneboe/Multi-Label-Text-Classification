# %%
import json
import utils.dataset as ds
import numpy as np
import pandas as pd

durations = None
with open('results/durations.json', 'r') as file:
    durations = json.load(file)

# %%
l1 = ds.amazoncat13k_top10_labels
l1_ub = [str(label) + '_unbalanced' for label in l1]
l1_b = [str(label) + '_50%positive' for label in l1]

l2 = ds.amazoncat13k_threshold_labels
l2_ub = [str(label) + '_unbalanced' for label in l2]
l2_b = [str(label) + '_50%positive' for label in l2]

# %%
d_l1_ub = [durations[label] for label in l1_ub]
d_l1_b = [durations[label] for label in l1_b]

d_l2_ub = [durations[label] for label in l2_ub]
d_l2_b = [durations[label] for label in l2_b]

# %%
c_l1 = ds.amazoncat13k_top10_label_counts
c_l2 = ds.amazoncat13k_threshold_label_counts

# %%

set_t_corr = [
    {
        'unbalanced': np.corrcoef(c_l1, d_l1_ub)[0,1],
        'balanced': np.corrcoef(c_l1, d_l1_b)[0,1],
    },
    {
        'unbalanced': np.corrcoef(c_l2, d_l2_ub)[0,1],
        'balanced': np.corrcoef(c_l2, d_l2_b)[0,1],
    }
]

pd.DataFrame(set_t_corr, ['L1', 'L2'])

# %%
