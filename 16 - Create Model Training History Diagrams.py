from utils.plots import plot_history
import matplotlib.pyplot as plt
import json

# %%
CLASS = 8842

# %%
keras_model_class_names = [f'{CLASS}_{name}' for name in ['50%positive', '20%positive', '10%positive', 'unbalanced']]

# %% [markdown]
# Plot the history diagrams

# %%
for name in keras_model_class_names:
    history = None
    with open(f'results/history/{name}.json', 'r') as fp:
        history = json.load(fp)
    plot_history(history, name)
    plt.tight_layout()
    plt.savefig(f'datasets/imgs/classifier_{name}_history.png', dpi=163)
    plt.savefig(f'datasets/imgs/classifier_{name}_history.svg')
    plt.show()
