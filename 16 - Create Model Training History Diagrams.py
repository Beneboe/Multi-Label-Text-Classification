# %%
from utils.plots import plot_history
import utils.storage as st
import matplotlib.pyplot as plt

# %%
CLASS = 8842

# %%
type_names = ['50%positive', '20%positive', '10%positive', 'unbalanced']

# %% [markdown]
# Plot the history diagrams

# %%
for type_name in type_names:
    name = st.get_name(CLASS, type_name)
    history = st.load_history(CLASS, type_name)
    plot_history(history, name)
    plt.tight_layout()
    plt.savefig(f'results/imgs/classifier_{name}_history.png', dpi=163)
    plt.savefig(f'results/imgs/classifier_{name}_history.pdf')
    plt.show()
