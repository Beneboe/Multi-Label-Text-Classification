import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm

def get_title(name):
    return ' '.join(word.capitalize() for word in name.split('_'))

def convert_cm(cm):
    tp, fp, fn, tn = cm

    categories = ['True', 'False']
    return pd.DataFrame([[tp, fp], [fn, tn]],
        index=pd.Index(categories, name='Predicted'),
        columns=pd.Index(categories, name='Actual'))

def plot_confusion(cm, name, ax=None):
    cm = convert_cm(cm)

    sns.set(font_scale=1.4)
    nax = sns.heatmap(cm,
        fmt=',d',
        annot=True,
        annot_kws={"size": 16},
        norm=LogNorm(vmin=cm.min().min(),
        vmax=cm.max().max()),
        ax=ax)

    nax.set_title(get_title(name))

def plot_history(history, name):
    for value in history.values():
        plt.plot(value)
    plt.title(f'Classifier {get_title(name)} Performance')
    plt.ylabel('Metric Performance')
    plt.xlabel('epoch')
    plt.legend(history.keys(), loc='lower right')
