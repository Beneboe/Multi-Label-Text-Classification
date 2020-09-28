import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from matplotlib.colors import LogNorm


def history(hist):
    plt.plot(hist.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['test'], loc='upper left')
    plt.show()

def confusion(tp, fp, fn, tn):
    cm = np.array([tp, fp, fn, tn]).reshape((2,2))
    df_cm = pd.DataFrame(cm,
        ['predicted true', 'predict false'],
        ['actual true', 'actual false'])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm,
        fmt=',d',
        annot=True,
        annot_kws={"size": 16},
        norm=LogNorm(vmin=cm.min().min(),
        vmax=cm.max().max()))
    plt.show()