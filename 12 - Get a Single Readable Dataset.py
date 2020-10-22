# %%
import numpy as np
import pandas as pd
from utils.dataset import import_dataset, get_dataset
from utils.text_preprocessing import from_token_ids

PATH = 'datasets/AmazonCat-13K/tst.processed.json'
OUTPUT_PATH = 'datasets/AmazonCat-13K/tst(8842).processed.text.json'
CLASS = 8842

X, y = import_dataset(PATH, 10)
Xi, yi = get_dataset(X, y, 8842)

mXi = np.apply_along_axis(from_token_ids, 1, Xi)

df = pd.DataFrame({ 'X': mXi.tolist(), 'y': yi })
df.to_json(OUTPUT_PATH, orient='records', lines=True)