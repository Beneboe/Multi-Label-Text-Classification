# Multi-label Text Classification

## Requirements

- Python 3.8
- All the modules in `requirements.txt`
- The datasets
  - [GoogleNews Word embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
  - [AG new corpus](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)
  - [AmazonCat-13K](https://drive.google.com/file/d/17rVRDarPwlMpb3l5zof9h34FlwbpTu4l)

Before we can use NLTK for tokenization some steps need to be completed. Open a new python session and run:

```python
import nltk
nltk.download('punkt')
```

## Preprocessing the AmazonCat-13K dataset

Ensure that the project folder has the following directory structure:

- datasets/
  - AmazonCat-13K/
    - `tst.json`
    - `trn.json`
  - `GoogleNews-vectors-negative.bin.gz`
- `09 - Preprocess the AmazonCat-13k Dataset.py`

The datasets folder contains the extracted archive files. Run task `09 - Preprocess the AmazonCat-13k Dataset`. This should create the following files in the `datasets/AmazonCat-13K` folder:

- `X.trn.raw.npy` (an ndarray)
- `Y.trn.raw.npz` (a csc sparse matrix)
- `X.trn.processed.npy` (an ndarray)
- `Y.trn.processed.npz` (a csc sparse matrix)
- `X.tst.npy` (an ndarray)
- `Y.tst.npz` (a csc sparse matrix)

## Training classifiers with the AmazonCat-13K dataset

Ensure that the project folder has the following directory structure:

- datasets/
  - AmazonCat-13K/
    - `X.trn.processed.npy`
    - `Y.trn.processed.npz`
    - `X.tst.npy`
    - `Y.tst.npz`
  - `GoogleNews-vectors-negative.bin.gz`
- results/
  - history/
  - predict/
  - weights/
- `10 - Training the AmazonCat-13k Dataset.py`

Before running the training, the dataset needs to be [preprocessed](#preprocessing-the-amazoncat-13k-dataset). To train the models, run the task `10 - Training the AmazonCat-13k Dataset`.

## Important labels

The top 10 most frequent labels are

| Label id | Label | # of occurences |
| - | - | - |
| 1471 | books | 355211 |
| 7961 | music | 194561 |
| 7892 | movies & tv | 128026 |
| 9237 | pop | 120090 |
| 7083 | literature & fiction | 97803 |
| 7891 | movies | 88967 |
| 4038 | education & reference | 76277 |
| 10063 | rock | 75035 |
| 12630 | used & rental textbooks | 71667 |
| 8108 | new | 71667 |

Labels that occur at most

| Label id | Label | threshold | # of occurences |
| - | - | - | - |
| 6554 | john | 50 | 50 |
| 4949 | fountains | 100 | 100 |
| 7393 | marriage | 1,000 | 996 |
| 84 | accessories & supplies | 10,000 | 9976 |
| 9202 | politics & social sciences | 50,000 | 48521 |
| 7083 | literature & fiction | 100,000 | 96012 |
