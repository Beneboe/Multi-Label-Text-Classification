# Multi-label Text Classification

## Prerequisite

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
    - tst.json
    - trn.json
  - GoogleNews-vectors-negative.bin.gz
- 09 - Preprocess the AmazonCat-13k Dataset.py

Where datasets folder contains the extracted archive files. Before running the script look for the `DATASET_TYPE` variable at the top of the file. To preprocess the train set change it to 'trn', for the test set change it to 'tst'.

Finally, run the *09 - Preprocess the AmazonCat-13k Dataset.py* file.

## Training classifiers for the AmazonCat-13K dataset

Ensure that the project folder has the following directory structure:

- datasets/
  - AmazonCat-13K/
    - tst.processed.json
    - trn.processed.json
  - GoogleNews-vectors-negative.bin.gz
- results/
  - history/
  - metrics/
  - weights/
- 10 - Training the AmazonCat-13k Dataset.py

Finally, run the *10 - Training the AmazonCat-13k Dataset.py* file.

## Sample Classes that Occur at least 10,000 times

- 35 "18th century"
- 38 "19th century"
- 39 "20th century"
- 49 "21st century"
- 81 "accessories" (28555 occurences)
- 5960 "home improvement"