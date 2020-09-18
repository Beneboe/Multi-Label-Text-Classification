# Multi-label Text Classification

## Prerequisite

- Python 3.8
- All the modules in `requirements.txt`
- The datasets
  - [GoogleNews Word embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
  - [AG new corpus](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)
  - [AmazonCat-13K](https://drive.google.com/file/d/17rVRDarPwlMpb3l5zof9h34FlwbpTu4l)

## Preprocessing the AmazonCat-13K dataset

Ensure that the project folder has the following directory structure:

- datasets/
  - AmazonCat-13K/
    - tst.json
    - trn.json
  - GoogleNews-vectors-negative.bin.gz

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

Finally, run the *10 - Training the AmazonCat-13k Dataset.py* file.

## Sample Classes that Occur at least 10,000 times

- 35 "18th century"
- 38 "19th century"
- 39 "20th century"
- 49 "21st century"
- 81 "accessories" (28555 occurences)
- 5960 "home improvement"

## Todos

### Todo 17-09

- [x] Create a test file for the metrics file
- [x] Generate a develop dataset
- [x] Implement missing steps in train function
- [x] Do a test run
- [x] After the cutoff calculate the statistics again

### Todo 18-09

- [x] Solve issue where there are classes that never occur in trn.json because of cutoff
- [x] Investigate (potential) low accuracy issue
- [x] Write handover protocol
