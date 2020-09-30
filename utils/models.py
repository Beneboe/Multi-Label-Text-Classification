from utils.dataset import get_dataset
from os.path import isfile
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
from sklearn.model_selection import train_test_split
import utils.metrics as mt
import keras.metrics as kmt
import json

WEIGHTS_FILE_TEMPLATE = 'results/weights/cl_class={0}'
HISTORY_FILE_TEMPLATE = 'results/history/cl_class={0}.json'
METRICS_FILE_TEMPLATE = 'results/metrics/cl_class={0}.json'

class BaseClassifier:
    def __init__(self, model, inner_model, id):
        self.id = id
        self.inner_model = inner_model
        self.model = model

    def save_weights(self):
        # Do not save the weights of the embedding layer
        self.inner_model.save_weights(WEIGHTS_FILE_TEMPLATE.format(self.id))

    def load_weights(self):
        self.inner_model.load_weights(WEIGHTS_FILE_TEMPLATE.format(self.id))

    # metric methods
    def get_metrics(self, X, y_expected):
        try:
            return self.load_metrics()

        # path doesn't exist
        except IOError:
            return self.save_metrics(X, y_expected)

    def load_metrics(self):
        with open(METRICS_FILE_TEMPLATE.format(self.id), 'r') as fp:
            return json.load(fp)

    def save_metrics(self, X, y_expected):
        y_predict = mt.get_prediction(self.model, X)

        metrics = {
            'count': mt.count(y_predict, y_expected),
            'accuracy': mt.accuracy(y_predict, y_expected),
            'recall': mt.recall(y_predict, y_expected),
            'precision': mt.precision(y_predict, y_expected),
            'f1 measure': mt.f1measure(y_predict, y_expected),
        }

        with open(METRICS_FILE_TEMPLATE.format(self.id), 'w') as fp:
            json.dump(metrics, fp)

        return metrics

    def get_prediction(self, X):
        return mt.get_prediction(self.model, X)

    def get_confusion(self, X, y_expected):
        y_predict = self.get_prediction(X)
        return mt.get_confusion(y_predict, y_expected)

    # model methods
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def summary(self):
        self.model.summary()

def save_history(history, id):
    with open(HISTORY_FILE_TEMPLATE.format(id), 'w') as fp:
        json.dump(history.history, fp)

class Trainer:
    def __init__(self, Model, X_train, y_train, X_test, y_test, threshold=10, epochs=30, batch_size=32):
        self.Model = Model
        self.threshold = threshold
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, i):
        print(f'Processing classifier {i}...')

        # Create the classifier
        classifier = self.Model(i)
        classifier.summary()

        # Get the dataset
        Xi, yi = get_dataset(self.X_train, self.y_train, i)

        # Only split and train dataset if there is enough data
        if Xi.shape[0] > self.threshold:
            # Train the classifier
            history = classifier.fit(Xi, yi, epochs=self.epochs, verbose=1, batch_size=self.batch_size)

            # Save the history
            save_history(history, i)

            # Save the weights
            classifier.save_weights()

        # Save the metrics
        Xi_test, yi_test = get_dataset(self.X_test, self.y_test, i, balanced=False)
        classifier.save_metrics(Xi_test, yi_test)
