from os.path import isfile
from keras import Sequential
from keras.layers import LSTM, Dense, InputLayer, Dropout, Flatten
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
import utils.metrics as mt
import json

WEIGHTS_FILE_TEMPLATE = 'results/weights/cl_class={0}'
HISTORY_FILE_TEMPLATE = 'results/history/cl_class={0}.json'
METRICS_FILE_TEMPLATE = 'results/metrics/cl_class={0}.json'

class EmbeddingClassifier(Sequential):
    def __init__(self, embedding_layer, input_length):
        super(EmbeddingClassifier, self).__init__()

        self.add(InputLayer(input_shape=(input_length,)))
        self.add(embedding_layer)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Do not save the weights of the embedding layer
        return self.inner_model.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        return self.inner_model.load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

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

# class LSTMModel(EmbeddingClassifier):
#     def __init__(self, embedding_layer, input_length):
#         super(LSTMModel, self).__init__(embedding_layer, input_length)

#         self.inner_model = Sequential([
#             LSTM(units=128, return_sequences=True),
#             Dropout(0.5),
#             LSTM(units=64),
#             Dropout(0.5),
#             Dense(units=1, activation='sigmoid'),
#         ])
#         self.add(self.inner_model)

# class DenseClassifier(EmbeddingClassifier):
#     def __init__(self, embedding_layer, input_length):
#         super(DenseClassifier, self).__init__(embedding_layer, input_length)

#         self.inner_model = Sequential([
#             Dense(units=8, activation='relu'),
#             Flatten(),
#             Dense(units=1, activation='sigmoid'),
#         ])
#         self.add(self.inner_model)

def save_history(history, id):
    with open(HISTORY_FILE_TEMPLATE.format(id), 'w') as fp:
        json.dump(history.history, fp)