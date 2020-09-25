from keras import Sequential
from keras.layers import LSTM, Dense, InputLayer, Dropout, Flatten
import utils.metrics as mt
import json

WEIGHTS_FILE_TEMPLATE = 'results/weights/cl_class={0}'
HISTORY_FILE_TEMPLATE = 'results/history/cl_class={0}.json'
METRICS_FILE_TEMPLATE = 'results/metrics/cl_class={0}.json'

class EmbeddingClassifier(Sequential):
    def __init__(self, embedding_layer, input_length):
        super(LSTMModel, self).__init__()

        self.add(InputLayer(input_shape=(input_length,)))
        self.add(embedding_layer)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Do not save the weights of the embedding layer
        return self.inner_model.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        return self.inner_model.load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

class LSTMModel(EmbeddingClassifier):
    def __init__(self, embedding_layer, input_length):
        super(LSTMModel, self).__init__(embedding_layer, input_length)

        self.inner_model = Sequential([
            LSTM(units=128, return_sequences=True),
            Dropout(0.5),
            LSTM(units=64),
            Dropout(0.5),
            Dense(units=1, activation='sigmoid'),
        ])
        self.add(self.inner_model)

class DenseClassifier(EmbeddingClassifier):
    def __init__(self, embedding_layer, input_length):
        super(DenseClassifier, self).__init__(embedding_layer, input_length)

        self.inner_model = Sequential([
            Dense(units=8, activation='relu'),
            Flatten(),
            Dense(units=1, activation='sigmoid'),
        ])
        self.add(self.inner_model)

def get_metrics(classifier, X, y_expected):
    y_predict = mt.get_prediction(classifier, X)

    return {
        'count': mt.count(y_predict, y_expected),
        'accuracy': mt.accuracy(y_predict, y_expected),
        'recall': mt.recall(y_predict, y_expected),
        'precision': mt.precision(y_predict, y_expected),
        'f1 measure': mt.f1measure(y_predict, y_expected),
    }

def save_weights(classifier, i):
    classifier.save_weights(WEIGHTS_FILE_TEMPLATE.format(i))

def load_weights(classifier, i):
    classifier.load_weights(WEIGHTS_FILE_TEMPLATE.format(i))

def save_metrics(metrics, i):
    with open(METRICS_FILE_TEMPLATE.format(i), 'w') as fp:
        json.dump(metrics, fp)

def load_metrics(i):
    with open(METRICS_FILE_TEMPLATE.format(i), 'r') as fp:
        return json.load(fp)

def save_history(history, i):
    with open(HISTORY_FILE_TEMPLATE.format(i), 'w') as fp:
            json.dump(history.history, fp)
