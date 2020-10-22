from utils.dataset import get_dataset
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import utils.metrics as mt
import numpy as np
import json

def prediction_threshold(y_predict):
    return np.count_nonzero(y_predict < 0.5) / y_predict.shape[0]

class BaseClassifier:
    def __init__(self, model, inner_model, id):
        self.id = id
        self.inner_model = inner_model
        self.model = model

    # Metric methods
    def get_prediction(self, X):
        return mt.get_prediction(self.model, X)

    def get_metrics(self, X, y_expected):
        try:
            return self.load_metrics()

        # path doesn't exist
        except IOError:
            return self.save_metrics(X, y_expected)

    def get_metrics_path(self):
        return f'results/metrics/{self.id}.json'

    def load_metrics(self):
        with open(self.get_metrics_path(), 'r') as fp:
            return json.load(fp)

    def save_metrics(self, X, y_expected):
        y_predict = self.get_prediction(X)

        metrics = mt.all_metrics(y_predict, y_expected)

        with open(self.get_metrics_path(), 'w') as fp:
            json.dump(metrics, fp)

        return metrics

    def get_confusion(self, X, y_expected):
        y_predict = self.get_prediction(X)
        return mt.get_confusion(y_predict, y_expected)

    # Model methods
    def get_weights_path(self):
        return f'results/weights/{self.id}'

    def save_weights(self):
        # Do not save the weights of the embedding layer
        self.inner_model.save_weights(self.get_weights_path())

    def load_weights(self):
        self.inner_model.load_weights(self.get_weights_path())

    def get_history_path(self):
        return f'results/history/{self.id}.json'

    def fit(self, *args, **kwargs):
        history = self.model.fit(*args, **kwargs)

        # Save the history
        with open(self.get_history_path(), 'w') as fp:
            json.dump(history.history, fp)
        return history

    def summary(self):
        self.model.summary()

class BaseBalancedClassifier(BaseClassifier):
    def __init__(self, model, inner_model, id):
        super().__init__(model, inner_model, id)

    def get_weights_path(self):
        return f'results/weights/{self.id}_balanced'

    def get_history_path(self):
        return f'results/history/{self.id}_balanced.json'

    def get_metrics_path(self):
        return f'results/metrics/{self.id}_balanced.json'

class BaseUnbalancedClassifier(BaseClassifier):
    def __init__(self, model, inner_model, id):
        super().__init__(model, inner_model, id)

    def get_weights_path(self):
        return f'results/weights/{self.id}_unbalanced'

    def get_history_path(self):
        return f'results/history/{self.id}_unbalanced.json'

    def get_metrics_path(self):
        return f'results/metrics/{self.id}_unbalanced.json'

class RandomClassifier:
    def __init__(self, id, threshold):
        self.id = id
        self.threshold = threshold

    # Metric methods
    def get_prediction(self, X):
        y_predict = default_rng(42).uniform(size=X.shape[0])
        y_predict = (y_predict >= self.threshold).astype('int32')
        return y_predict

    def get_metrics(self, X, y_expected):
        y_predict = self.get_prediction(X)
        metrics = mt.all_metrics(y_predict, y_expected)
        return metrics

    def get_confusion(self, X, y_expected):
        y_predict = self.get_prediction(X)
        return mt.get_confusion(y_predict, y_expected)

    def load_weights(self):
        pass

class BalancedRandomClassifier(RandomClassifier):
    def __init__(self, id):
        super().__init__(id, 0.5)

class UnbalancedRandomClassifier(RandomClassifier):
    def __init__(self, id):
        super().__init__(id, 0.5)

    def get_metrics(self, X, y_expected):
        self.threshold = prediction_threshold(y_expected)
        return super().get_metrics(X, y_expected)

    def get_confusion(self, X, y_expected):
        self.threshold = prediction_threshold(y_expected)
        return super().get_confusion(X, y_expected)

class Trainer:
    def __init__(self, Model, X_train, y_train, X_test, y_test, train_balance=True, threshold=10, epochs=30, batch_size=32):
        self.Model = Model
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.p_weight = 0.5 if train_balance else None

        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = [EarlyStopping(monitor='loss', patience=3, min_delta=0.0001)]

    def train(self, i):
        print(f'Processing classifier {i}...')

        # Create the classifier
        classifier = self.Model(i)
        classifier.summary()

        # Get the dataset
        Xi, yi = get_dataset(self.X_train, self.y_train, i, self.p_weight)

        # Only split and train dataset if there is enough data
        if Xi.shape[0] > self.threshold:
            # Train the classifier and save the history
            classifier.fit(
                Xi, yi,
                epochs=self.epochs, batch_size=self.batch_size, callbacks=self.callbacks,
                verbose=1)

            # Save the weights
            classifier.save_weights()

        # Save the metrics
        Xi_test, yi_test = get_dataset(self.X_test, self.y_test, i, balanced=False)
        classifier.save_metrics(Xi_test, yi_test)
