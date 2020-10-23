from utils.dataset import get_dataset, import_embedding_layer
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.callbacks import EarlyStopping
import utils.metrics as mt
import numpy as np
import json

def prediction_threshold(y_predict):
    return np.count_nonzero(y_predict < 0.5) / y_predict.shape[0]

def load_model(input_length, model_type=1):
    inner_model = None
    if model_type == 1:
        inner_model = Sequential([
            LSTM(units=128),
            Dense(units=32),
            Dense(units=1, activation='sigmoid'),
        ])
    elif model_type == 2:
        inner_model = Sequential([
            LSTM(units=128, return_sequences=True),
            Dropout(0.5),
            LSTM(units=64),
            Dropout(0.5),
            Dense(units=1, activation='sigmoid'),
        ])
    elif model_type == 3:
        inner_model = Sequential([
            Dense(units=8),
            Dropout(0.5),
            Flatten(),
            Dense(units=1, activation='sigmoid'),
        ])

    model = Sequential([
        InputLayer(input_shape=(input_length,)),
        import_embedding_layer(),
        inner_model
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        'Recall',
        'Precision',
    ])

    return (model, inner_model)

class BaseClassifier:
    def __init__(self, model, inner_model, id, p_weight=None, threshold=10, epochs=30, batch_size=32):
        self.id = id
        self.inner_model = inner_model
        self.model = model

        self.p_weight = p_weight

        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = [EarlyStopping(monitor='loss', patience=3, min_delta=0.0001)]

    # Path methods
    def get_metrics_path(self):
        return f'results/metrics/{self.id}.json'

    def get_weights_path(self):
        return f'results/weights/{self.id}'

    def get_history_path(self):
        return f'results/history/{self.id}.json'

    # Metric methods
    def get_prediction(self, X):
        return mt.get_prediction(self.model, X)

    def get_metrics(self, X, y_expected):
        try:
            return self.load_metrics()

        # path doesn't exist
        except IOError:
            return self.save_metrics(X, y_expected)

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
    def save_weights(self):
        # Do not save the weights of the embedding layer
        self.inner_model.save_weights(self.get_weights_path())

    def load_weights(self):
        self.inner_model.load_weights(self.get_weights_path())

    def train(self, X_train, y_train, X_test, y_test):
        print(f'Training classifier {self.id}...')

        # Create the classifier
        self.model.summary()

        # Get the dataset
        Xi, yi = get_dataset(X_train, y_train, self.id, self.p_weight)

        # Only split and train dataset if there is enough data
        if Xi.shape[0] > self.threshold:
            # Train the classifier and save the history
            history = self.model.fit(
                Xi, yi,
                epochs=self.epochs, batch_size=self.batch_size, callbacks=self.callbacks,
                verbose=1)

            with open(self.get_history_path(), 'w') as fp:
                json.dump(history.history, fp)

            # Save the weights
            self.save_weights()

        # Save the metrics
        Xi_test, yi_test = get_dataset(X_test, y_test, self.id)
        self.save_metrics(Xi_test, yi_test)

class BaseBalancedClassifier(BaseClassifier):
    def __init__(self, model, inner_model, id):
        super().__init__(model, inner_model, id, p_weight=0.5)

    def get_weights_path(self):
        return f'results/weights/{self.id}_balanced'

    def get_history_path(self):
        return f'results/history/{self.id}_balanced.json'

    def get_metrics_path(self):
        return f'results/metrics/{self.id}_balanced.json'

class BaseWeightedClassifier(BaseClassifier):
    def __init__(self, model, inner_model, id, p_weight):
        super().__init__(model, inner_model, id, p_weight)

    def get_weights_path(self):
        return f'results/weights/{self.id}_p{self.p_weight * 100}'

    def get_history_path(self):
        return f'results/history/{self.id}_p{self.p_weight * 100}.json'

    def get_metrics_path(self):
        return f'results/metrics/{self.id}_p{self.p_weight * 100}.json'

class BaseUnbalancedClassifier(BaseClassifier):
    def __init__(self, model, inner_model, id):
        super().__init__(model, inner_model, id, p_weight=None)

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
