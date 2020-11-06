import itertools
from utils.dataset import get_dataset, import_embedding_layer
from numpy.random import default_rng
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

class Classifier:
    def __init__(self, id) -> None:
        self.id = id

    def get_name(self) -> str:
        return self.id

    def get_metrics_path(self) -> str:
        return f'results/metrics/{self.get_name()}.json'

    def get_weights_path(self) -> str:
        return f'results/weights/{self.get_name()}'

    def get_history_path(self) -> str:
        return f'results/history/{self.get_name()}.json'

    # Metric methods
    def get_prediction(self, X):
        pass

    def get_metrics(self, X, y_expected):
        y_predict = self.get_prediction(X)
        return mt.all_metrics(y_predict, y_expected)

    def load_metrics(self):
        with open(self.get_metrics_path(), 'r') as fp:
            return json.load(fp)

    def save_metrics(self, X, y_expected):
        metrics = self.get_metrics(X, y_expected)
        with open(self.get_metrics_path(), 'w') as fp:
            json.dump(metrics, fp)

        return metrics

    def get_confusion(self, X, y_expected):
        y_predict = self.get_prediction(X)
        return mt.get_confusion(y_predict, y_expected)

    # Model methods
    def save_weights(self):
        pass

    def load_weights(self):
        pass

    def train(self, X_train, y_train, X_test, y_test):
        pass

class KerasClassifier(Classifier):
    def __init__(self, id, p_weight=None, threshold=10, epochs=30, batch_size=32, skip_model=False):
        super().__init__(id)

        if not skip_model:
            self.model, self.inner_model = load_model(10)

        self.p_weight = p_weight

        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = [EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]

    # Metric methods
    def get_prediction(self, X):
        return mt.get_prediction(self.model, X)

    # Model methods
    def save_weights(self):
        # Do not save the weights of the embedding layer
        self.inner_model.save_weights(self.get_weights_path())

    def load_weights(self):
        self.inner_model.load_weights(self.get_weights_path())

    def train(self, X_train, y_train, X_test, y_test):
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
                validation_split=0.2,
                verbose=1)

            with open(self.get_history_path(), 'w') as fp:
                json.dump(history.history, fp)

            # Save the weights
            self.save_weights()

        # Save the metrics
        Xi_test, yi_test = get_dataset(X_test, y_test, self.id)
        self.save_metrics(Xi_test, yi_test)

class BaseWeightedClassifier(KerasClassifier):
    def __init__(self, id, p_weight, **kwargs):
        super().__init__(id, p_weight, **kwargs)

    def get_name(self):
        percent = int(self.p_weight * 100)
        return f'{self.id}_{percent}%positive'

class Weighted10Classifier(BaseWeightedClassifier):
    def __init__(self, id, **kwargs):
        super().__init__(id, 0.10, **kwargs)

class Weighted20Classifier(BaseWeightedClassifier):
    def __init__(self, id, **kwargs):
        super().__init__(id, 0.20, **kwargs)

class Weighted50Classifier(BaseWeightedClassifier):
    def __init__(self, id, **kwargs):
        super().__init__(id, 0.5, **kwargs)

class UnbalancedClassifier(KerasClassifier):
    def __init__(self, id, **kwargs):
        super().__init__(id, p_weight=None, **kwargs)

    def get_name(self):
        return f'{self.id}_unbalanced'

class RandomClassifier(Classifier):
    def __init__(self, id, threshold):
        self.id = id
        self.threshold = threshold

    # Metric methods
    def get_prediction(self, X):
        y_predict = default_rng(42).uniform(size=X.shape[0])
        y_predict = (y_predict >= self.threshold).astype('int32')
        return y_predict

class Weighted50RandomClassifier(RandomClassifier):
    def __init__(self, id):
        super().__init__(id, 0.5)

    def get_name(self):
        return f'{self.id}_50%positive_random'

class UnbalancedRandomClassifier(RandomClassifier):
    def __init__(self, id):
        super().__init__(id, 0.5)

    def get_metrics(self, X, y_expected):
        self.threshold = prediction_threshold(y_expected)
        return super().get_metrics(X, y_expected)

    def get_confusion(self, X, y_expected):
        self.threshold = prediction_threshold(y_expected)
        return super().get_confusion(X, y_expected)

    def get_name(self):
        return f'{self.id}_unbalanced_random'

all_keras_classes = [
    Weighted50Classifier,
    Weighted10Classifier,
    Weighted20Classifier,
    UnbalancedClassifier,
]

all_random_classes = [
    Weighted50RandomClassifier,
    UnbalancedRandomClassifier,
]

def keras_classifiers(id, classes=all_keras_classes, skip_model=False):
    for c in classes:
        classifier = c(id, skip_model)
        if not skip_model:
            classifier.load_weights()
        yield classifier

def random_classifiers(id, classes=all_random_classes):
    for c in classes:
        yield c(id)

def classifiers(id, keras_classes=all_keras_classes, random_classes=all_random_classes, skip_model=False):
    return itertools.chain(
        keras_classes(id, keras_classes, skip_model),
        random_classes(id, random_classes)
    )
