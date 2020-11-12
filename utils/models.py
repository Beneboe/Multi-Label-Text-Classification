from utils.dataset import get_dataset, import_embedding_layer
from numpy.random import default_rng
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.callbacks import EarlyStopping
import utils.storage as st
import numpy as np
import json

embedding_layer = None
def prediction_threshold(y_predict):
    return np.count_nonzero(y_predict < 0.5) / y_predict.shape[0]

def random_prediction(X, threshold):
    y_predict = default_rng(42).uniform(size=X.shape[0])
    y_predict = (y_predict >= threshold).astype('int32')
    return y_predict

def load_model(input_length, model_type=1):
    global embedding_layer
    if embedding_layer is None:
        embedding_layer = import_embedding_layer()

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
        embedding_layer,
        inner_model
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
        'accuracy',
        'Recall',
        'Precision',
    ])

    return (model, inner_model)

class Classifier:
    def __init__(self, id, type_name, p_weight, threshold=10, epochs=30, batch_size=32, skip_model=False) -> None:
        self.id = id
        self.type_name = type_name
        self.name = st.get_name(self.id, self.type_name)

        if not skip_model:
            self.model, self.inner_model = load_model(10)

        self.p_weight = p_weight

        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = [EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]

    def get_prediction(self, X):
        return self.model.predict(X).flatten()

    def save_weights(self):
        # Do not save the weights of the embedding layer
        self.inner_model.save_weights(st.get_weights_path(self.name))

    def load_weights(self):
        self.inner_model.load_weights(st.get_weights_path(self.name))

    def train(self, X_train, y_train):
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

            st.save_history(self.id, self.type_name, history.history)

            # Save the weights
            self.save_weights()

def create_classifier(id, type_name, **kwargs):
    if type_name[2:] == '%positive':
        p_weight = float(type_name[0:2]) / 100.0
        return Classifier(id, type_name, p_weight, **kwargs)
    elif type_name == 'unbalanced':
        return Classifier(id, 'unbalanced', None, **kwargs)

def create_classifiers(id, type_names, **kwargs):
    for type_name in type_names:
        yield create_classifier(id, type_name, **kwargs)
