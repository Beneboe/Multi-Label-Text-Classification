# %% [markdown]
# # Training the AmazonCat-13k Dataset
# First, setup the hyperparameters.

# %%
INPUT_LENGTH = 100
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
# CLASS_COUNT = 30
TRAIN_PATH = 'datasets/AmazonCat-13K/trn.processed.json'
TEST_PATH = 'datasets/AmazonCat-13K/tst.processed.json'
EPOCHS = 10
TRAINING_THRESHOLD = 10

# %% [markdown]
# Import the dataset and the embedding layer

# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset

X_train, y_train = import_dataset(TRAIN_PATH, INPUT_LENGTH)
X_test, y_test = import_dataset(TEST_PATH, INPUT_LENGTH)
embedding_layer = import_embedding_layer()

# %% [markdown]
# Define the model

# %%
from utils.models import BaseClassifier
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives

inner_model = Sequential([
    Dense(units=8),
    Flatten(),
    Dense(units=1, activation='sigmoid'),
])

model = Sequential([
    InputLayer(input_shape=(INPUT_LENGTH,)),
    embedding_layer,
    inner_model
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
    'accuracy',
    Recall(),
    Precision(),
])

class Classifier(BaseClassifier):
    def __init__(self, id):
        super(Classifier, self).__init__(model, inner_model, id)

# %% [markdown]
# ## Create, fit, and save the classifier Models
# Define the steps.

# %%
from utils.models import save_history
from sklearn.model_selection import train_test_split
import keras.metrics as kmt

def train_classifier(i):
    print(f'Processing classifier {i}...')

    # Create the classifier
    classifier = Classifier(i)
    classifier.summary()

    # Get the dataset
    Xi, yi = get_dataset(X_train, y_train, i)

    # Only split and train dataset if there is enough data
    if Xi.shape[0] > TRAINING_THRESHOLD:
        # Split the dataset
        # Xi_train, Xi_train_test, yi_train, yi_train_test = train_test_split(
            # Xi, yi, test_size=VALIDATION_SPLIT, random_state=42)

        # Train the classifier
        # history = classifier.fit(Xi_train, yi_train,
            # epochs=EPOCHS, verbose=1, validation_data=(Xi_train_test, yi_train_test), batch_size=20)
        history = classifier.fit(Xi, yi, epochs=EPOCHS, verbose=1, batch_size=20)

        # Save the history
        save_history(history, i)

        # Save the weights
        classifier.save_weights()

    # Save the metrics
    Xi_test, yi_test = get_dataset(X_test, y_test, i, balanced=False)
    classifier.save_metrics(Xi_test, yi_test)

# %% [markdown]
# Actually train the classifiers.

# %%
for i in range(CLASS_COUNT):
    train_classifier(i)

# %%
# train_classifier(81)