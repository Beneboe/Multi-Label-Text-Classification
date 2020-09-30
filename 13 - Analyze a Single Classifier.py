# %%
from utils.dataset import import_dataset, import_embedding_layer, get_dataset
from utils.plots import confusion
from utils.models import BaseClassifier, Trainer
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten,InputLayer
from keras.metrics import Recall, Precision, TrueNegatives, TruePositives
import matplotlib.pyplot as plt

# %%
INPUT_LENGTH = 10
VALIDATION_SPLIT = 0.2
CLASS_COUNT = 13330
# CLASS_COUNT = 30
TRAIN_PATH = 'datasets/AmazonCat-13K/trn.processed.json'
TEST_PATH = 'datasets/AmazonCat-13K/tst.processed.json'
EPOCHS = 30
TRAINING_THRESHOLD = 2
CLASS = 8842

# %%
X_train, y_train = import_dataset(TRAIN_PATH, INPUT_LENGTH)
X_test, y_test = import_dataset(TEST_PATH, INPUT_LENGTH)
embedding_layer = import_embedding_layer()

# %%
# Alternative 1
inner_model = Sequential([
    LSTM(units=256),
    Dense(units=64),
    Dense(units=1, activation='sigmoid'),
])

# Alternative 2
# inner_model = Sequential([
#     LSTM(units=128, return_sequences=True),
#     Dropout(0.5),
#     LSTM(units=64),
#     Dropout(0.5),
#     Dense(units=1, activation='sigmoid'),
# ])

# Alternative 3
# inner_model = Sequential([
#     Dense(units=4),
#     Dropout(0.5),
#     Flatten(),
#     Dense(units=1, activation='sigmoid'),
# ])

model = Sequential([
    InputLayer(input_shape=(INPUT_LENGTH,)),
    embedding_layer,
    inner_model
])

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
    'accuracy',
    Recall(),
    Precision(),
])

class Classifier(BaseClassifier):
    def __init__(self, id):
        super(Classifier, self).__init__(model, inner_model, id)

trainer = Trainer(Classifier, X_train, y_train, X_test, y_test)

# %%
trainer.train(CLASS)

# %%
classifier = Classifier(CLASS)
classifier.load_weights()

Xi, y_expected = get_dataset(X_test, y_test, CLASS, False)
(tp, fp, fn, tn) = classifier.get_confusion(Xi, y_expected)
confusion(tp, fp, fn, tn)
plt.savefig('datasets/imgs/classifier_8842_confusion.png', dpi=163)
