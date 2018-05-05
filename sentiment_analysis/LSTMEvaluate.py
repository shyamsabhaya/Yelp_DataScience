from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from keras.models import load_model

print("Reading Data")

data = pd.read_csv("filtered_data.csv")
data = data.dropna(axis=0, how='any')

y = [0 if rating <= 3 else 1 for rating in data['stars']]

print("Splitting")
# same random state as in training so that we get the same test set
x_train, x_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.20, random_state=17)

max_features = 100000

print("Fitting tokenizer to training data")
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train.values)

print("Tokenizing test set")
x_test = pad_sequences(tokenizer.texts_to_sequences(x_test.values))

print("Loading Keras model")
model = load_model("lstm.h5")

print("Getting ROC score")
auc = roc_auc_score(y_test, model.predict(x_test))

print("AUC score on test set: " + str(auc))

neg_sentiment = ["This place sucks"]
positive_sentiment = ["I love the food here. The staff is friendly and the service is fast"]

print("Prediction for \"This place sucks\" : " + str(model.predict(pad_sequences(tokenizer.texts_to_sequences(neg_sentiment)))))
print("Prediction for \"I love the food here. The staff is friendly and the service is fast\" : " + str(model.predict(pad_sequences(tokenizer.texts_to_sequences(positive_sentiment)))))
