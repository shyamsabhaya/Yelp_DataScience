# Header Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as sk
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew, skewtest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import SVC
from sklearn import svm


def rms(testY,results):
    value = np.sqrt(mean_squared_error(testY,results))
    print ("RMS: ", value)
    return;


data = pd.read_csv('TrainingDataSet.csv')
data.drop(['Unnamed: 0'],axis=1,inplace = True)
data['review_rating'] = data['review_rating'] - 1

X = data.drop(['review_rating'],axis =1)
y = data['review_rating']
trainX, testX, trainY, testY = sk.train_test_split(X,y,train_size = .8, random_state = 12)

model = RidgeClassifierCV()
model.fit((trainX),(trainY))
results = model.predict(testX)
rms((testY),results)
print (results[:10])

model = SVC()
model.fit((trainX),(trainY))
results = model.predict(testX)
rms((testY),results)
print (results[:10])
