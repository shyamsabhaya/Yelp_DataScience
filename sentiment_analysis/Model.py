# Header Imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as sk
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import skew, skewtest
from sklearn import metrics


def rms(testY,results):
        value = np.sqrt(mean_squared_error(testY,results))
        print ( "RMS: ", value)
        return

if __name__ ==  '__main__':
    print ("Started Script")


    data = pd.read_csv('AUC_data.csv')
    data.drop(['Unnamed: 0'],axis=1,inplace = True)

    X = data.drop(['review_rating'],axis =1)
    y = data['review_rating']
    trainX, testX, trainY, testY = sk.train_test_split(X,y,train_size = .8, test_size = .2, random_state = 12)

    print("Loaded CSV")
    boost = xgb.XGBClassifier(n_estimators=500, nthreads = 12, n_jobs = 12, min_child_weight=1, subsample = .8, max = 5, gamma=5, colsample_bytree=1.0)
    print ("Starting Fit")
    boost.fit(trainX, trainY)
    print("Fit Complete")

    results =boost.predict_proba(testX)[:,1]
    print("AUC: ", metrics.roc_auc_score(testY,results))
    results = boost.predict(testX)
    print("Accuracy: ", metrics.accuracy_score(testY,results))
    xgb.plot_importance(boost)
    plt.show()
    print("Feature Importance Shown")
    pickle.dump(boost, open("pima.pickle.dat", "wb"))
    print("Done")
    '''
    #fpr, tpr, thresholds = metrics.roc_curve(testY, results, pos_label = 2)
    print("AUC: ", metrics.roc_auc_score(testY,results))

    # A parameter grid for XGBoost
    params = {
        'min_child_weight': [1, 3, 5, 7, 9, 11],
        'gamma': [0, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
        'colsample_bytree':  [0.2, 0.4, 0.6, 0.8, 1.0],
        'max_depth': [3, 5 , 7]
        }  

    random_search = RandomizedSearchCV(boost, param_distributions=params, n_iter=200, scoring='roc_auc', n_jobs=12, cv=5, verbose=3, random_state=1001 ) 
    random_search.fit(trainX,trainY)
    print (random_search.best_params_)
    results = boost.predict_proba(testX)[:,1]

    #fpr, tpr, thresholds = metrics.roc_curve(testY, results, pos_label = 2)
    print("AUC: ", metrics.roc_auc_score(testY,results))
    '''
