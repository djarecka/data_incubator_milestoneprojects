"""
code to build model for question 2
"""

import json
import dill
import pandas as pd
import numpy as np
import pdb
from sklearn import base, ensemble, pipeline


class Transformer(base.BaseEstimator, base.TransformerMixin):
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_transf = np.empty((len(X), 2))
        for i, dir_data in enumerate(X):
            X_transf[i,:] = dir_data["latitude"], dir_data["longitude"]
        
        return X_transf
        

class Estimator(base.BaseEstimator, base.RegressorMixin):

    def fit(self, X, y):
        self.random_forest = ensemble.RandomForestRegressor(min_samples_leaf=20).fit(X, y)
        return self
        
    def predict(self, dict_test):
        return self.random_forest.predict([dict_test["latitude"], dict_test["longitude"]])[0]
        

with open("list_data.txt", "r") as f:
    X_train = json.load(f)

y_train = [d['stars'] for d in X_train]
    
trans = Transformer()
X_train_trans = trans.transform(X_train)
    
est = Estimator()
est.fit(X_train_trans, y_train)

f = open("est_latlon_model", "wb")
dill.dump(est, f)
f.close()
