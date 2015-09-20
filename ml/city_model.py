""" 
code to buil a city model (only estimator used) 
and transformer that is used in the full model
"""

import pandas as pd
import numpy as np
import pdb
import dill
from sklearn import base

class Estimator:
    def __init__(self):
        self.df = pd.read_csv("city_stas.csv")

    def fit(self):
        # simple mean using pandas 
        self.citymean = self.df.groupby(["city"])["stars"].mean()
        return self
        
    def predict(self, dir_test):
        if dir_test["city"] in self.citymean.keys():
            return float(self.citymean[dir_test["city"]])
        else:
            return 0.


est = Estimator()
est.fit()

f = open("est_city_model", "wb")
dill.dump(est, f)
f.close()


# this Transformer is for full model
class Transformer(base.BaseEstimator, base.TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transf = [d['city'] for d in X]
        return X_transf
                                                            
