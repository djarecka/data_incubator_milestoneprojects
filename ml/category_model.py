"""
code to build a model for question 3
"""

import json
import pandas as pd
import numpy as np
import pdb
from sklearn import base, ensemble, pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn import linear_model
import dill

class Transformer(base.BaseEstimator, base.TransformerMixin):
        
    def fit(self, X, y=None): 
        return self
        
    def transform(self, X):
        X_select = []
        for i, dic_data in enumerate(X):
            X_select.append({k: 1 for k in dic_data["categories"]})
        return X_select
        

with open("list_data.txt", "r") as f:
    X_train = json.load(f)

y_train = [d['stars'] for d in X_train]

est_new = pipeline.Pipeline([('trans', Transformer()),
                             ('trans_vec', DictVectorizer(sparse=True)),
                             ('trans_tfidf', TfidfTransformer()),
                             ("est", linear_model.Ridge())])
est_new.fit(X_train, y_train)

f = open("est_category_model", "wb")
dill.dump(est_new, f)
f.close()
