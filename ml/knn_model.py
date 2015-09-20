"""
code to build model for question 4
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

    def flatten_dict_att(self, dict_att):
        dd_fl = {}
        for kk, vv in dict_att.items():
            if (vv == True) or (vv == False):
                dd_fl[kk] = vv
            elif isinstance(vv, basestring):
                dd_fl[kk + "_" + vv] = True
            elif isinstance(vv, dict) and len(vv)>0:
                for k, v in vv.items():
                    if (v == True) or (v == False):
                        dd_fl[kk + "." + k] = v
                    elif isinstance(v, basestring):
                        dd_fl[kk + "." + k + "_" + v] = True
            else:
                continue
        return dd_fl                                            


    def transform(self, X, y_fl=False):
        X_select = []
        for i, dic_data in enumerate(X):
            att = dic_data["attributes"]
            if len(att):
                X_select.append(self.flatten_dict_att(att))
            else:
                X_select.append({})

        return X_select
        


with open("list_data.txt", "r") as f:
    X_train = json.load(f)

y_train = [d['stars'] for d in X_train]

est_new = pipeline.Pipeline([('trans', Transformer()),
                             ('trans_vec', DictVectorizer(sparse=True)),
                             ('trans_tfidf', TfidfTransformer()),
                             ("est", linear_model.Ridge())])
est_new.fit(X_train, y_train)

f = open("est_knn_model", "wb")
dill.dump(est_new, f)
f.close()
