"""
creates a full model for question 5 (use latlon_model, category_model, knn_model)
couldn't tested this version with grader (some errors), used the full_model_pip
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


with open("list_data.txt", "r") as f:
    X_train = json.load(f)

y_train = [d['stars'] for d in X_train]


import knn_model as knn
import latlon_model as lat
import category_model as cat
import city_model as cit

est_new = pipeline.Pipeline([
    ('features', pipeline.FeatureUnion([
              ('cat_trans', pipeline.Pipeline([
                  ('trans', cat.Transformer()),
                  ('trans_vec', DictVectorizer(sparse=True)),
                  ('trans_tfidf', TfidfTransformer())])),
             ('knn_trans', pipeline.Pipeline([
                 ('trans', knn.Transformer()),
                 ('trans_vec', DictVectorizer(sparse=True)),
                 ('trans_tfidf', TfidfTransformer())])),
            ('latlon_trans', lat.Transformer())
    ])),
      ('est', linear_model.Ridge())
    ])


est_new.fit(X_train, y_train)

f = open("est_full_model", "wb")
dill.dump(est_new, f)
f.close()
