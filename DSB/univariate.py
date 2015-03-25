# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:27:07 2015

@author: patanjali
"""

import pandas
from sklearn.metrics import roc_auc_score

dataset = pandas.read_csv('/home/patanjali/Clintara/s3/clintara_meta_and_audio1N2_features_v2.csv')

#%%

for field in dataset.columns:
    if field not in ['Final Score'] and dataset[field].dtype in ['int64', 'float64']:
        score = roc_auc_score(dataset['Final Score'], dataset[field])
        if score < 0.5:
            score = -1*roc_auc_score(dataset['Final Score'], -1*dataset[field])
        if score > 0.61:
            print field, score
    else:
        pass#print field
#%%