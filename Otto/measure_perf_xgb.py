# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:06:56 2015

@author: patanjali
"""

import pandas

from sklearn.metrics import log_loss

import os

os.chdir('/home/patanjali/K/Otto/')

val = pandas.read_csv('pred.txt',header=None,names=['pred']).values
test_raw = pandas.read_csv('test.csv')

preds = val.reshape(val.shape[0]/9,9)

#actuals = pandas.read_csv('val.libsvm', sep = ' ', names = ['dv']+[str(x) for x in range(93)])

#print log_loss(actuals.dv,preds)

preds = pandas.DataFrame(preds)

preds.columns = ['Class_' + str(i) for i in range(1,10)]
preds['id']  = test_raw['id']

preds.to_csv('xgboost2.csv',index=False)