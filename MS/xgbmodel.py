# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:16:58 2015

@author: patanjali
"""

import os, sys

sys.path.append('/home/patanjali/xgboost/wrapper/')

import xgboost

DATA_DIR = '/home/patanjali/Kaggle/Data/MS/'
VAL_PCT = 0.2
dev_file = 'dev_bytes_asm_tfs.libsvm.buffer'
val_file = 'val_bytes_asm_tfs.libsvm.buffer'

os.chdir(DATA_DIR)

dev = xgboost.DMatrix(dev_file)
val = xgboost.DMatrix(val_file)

#%%

for i in xrange(1,11):
    param = {'max_depth':i, 'gamma':1, 'silent':0, 'objective':'multi:softprob', 'num_class':9, 
         'min_child_weight' : 4, 'subsample' : 0.9, 'colsample_bytree' : 0.8, 'eval_metric':'mlogloss' }
    param['nthread'] = 4
    evallist  = [(val,'val'), (dev,'dev')]
    num_round = 100
    bst = xgboost.train(param, dev, num_round, evallist)
    bst.save_model('depth_%s.model' %(i))
