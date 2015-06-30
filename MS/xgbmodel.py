# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:16:58 2015

@author: patanjali
"""

import os, sys, csv

sys.path.append('/home/patanjali/xgboost/wrapper/')

import xgboost, pandas

DATA_DIR = '/home/patanjali/Kaggle/Data/MS/'
VAL_PCT = 0.2

dev_file = 'dev_bytes_asm_ntfs_0.001_0.1.libsvm'
val_file = 'val_bytes_asm_ntfs_0.001_0.1.libsvm'

test_file = 'test_bytes_asm_tfs.libsvm'

os.chdir(DATA_DIR)

#dev = xgboost.DMatrix(dev_file)
#val = xgboost.DMatrix(val_file)

#%% Training
"""
for i in xrange(1,11):
    param = {'max_depth':i, 'gamma':1, 'silent':1, 'objective':'multi:softprob', 'num_class':9, 
         'min_child_weight' : 4, 'subsample' : 0.9, 'colsample_bytree' : 0.8, 'eval_metric':'mlogloss' }
    param['nthread'] = 4
    evallist  = [(val,'val'), (dev,'dev')]
    num_round = 100
    bst = xgboost.train(param, dev, num_round, evallist)
    bst.save_model('depth_%s_0.001_0.1.model' %(i))
"""

#%% Test file for upload

test = xgboost.DMatrix(test_file+'.buffer')
#test.save_binary(test_file + '.buffer')

bst = xgboost.Booster({'nthread':4, 'num_class':9}) #init model
bst.load_model('depth_4.model') # load data

test_predictions = bst.predict(test)
preds = pandas.DataFrame(test_predictions)
preds.columns = ['Prediction'+str(i) for i in range(1,10)]
labels = sorted([x.split('.')[0] for x in os.listdir('/media/patanjali/Elements/Kaggle/Data/MS/test/') if x.split('.')[1] == 'asm'])
preds['Id'] = labels
preds.to_csv('XGB_100trees_depth4_.csv',quoting=csv.QUOTE_NONNUMERIC,index=False)
