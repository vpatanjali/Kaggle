# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:04:57 2015

@author: patanjali
"""

import sys, pandas, numpy

sys.path.append('/home/patanjali/xgboost/wrapper/')

import xgboost

from sklearn.metrics import log_loss

DATA_PATH = '/home/patanjali/Kaggle/Data/Otto/'

DEV_LIMIT = 0.5

#%% Loading data

data = pandas.read_csv(DATA_PATH+'train.csv')

idvs = list(data.columns)

idvs.remove('id')
idvs.remove('target')

idvs_data = data[idvs].values
dvs = (data['target'].str.slice(6).astype(int) -1).values

dev = numpy.random.rand(idvs_data.shape[0]) < DEV_LIMIT
val = ~dev

dev_data = xgboost.DMatrix(idvs_data[dev], label=dvs[dev])
val_data = xgboost.DMatrix(idvs_data[val], label=dvs[val])

#%% Setting parameters

param = {'max_depth':10, 'gamma':1, 'silent':1, 'objective':'multi:softprob', 'num_class':9, 'eval_metric':'mlogloss' }

param['nthread'] = 4

evallist  = [(val_data,'val'), (dev_data,'dev')]

#%% Training model

num_round = 10
bst = xgboost.train(param, dev_data, num_round, evallist)

bst.save_model('0002.model')

# dump model
#bst.dump_model('dump.raw.txt')
# dump model with feature map
#bst.dump_model('dump.raw.txt','featmap.txt')

#%% Loading and predicting

#bst = xgboost.Booster({'nthread':4}) #init model
#bst.load_model("0001.model") # load data

#%%

ypred = bst.predict(val_data)

baseline = log_loss(dvs[val],ypred)

_extra_sums = []

for i in xrange(len(idvs)):
    for j in xrange(i+1, len(idvs)):
        print i,j
        data['extra'] = data[idvs[i]] + data[idvs[j]]
        idvs_data = data[idvs + _extra_sums + ['extra']].values
        dev_data = xgboost.DMatrix(idvs_data[dev], label=dvs[dev])
        val_data = xgboost.DMatrix(idvs_data[val], label=dvs[val])
        bst = xgboost.train(param, dev_data, num_round, evallist)
        ypred = bst.predict(val_data)
        perf = log_loss(dvs[val],ypred)
        print perf, _extra_sums
        if perf < baseline:
            _extra_sums.append(str(i)+'_'+str(j))
            data[str(i)+'_'+str(j)] = data[idvs[i]] + data[idvs[j]]
            baseline = perf

#%%% Differences
for i in xrange(len(idvs)):
    for j in xrange(i+1, len(idvs)):
        print i,j
        data['extra'] = data[idvs[i]] - data[idvs[j]]
        idvs_data = data[idvs + _extra_sums + ['extra']].values
        dev_data = xgboost.DMatrix(idvs_data[dev], label=dvs[dev])
        val_data = xgboost.DMatrix(idvs_data[val], label=dvs[val])
        bst = xgboost.train(param, dev_data, num_round, evallist)
        ypred = bst.predict(val_data)
        perf = log_loss(dvs[val],ypred)
        print perf, _extra_sums
        if perf < baseline:
            _extra_sums.append(str(i)+'minus'+str(j))
            data[str(i)+'minus'+str(j)] = data[idvs[i]] + data[idvs[j]]
            baseline = perf
"""
#%%% Products            
for i in xrange(len(idvs)):
    for j in xrange(i+1, len(idvs)):
        print i,j
        data['extra'] = data[idvs[i]] + data[idvs[j]]
        idvs_data = data[idvs + _extra_sums + ['extra']].values
        dev_data = xgboost.DMatrix(idvs_data[dev], label=dvs[dev])
        val_data = xgboost.DMatrix(idvs_data[val], label=dvs[val])
        bst = xgboost.train(param, dev_data, num_round, evallist)
        ypred = bst.predict(val_data)
        perf = log_loss(dvs[val],ypred)
        print perf, _extra_sums
        if perf < baseline:
            _extra_sums.append(str(i)+'_'+str(j))
            data[str(i)+'_'+str(j)] = data[idvs[i]] + data[idvs[j]]
            baseline = perf
"""