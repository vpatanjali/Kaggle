# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/patanjali/.spyder2/.temp.py
"""

import os
import pandas

from pandas.io.pytables import HDFStore, get_store
from utils import *

data_dir = '/home/patanjali/Kaggle/Data/Fire/'

os.chdir(data_dir)

train = pandas.read_hdf('idvs_v3.h5','train')
predictors = pandas.read_hdf('idvs_v3.h5','predictors')
predictors = list(predictors[0].values)
predictors2 = predictors + ['weights']
predictors_ = [x for x in predictors if x[:3] == 'var'] + ['weights']

#%%

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,\
                        GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

#%% Sampling schemes

dev = (train.id%5!=0)
val = train.id%5==0

#%%
"""
tuned_parameters = {'min_samples_leaf' : [100],
                    'max_depth' : [3],
                    'n_estimators' : [100],
                    'max_features' : ['sqrt'],
                    #'loss' : ['ls'],#, 'lad','huber','quantile'],
                    'learning_rate' : [0.05],
                    #'subsample' : [0.3,0.5,0.8,1]
                    }

model = GridSearchCV(GradientBoostingClassifier(#learning_rate = 0.05,
                                               verbose=5),
                     tuned_parameters, scoring = wg,
                     cv=StratifiedKFold(train['bin_dv'].values,n_folds = 3),
                     verbose = 3, n_jobs = 3, pre_dispatch = 3)

"""
#%%

N = 5
model = range(N)
for i in xrange(N):
    model[i] = GradientBoostingRegressor(max_features = 'sqrt', subsample = .8,
                                        max_depth = 2, n_estimators = 100,
                                        learning_rate = 0.05,
                                        min_samples_leaf = 10, verbose = 0)
    dev = (train.id%N!=i)
    val = train.id%N==i
    model[i].fit(train.ix[dev][predictors_],train.ix[dev]['target'])
    train['pred_dv'] = model[i].predict(train[predictors_].values)
    print "Iteration %s of %s " %(i+1, N), normalized_weighted_gini2(train.ix[dev]['pred_dv'],
                                                train.ix[dev]['target'],
                                                train.ix[dev]['weights']),\
        normalized_weighted_gini2(train.ix[val]['pred_dv'],
                                                train.ix[val]['target'],
                                                train.ix[val]['weights'])

#%%
for i in xrange(N):
    dev = (train.id%N!=i)
    val = train.id%N==i
    j = 0
    for predictions in model[i].staged_predict(train[predictors_].values):
        j += 1
        train['pred_dv'] = predictions
        print "Iteration %s of %s, step %s " %(i+1, N, j), \
            normalized_weighted_gini2(train.ix[dev]['pred_dv'],
                                                train.ix[dev]['target'],
                                                train.ix[dev]['weights']),\
            normalized_weighted_gini2(train.ix[val]['pred_dv'],
                                                train.ix[val]['target'],
                                                train.ix[val]['weights'])
    
#%%
#figure()  
#plot_ks(train.ix[dev].pred_dv,train.ix[dev].dv0,train.ix[dev2].dv,train.ix[dev2].weights)
#figure()
#plot_ks(train.ix[val].pred_dv,train.ix[val].dv0,train.ix[val].dv,train.ix[val].weights)
#figure()
#plot_ks(-1*train.var13,train.dv0,train.dv,train.weights)

#%%

print normalized_weighted_gini(-1*train.var13,
                                            train['target_scaled'],
                                            train['weights_scaled'])

#%%
train_ginis = []
test_ginis = []
i = 0
for predictions in model.staged_predict(train[predictors_].values):
    i += 1
    print i
    train['pred_dv'] = predictions
    train_ginis.append(normalized_weighted_gini(train.ix[dev2]['pred_dv'],
                                            train.ix[dev2]['target'],
                                            train.ix[dev2]['weights']))
    test_ginis.append(normalized_weighted_gini(train.ix[val]['pred_dv'],
                                            train.ix[val]['target'],
                                            train.ix[val]['weights']))
figure()
plot(train_ginis)
plot(test_ginis)
#%%
for s in model.grid_scores_:
    print s
#%%
"""

train['pred_dv'] = model.predict_proba(train[predictors].values)[:,1]
test['target'] = model.predict_proba(test[predictors].values)[:,1]

i = 0
for preds in model.staged_predict_proba(train[predictors]):
    i += 1
    print i
    train['pred_dv'] = preds[:,1]
    train_ginis.append(normalized_weighted_gini(train.ix[dev]['pred_dv'],
                                                train.ix[dev]['target'],
                                                train.ix[dev]['var11']))
    
    test_ginis.append(normalized_weighted_gini(train.ix[val]['pred_dv'],
                                               train.ix[val]['target'],
                                               train.ix[val]['var11']))
"""
#%%
train_ginis = []
test_ginis = []
i = 0;j=0

print normalized_weighted_gini2(train.ix[dev]['pred_dv'],
                                            train.ix[dev]['target'],
                                            train.ix[dev]['weights'])

print normalized_weighted_gini2(train.ix[dev2]['pred_dv'],
                                            train.ix[dev2]['target'],
                                            train.ix[dev2]['weights'])
                                            
print normalized_weighted_gini2(train.ix[val]['pred_dv'],
                                           train.ix[val]['target'],
                                           train.ix[val]['weights'])

#%%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
print classification_report(train.ix[dev]['dv0'],train.ix[dev]['var13'])
print classification_report(train.ix[dev]['dv0'],train.ix[dev]['pred_dv'])
print roc_auc_score(train.ix[dev]['dv0'],train.ix[dev]['var13'])
print roc_auc_score(train.ix[dev]['dv0'],train.ix[dev]['pred_dv'])
print roc_auc_score(train.ix[dev2]['dv0'],train.ix[dev2]['var13'])
print roc_auc_score(train.ix[dev2]['dv0'],train.ix[dev2]['pred_dv'])
print roc_auc_score(train.ix[val]['dv0'],train.ix[val]['var13'])
print roc_auc_score(train.ix[val]['dv0'],train.ix[val]['pred_dv'])

#%%

test = pandas.read_hdf('idvs_v3.h5','test')
test['target'] = 0
for i in xrange(N):
    test['target'] = test['target'] + model[i].predict(test[predictors2])

test.to_csv('output_idvs3_dv_target_gbc_cvmean_3_learning_rate-0.1_max_depth-2_max_features-sqrt_n_estimators-100_min_samples_leaf-100.csv',
            columns=['id','target'],index=False)

#%%
"""
i = 1

for preds in model.staged_predict(test[predictors]):
    print i
    if i == best_iter:
        test['target'] = preds
        break
    i+= 1

test.to_csv('output_abc_nestimators-2.csv',columns=['id','target'],index=False)

#%%

for var in predictors:
    print var, normalized_weighted_gini(train[var], train.ix[dev]['target'],
                                                train.ix[dev]['var11'])
"""
#%%
nwg = []
coeffs = []
#%%
for i in xrange (100):
    print i
    nwg.append(normalized_weighted_gini(train['pred_dv']-i/100*train['weights'],
                                            train['target'],
                                            train['weights']))
    coeffs.append(-i/100)
    
#%%

#%timeit weighted_gini(train[idv1],train['target'],train['weights'])

#%%
_ws = train['weights'].sum()
_wts = numpy.dot(train['weights'],train['target'])
train['weights_scaled'] = train['weights']/_ws
train['target_scaled'] = train['target']*_ws/_wts

#%%

print weighted_gini(train['pred_dv'],train['target_scaled'],train['weights_scaled'])

#%%
i = 0
for idv1 in predictors2:
    for idv2 in predictors2:
        plus = weighted_gini(train[idv1]-train[idv2],train['target_scaled'],train['weights_scaled'])
        prod = weighted_gini(train[idv1]/train[idv2],train['target_scaled'],train['weights_scaled'])
        if abs(plus) > 0.25 or abs(prod) > 0.25:
            print idv1, idv2, plus, prod
                
