# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:03:24 2015

@author: patanjali
"""

import pandas
import os
import numpy

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfTransformer

os.chdir('/home/patanjali/Kaggle/Otto/')

train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')

train['dv'] = train.target.str.slice(-1).astype('int')

del train['target']

idvs = list(train.columns)
idvs.remove('dv')
idvs.remove('id')

dv = 'dv'

dev = numpy.random.rand(train.shape[0])<0.8
dev2 = numpy.random.rand(train.shape[0])<0.2
val = ~dev

#%%

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(train[dev2][idvs])

tfidfs = tfidf_transformer.transform(train[idvs])
tfidfs = tfidfs.todense()

idfs = []

for i, idv in enumerate(idvs):
    print i
    train[idv+'_idf'] = tfidfs[:,i]
    idfs.append(idv+'_idf')

#%%
interactions = []

dd = train[dev]

counter = 0

for idv1 in idvs:
    for idv2 in idvs:
        #train[idv+'_0'] = (train[idv]==0)*1
        ll = min([log_loss((dd.dv==i)*1,(dd[idv1]-dd[idv2]).astype('float').values) for i in range(1,10)])
        if ll < 1.5 and idv1 != idv2:
            counter += 1
            interaction = 'diff_'+idv1+'_'+idv2
            interactions.append(interaction)
            train[interaction] = train[idv1] - train[idv2]
            print counter, interaction, ll

for i, idv1 in enumerate(idvs)        :
    for idv2 in idvs[i:]:
        ll = min([log_loss((dd.dv==i)*1,(dd[idv1]+dd[idv2]).astype('float').values) for i in range(1,10)])
        if ll < 1.5 and idv1 != idv2:
            counter += 1
            interaction = 'sum_'+idv1+'_'+idv2
            interactions.append(interaction)
            train[interaction] = train[idv1] + train[idv2]
            print counter, interaction, ll
        
        ll = min([log_loss((dd.dv==i)*1,(dd[idv1]*dd[idv2]).astype('float').values) for i in range(1,10)])
        if ll < 1.5 and idv1 != idv2:
            counter += 1
            interaction = 'prod_'+idv1+'_'+idv2
            interactions.append(interaction)
            train[interaction] = train[idv1] + train[idv2]
            print counter, interaction, ll
            
    
#idvs = idvs + [temp + '_0' for temp in idvs]

#%%

grid = {'n_estimators' : [10,50,100,250],
#        'criterion' : ['gini','entropy'],
#        'max_features' : [None]+[2**n for n in range(1,7)],
        'max_depth' : range(5,8),
        'min_samples_leaf' : [2**n for n in range(7)],
        'learning_rate' : [0.1,0.01,0.005]
}

clf = GradientBoostingClassifier()
cv = StratifiedKFold(train[dev].dv, n_folds = 4)

model = GridSearchCV(clf, grid, n_jobs=4, verbose = 2, cv=cv)

model.fit(train[dev][idvs],train[dev][dv])

preds = model.predict_proba(train[idvs])

print log_loss(train[dev][dv],preds[dev,:])
print log_loss(train[val][dv],preds[val,:])

test_predictions = pandas.DataFrame(model.predict_proba(test[idvs]))

#%%

feats = idvs
clf = GradientBoostingClassifier(max_depth=7, warm_start=True, min_samples_split = 400,
                                 subsample=0.8, min_samples_leaf = 200, learning_rate = 0.1,
                                 verbose=0)
for i in xrange(1,50):
    ret = clf.set_params(n_estimators = 10*i)
    ret = clf.fit(train[feats][dev],train[dv][dev])
    preds = clf.predict_proba(train[feats])
    print i, log_loss(train[dv][dev],preds[dev,:]), log_loss(train[dv][val],preds[val,:])
    
#%%

for stage, preds in enumerate(clf.staged_predict_proba(train[feats])):
    print stage, log_loss(train[dv][dev],preds[dev,:]), log_loss(train[dv][val],preds[val,:])

#%%

for stage, preds in enumerate(clf.staged_predict_proba(test[feats])):
    print stage
    if stage == 270:
        test_predictions = pandas.DataFrame(preds)#pandas.DataFrame(clf.predict_proba(test[idvs]))
        test_predictions.columns = ['Class_'+str(i) for i in range(1,10)]
        test_predictions['id'] = test['id']
        break

test_predictions.to_csv('GB_270trees_depth8_lr0.1_subsample0.8_minsampleleaf200_min_sample_split_400_nodev.csv',index=False)