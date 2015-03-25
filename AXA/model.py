# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:39:33 2015

@author: patanjali
"""

import pandas, numpy
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.externals.joblib import Parallel, delayed

global ds, idvs, dv

ds = pandas.read_csv('idvs_smoothing_2_tr_5.csv', memory_map=True)
ds = ds.replace([numpy.inf, -numpy.inf],numpy.nan)
outfile = 'submission_idvs_smoothing_2_tr_5_other_1_part_idvs_v2.csv'

#%%

idvs = ['x_range', 'y_range', 'x_distance', u'y_distance', 'total_block_distance',\
         'total_distance', 'trip_duration', 'avg_x_speed', 'avg_y_speed', 'avg_speed', \
         'speed_1','speed_5', 'speed_10', 'speed_25', 'speed_50', 'speed_75', 'speed_90', 'speed_95','speed_99',\
         'acc_1','acc_5', 'acc_10', 'acc_25', 'acc_50', 'acc_75', 'acc_90', 'acc_95','acc_99',\
         'jerk_1','jerk_5', 'jerk_10', 'jerk_25', 'jerk_50', 'jerk_75', 'jerk_90', 'jerk_95','jerk_99',\
         'tang_acc_1','tang_acc_5', 'tang_acc_10', 'tang_acc_25', 'tang_acc_50', 'tang_acc_75', 'tang_acc_90', 'tang_acc_95','tang_acc_99',\
         'rad_acc_1','rad_acc_5', 'rad_acc_10', 'rad_acc_25', 'rad_acc_50', 'rad_acc_75', 'rad_acc_90', 'rad_acc_95','rad_acc_99',\
         'turning_rad_1','turning_rad_5', 'turning_rad_10', 'turning_rad_25', 'turning_rad_50', 'turning_rad_75', 'turning_rad_90', 'turning_rad_95','turning_rad_99']

idvs += ['num_turns']

#idvs = list(ds.columns)
#idvs.remove('id')
#idvs.remove('dv')

dv = 'dv'

#%%

submission = open(outfile,'w')
submission.write('driver_trip,prob\n')
submission.close()

def build_transform(driver):
    global ds, idvs, dv
    ds2 = ds[numpy.add(ds.dv==driver, numpy.random.rand(ds.shape[0])<0.0006)][['id']+idvs+[dv]].dropna(axis=1)
    ds2 = ds2.dropna(axis=1)
    idvs2 = list(ds2.columns)
    idvs2.remove('id')
    idvs2.remove('dv')
    
    dev = numpy.random.rand(ds2.shape[0])<=0.8
    val = ~dev
    ds2.dv[ds2.dv!=driver] = 0
    ds2.dv[ds2.dv==driver] = 1
    models = [GradientBoostingClassifier(max_depth=2, learning_rate=0.01), \
                GradientBoostingClassifier(max_depth=3, learning_rate=0.01),\
                RandomForestClassifier(max_depth=2, n_estimators = 100),\
                RandomForestClassifier(max_depth=3, n_estimators = 100),\
                RandomForestClassifier(max_depth=4, n_estimators = 100),\
                RandomForestClassifier(max_depth=5, n_estimators = 100),\
                LogisticRegression(penalty = 'l1', C = 0.01),\
                LogisticRegression(penalty='l1', C = 0.0001)]
    
    print ds2.shape, (ds.dv==driver).sum()
    
    for i, model in enumerate(models):
        model.fit(ds2[idvs2][dev],ds2[dv][dev])
        ds2['predictions'+str(i)] = model.predict_proba(ds2[idvs2])[:,1]
    
    best_i = 0
    best_val = 0
    best_dev = 0
    
    for i in xrange(len(models)):
        dev_auc = roc_auc_score(ds2[dv][dev],ds2['predictions'+str(i)][dev])
        val_auc = roc_auc_score(ds2[dv][val],ds2['predictions'+str(i)][val])
        if val_auc > best_val:
            best_val = val_auc
            best_dev = dev_auc
            best_i = i
    ds2['predictions'] = ds2['predictions'+str(best_i)]
    
    print driver, best_i, best_dev, best_val#, idvs2#, ds2[ds2.dv==1].shape
    
    if ds2[ds2.dv==1].shape[0] != 200:
        raise "Error"
    return ds2[ds2.dv==1], best_dev, best_val
    
#%%
result = Parallel(n_jobs = 4)(delayed(build_transform)(driver) for driver in sorted(ds.dv.unique()))

#%%
dev_auc = 0
val_auc = 0
count = 0
for prediction in result:
    prediction[0].to_csv(outfile,index=False,columns=['id','predictions'],header=False,mode='a')
    dev_auc += prediction[1]
    val_auc += prediction[2]
    count += 1
    
print dev_auc/count, val_auc/count

#%%
"""
dv = 'dv'
driver = 1
ds2 = ds[numpy.add(ds.dv==driver, numpy.random.rand(ds.shape[0])<0.0005)][['id']+idvs+[dv]].dropna(axis=1)
ds2 = ds2.dropna(axis=1)
idvs2 = list(ds2.columns)
idvs2.remove('id')
idvs2.remove('dv')

dev = numpy.random.rand(ds2.shape[0])<=0.7
val = ~dev
ds2.dv[ds2.dv!=driver] = 0
ds2.dv[ds2.dv==driver] = 1
from sklearn.grid_search import GridSearchCV
grid = {'n_estimators': [10,25,50,100,500,1000],
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [2,3,4,5,None],
        'min_samples_leaf': [1,2,5,10,20,50]}
model = GridSearchCV(RandomForestClassifier(),grid,cv=5,verbose=2,n_jobs =3)

model = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=5)

GradientBoostingClassifier(max_depth=2, learning_rate=0.001, n_estimators = 10000)
model.fit(ds2[idvs2][dev],ds2[dv][dev])
ds2['predictions'] = model.predict_proba(ds2[idvs2])[:,1]
dev_auc = roc_auc_score(ds2[dv][dev],ds2['predictions'][dev])
val_auc = roc_auc_score(ds2[dv][val],ds2['predictions'][val])
print dev_auc, val_auc

for values in model.grid_scores_:
    if values[1] > 0.7:
        print values[0], values[1], values[2]
"""
