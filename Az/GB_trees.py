# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:23:19 2015

@author: patanjali
"""

import os, gc
import time
import pandas
import numpy
import cPickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

data_dir = '/home/patanjali/K/Az/train_prof/'

files = sorted(os.walk(data_dir).next()[2])

models = []

dv = 'click'

losses = {}

start_time = time.time()

def updateloss(f, loss, n):
    losses[int(f[4:6])][0] += loss*n
    losses[int(f[4:6])][1] += n

for i in xrange(21,31):
    losses[i] = [0,0]


for f in files[24:25]:
    gc.collect()
    data = pandas.read_csv(data_dir + f)
    idvs = list(data.columns)
    idvs.remove(dv)
    model = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 100, warm_start = True)
    model.fit(data[idvs],data[dv])
    models.append(model)
    _loss = [log_loss(data[dv],_model.predict_proba(data[idvs])[:,1]) for _model in models]
    loss = sum(_loss)/len(_loss)
    print f, loss, min(_loss), _loss, time.time() - start_time
    updateloss(f, loss, data.shape[0])
    

for f in files[25:-24]:
    gc.collect()
    data = pandas.read_csv(data_dir + f)
    _loss = [log_loss(data[dv],_model.predict_proba(data[idvs])[:,1]) for _model in models]
    loss = sum(_loss)/len(_loss)
    print f, loss, min(_loss), _loss, time.time() - start_time
    updateloss(f, loss, data.shape[0])
    model = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 100, warm_start = True)
    #model.set_params(n_estimators = model.get_params()['n_estimators']+10)
    model.fit(data[idvs],data[dv])
    models.append(model)

for f in files[-24:]:
    data = pandas.read_csv(data_dir + f)
    _loss = [log_loss(data[dv],_model.predict_proba(data[idvs])[:,1]) for _model in models]
    loss = sum(_loss)/len(_loss)
    print f, loss, min(loss), _loss, time.time() - start_time
    updateloss(f, loss, data.shape[0])

for date in sorted(losses.keys):
    print date, losses[date][0]/losses[date][1]