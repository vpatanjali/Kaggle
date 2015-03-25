# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 23:31:11 2014

@author: Patanjali
"""

#%%

import pandas
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
import gc

#%%

def read_data():
    dtype_dict = {
    "id" : "object",
    "hour" : "int64",
    "C1" : "int64",
    "banner_pos" : "int64",
    "site_id" : "object",
    "site_domain" : "object",
    "site_category" : "object",
    "app_id" : "object",
    "app_domain" : "object",
    "app_category" : "object",
    "device_id" : "object",
    "device_ip" : "object",
    "device_model" : "object",
    "device_type": "int64",
    "device_conn_type" : "int64",
    "C14" : "int64",
    "C15" : "int64",
    "C16" : "int64",
    "C17" : "int64",
    "C18" : "int64",
    "C19" : "int64",
    "C20" : "int64",
    "C21" : "int64",
    }
    
    test_data = pandas.read_csv('E:/Users/Patanjali/Documents/Python Scripts/test.csv',
                                chunksize = 10000000, dtype = dtype_dict)
    train_data = pandas.read_csv('E:/Users/Patanjali/Downloads/train/train.csv', 
                                 chunksize = 10000000, dtype = dtype_dict)
    
    i = 0
    
    store = pandas.HDFStore('data.h5',mode='a')
    
    for chunk in test_data:
        i += 1
        print i
        store.append('test',chunk,data_columns=True)
    for chunk in train_data:
        i += 1
        print i
        store.append('train',chunk,data_columns=True)
    
    store.close()

def get_numeric_only():
    i = 0
    store = pandas.HDFStore('data.h5',mode='r')
    store2 = pandas.HDFStore('numeric.h5',mode='a')
    train_numeric = store.select('train',
                 columns=['hour','click','C1','banner_pos','device_type',
                          'device_conn_type','C14','Ç18','C19','C20','C21'],
                 chunksize=1000000)    
    for chunk in train_numeric:
        i += 1
        print i
        store2.append('train_numeric',chunk,data_columns=True)
        gc.collect()
    store.close()
    store2.close()

#def build_model():

#%%

store2 = pandas.HDFStore('numeric.h5',mode='r')
data = store2.select('train_numeric','hour<14102200')

#%%

def get_day_hour(data):
    data['hora'] = data.hour%100
    #data['dia'] = (data.hour//100)%100
    return data

def binarize(data):
    cat_vars = ['C1', 'banner_pos', 'device_type', 'device_conn_type','hora']
    return pandas.concat([data]+[pandas.get_dummies(data[var],prefix=var) 
                                for var in cat_vars],axis=1)
    
#%%

data = binarize(get_day_hour(data))
gc.collect()

#%% ----------- Model running from here...

idvs = list(data.columns)
idvs.remove('hour')
idvs.remove('click')

model = GradientBoostingClassifier(verbose=5, learning_rate=0.05, n_estimators = 100,
                                   max_depth=5, subsample=0.7)
model.fit(data[idvs],data['click'])
dev_out = model.predict_proba(data[idvs])[:,1]
print log_loss(data['click'],dev_out)

#%%
for day in xrange(21,31):
    gc.collect()
    val = binarize(get_day_hour(store2.select('train_numeric','hour>=1410%s00 and hour<1410%s00' %(day,day+1))))
    val_out = model.predict_proba(val[idvs])[:,1]
    print day, log_loss(val['click'],val_out)
    
#%%
store2.close()

#read_data()
#get_numeric_only()

#'id'
#'hour' - YYMMDDHH
#'C1' - 1000 + small integer
#'banner_pos' - 0 to 7
#'site_id' - 2825 unique entries
#'site_domain' - 3366 unique entries
#'site_category' - 22 unique entries
#'app_id' - 3952 unique entries
#'app_domain' - 201 unique entries
#'app_category' - 28 unique entries
#'device_id' - unique entries 6.37% of total
#'device_ip' - unique entries 23.5% of total
#'device_model' - 5438 unique entries
#'device_type' - 0 to 5 int
#'device_conn_type' - 0 to 5 int
#'C14' - 1257 unique integers
#'C15' - Looks like resolution's first dimension
#'C16' - Looks like resolution's other dimension
#'C17' - 240 unique integers
#'C18' - 0 to 3 int
#'C19' - 47 unique integers
#'C20' - 100000 + 3 digit integer, 162 unique values
#'C21' - 39 unique integers

