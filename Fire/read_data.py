# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 09:37:47 2014

@author: patanjali
"""

data_dir = '/home/patanjali/Kaggle/Data/Fire/'

ordinal_vars = ['var1','var3','var7','var8']
categorical_vars = ['var2','var4','var5','var6','var9']

train_file = 'train.csv'
test_file = 'test.csv'

#%%

import pandas
import gc
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

#%%

os.chdir(data_dir)

train = pandas.read_csv(train_file)
test = pandas.read_csv(test_file)

#%% Handling some special variables

del train['dummy']
del test['dummy']

train['var4'][train['var4']=='Z'] = 'ZZ'
train['var4_part1'] = train['var4'].str[0]
train['var4_part2'] = train['var4'].str[1]

test['var4'][test['var4']=='Z'] = 'ZZ'
test['var4_part1'] = test['var4'].str[0]
test['var4_part2'] = test['var4'].str[1]

categorical_vars.append('var4_part1')
ordinal_vars.append('var4_part2')

#%% Categorical variables

for var in ordinal_vars:
    train[var][train[var]=='Z']=0
    test[var][test[var]=='Z']=0
    train[var] = train[var].astype('int')
    test[var] = test[var].astype('int')

train = pandas.concat([train]+[pandas.get_dummies(train[var],prefix=var)\
                                for var in categorical_vars + ordinal_vars],
                        axis=1)
                        
gc.collect()

test = pandas.concat([test]+[pandas.get_dummies(test[var],prefix=var)\
                                for var in categorical_vars + ordinal_vars],
                        axis=1)

gc.collect()

predictors = list(train.columns[train.dtypes=='float64']) + \
                list(train.columns[train.dtypes=='int64'])
predictors.remove('target')
predictors.remove('id')

for var in categorical_vars:
    del train[var]
    del test[var]

gc.collect()

#%% Defining the DVs

train['bin_dv'] = (train['target']!=0)

#%%

train['weights'] = train['var11']
test['weights'] = test['var11']

#%% Scaling variables to 0 mean unit variance

imputer = Imputer()
scaler = StandardScaler()
train[predictors] = imputer.fit_transform(train[predictors])
test[predictors] = imputer.transform(test[predictors])
gc.collect()
train[predictors] = scaler.fit_transform(train[predictors])
test[predictors] = scaler.transform(test[predictors])
gc.collect()

#%% Writing idvs to file

from pandas.io.pytables import HDFStore

store = HDFStore('idvs_v3.h5','a')

store['train'] = train
store['test'] = test
store['predictors'] = pandas.DataFrame(predictors)

store.close()
