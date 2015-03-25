# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:08:08 2015

@author: patanjali
"""

#%%

prefix = '/home/patanjali/K/Otto/'

import numpy, pandas

import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

os.chdir(prefix)

train = pandas.read_csv('train.csv')
train['dv'] = train.target.str.slice(-1).astype('int')

idvs = list(train.columns)
idvs.remove('dv')
idvs.remove('id')
idvs.remove('target')

scaler = StandardScaler()
scaler.fit(train[idvs])

X = scaler.transform(train[idvs])

#%%

encoder = OneHotEncoder()
Y = encoder.fit_transform(pandas.DataFrame(train['dv'])).todense()

test = numpy.random.rand(X.shape[0])<0.1
dev = numpy.random.rand(X.shape[0])<0.8
val = ~dev

dev = ~(~dev+test)
val = ~(~val+test)

#%%

X_train = X[dev]; Y_train = Y[dev]
X_val = X[val]; Y_val = Y[val]
X_test = X[test]; Y_test = Y[test]

numpy.savez('train' ,X_train,Y_train)
numpy.savez('val' ,X_val,Y_val)
numpy.savez('test' ,X_test,Y_test)

numpy.save('train' ,X_train)
numpy.save('val' ,X_val)
numpy.save('test' ,X_test)

data_train = DenseDesignMatrix(X=X_train,y=Y_train)
data_val = DenseDesignMatrix(X=X_val,y=Y_val)
data_test = DenseDesignMatrix(X=X_test,y=Y_test)

serial.save('train.pkl' ,data_train)
serial.save('val.pkl' ,data_val)
serial.save('test.pkl' ,data_test)
