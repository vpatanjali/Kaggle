# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

prefix = '/home/patanjali/K/DSB/train/'

dim = 16

import Image, numpy

import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
import random

labels = sorted(os.listdir(prefix))

sample_size = 0

for label in labels:
    files = sorted(os.listdir(prefix + label))
    sample_size += len(files)
    
print sample_size

X = numpy.ndarray((sample_size,dim*dim))
Y = numpy.zeros((sample_size,len(labels)))

X_train = numpy.ndarray((20000,dim*dim))
Y_train = numpy.zeros((20000,len(labels)))

X_val = numpy.ndarray((5000,dim*dim))
Y_val = numpy.zeros((5000,len(labels)))

X_test = numpy.ndarray((sample_size-25000,dim*dim))
Y_test = numpy.zeros((sample_size-25000,len(labels)))

im_counter = 0

for i,label in enumerate(labels):
    files = sorted(os.listdir(prefix + label))
    print label,len(files)
    for image in files:
        X[im_counter,:] = numpy.array(Image.open(prefix+label+'/'+image).resize((dim,dim)).getdata())
        Y[im_counter,i] = 1
        im_counter += 1

order = range(im_counter)
random.shuffle(order)
im_counter = 0
for i in order:
    if im_counter < 20000:
        X_train[im_counter,:] = X[i,:]
        Y_train[im_counter,:] = Y[i,:]
    elif im_counter < 25000:            
        X_val[im_counter-20000,:] = X[i,:]
        Y_val[im_counter-20000,:] = Y[i,:]
    else:
        X_test[im_counter-25000,:] = X[i,:]
        Y_test[im_counter-25000,:] = Y[i,:]
    im_counter+=1

numpy.savez('%sp_train' %(dim),X_train,Y_train)
numpy.savez('%sp_val' %(dim),X_val,Y_val)
numpy.savez('%sp_test' %(dim),X_test,Y_test)

numpy.save('%sp_train' %(dim),X_train)
numpy.save('%sp_val' %(dim),X_val)
numpy.save('%sp_test' %(dim),X_test)

data_train = DenseDesignMatrix(X=X_train,y=Y_train)
data_val = DenseDesignMatrix(X=X_val,y=Y_val)
data_test = DenseDesignMatrix(X=X_test,y=Y_test)

serial.save('%sp_train.pkl' %(dim),data_train)
serial.save('%sp_val.pkl' %(dim),data_val)
serial.save('%sp_test.pkl' %(dim),data_test)
