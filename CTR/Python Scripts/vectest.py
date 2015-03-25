# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 18:49:29 2015

@author: Patanjali
"""
"""
import time, numpy, mmh3

st = time.time()

x = numpy.random.rand(1000000)

#y = numpy.vectorize(mmh3.hash)(x.astype('str'))

y = numpy.zeros(x.shape)
for i in xrange(x.shape[0]):
    y[i]=mmh3.hash(str(x[i]))

print time.time() - st
"""

import sys

sys.path.append('C:\\Users\\Patanjali\\Anaconda\\lib\\site-packages')

import mmh3