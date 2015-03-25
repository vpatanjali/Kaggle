#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 03:27:32 2015

@author: patanjali
"""

import numpy
import pandas
from sklearn.metrics import log_loss
import sys

prefix = sys.argv[1]

for _sample in ['_train','_val','_test']:
    sample = prefix + _sample
    actuals = numpy.load('data/'+sys.argv[2]+'p'+_sample+'.npz')['arr_1']
    predictions = pandas.read_csv(sample+'.csv',sep=" ",header=-1).values.copy() + 1e-32
    norm_factors = predictions.sum(1)
    predictions = (predictions.transpose()/norm_factors).transpose()
    print _sample[1:], "log loss is", log_loss(actuals,predictions)
    
