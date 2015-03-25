# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 17:51:45 2015

@author: patanjali
"""

# log loss calculator for the Avazu dataset

# To generate train_dv.csv run the below

#cut -d, -f2 train train_dv.csv

import pandas
import sys
from sklearn.metrics import log_loss

DEV_LIMIT = int(sys.argv[1])

#DEV_LIMIT = 36210029

actuals = pandas.read_csv('train_dv.csv')

predictions_dev = pandas.read_csv(sys.argv[2],names=['prediction'])
predictions_val = pandas.read_csv(sys.argv[3],names=['prediction'])

print log_loss(actuals['click'][:DEV_LIMIT].values,predictions_dev['prediction'].values)
print log_loss(actuals['click'][DEV_LIMIT:].values,predictions_val['prediction'].values)
