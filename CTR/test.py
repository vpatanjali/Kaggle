# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 19:49:36 2015

@author: Patanjali
"""
import time
from csv import DictReader

path = 'E:/Users/Patanjali/Documents/Python Scripts/test/train_1000000.csv'

start_time = time.time()

#f = DictReader(open(path))
f = open(path)
for line in f:
    line = line.split(',')
    pass

print time.time() - start_time