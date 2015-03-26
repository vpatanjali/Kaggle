# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:31:18 2015

@author: patanjali
"""

import random

VAL_PCT = 0.2

infile = open('/home/patanjali/Kaggle/Data/MS/train_bytes_1000.libsvm')
devfile = open('/home/patanjali/Kaggle/Data/MS/dev_800.libsvm', 'w')
valfile = open('/home/patanjali/Kaggle/Data/MS/val_200.libsvm', 'w')

for line in infile:
    line = line.split(' ')
    line[0] = str(int(line[0])-1)
    line = ' '.join(line)
    if random.random() < VAL_PCT:
        valfile.write(line)
    else:
        devfile.write(line)
        
infile.close()
devfile.close()
valfile.close()