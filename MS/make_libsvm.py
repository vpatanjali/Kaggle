# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:01:47 2015

@author: patanjali
"""

import os, random

DATA_DIR = '/home/patanjali/Kaggle/Data/MS/'
VAL_PCT = 0.2

infile = 'train_bytes_asm_tfs.csv'

dev_file = 'dev_bytes_asm_tfs_10_1000.libsvm'
val_file = 'val_bytes_asm_tfs_10_1000.libsvm'

#%%
os.chdir(DATA_DIR)

infile = open(infile)
dev_file = open(dev_file,'w')
val_file = open(val_file,'w')

for line in infile:
    line = line.strip().split(',')
    line = [int(x) for x in line[1:]]
    line = [str(line[0])] + [str(i)+':'+str(line[i]) for i in xrange(1,len(line)) if line[i] > 10 and line[i] < 1000]
    if random.random() < VAL_PCT:
        val_file.write(' '.join(line)+'\n')
    else:
        dev_file.write(' '.join(line)+'\n')
        
infile.close()
dev_file.close()
val_file.close()