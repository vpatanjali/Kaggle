# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:01:47 2015

@author: patanjali
"""

import os, random

DATA_DIR = '/home/patanjali/Kaggle/Data/MS/'
VAL_PCT = 0.2

infile = 'train_bytes_asm_tfs.csv'

dev_file = 'dev_bytes_asm_ntfs_0.001_0.9.libsvm'
val_file = 'val_bytes_asm_ntfs_0.001_0.9.libsvm'

test_infiles = ['test_bytes_asm_tfs.csv', 'test_bytes_asm_tfs_part2.csv']

os.chdir(DATA_DIR)

#%%

infile = open(infile)
dev_file = open(dev_file,'w')
val_file = open(val_file,'w')

for line in infile:
    line = line.strip().split(',')
    idvs = [int(x) for x in line[2:]]
    total = sum(idvs)*1.0
    dv = line[1]
    line = [dv] + [str(i)+':'+str(idvs[i]/total) for i in xrange(len(idvs)) if idvs[i]/total > 0.001 and idvs[i]/total < 0.9]
    if random.random() < VAL_PCT:
        val_file.write(' '.join(line)+'\n')
    else:
        dev_file.write(' '.join(line)+'\n')
        
infile.close()
dev_file.close()
val_file.close()

#%%

outfile = open('test_bytes_asm_tfs.libsvm','w')

for test_file in test_infiles:
    infile = open(test_file)
    for line in infile:
        line = line.strip().split(',')
        line = [line[1]] + [str(i)+':'+line[i] for i in xrange(2,len(line)) if line[i] !='0']
        outfile.write(' '.join(line)+'\n')
        
    infile.close()

outfile.close()